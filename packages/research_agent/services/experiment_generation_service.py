import copy
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sqlalchemy import select

from research_agent.domain.models.generated_experiment import GeneratedExperiment
from research_agent.domain.models.paper import Paper
from research_agent.domain.models.paper_analysis import PaperAnalysis
from research_agent.domain.models.paper_chunk import PaperChunk
from research_agent.infrastructure.db.session import SessionLocal
from research_agent.services.chunk_structure import build_chunk_structure
from research_agent.services.domain_adapters import ml_adapter
from research_agent.services.experiment_codegen_service import (
    build_dataset_code,
    build_model_code,
    build_requirements_txt,
    build_train_code,
    build_utils_code,
    detect_model_family,
    detect_task_type,
    normalize_loss_name,
    validate_generated_artifact,
)
from research_agent.services.repository_verification_service import get_repository_recommendation
from research_agent.tools.gemini_client import generate_json_with_reasoning_fallback


load_dotenv()

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
CONTROLLED_INFERENCE_RULES = (
    "You are filling in missing experiment configuration fields from a\n"
    "research paper analysis. Rules:\n"
    "- Only output the missing fields. Do not repeat fields already provided.\n"
    "- If a value is unclear, use the standard PyTorch or framework default.\n"
    "- Do NOT invent uncommon or paper-specific hyperparameters without\n"
    "  explicit evidence in the paper.\n"
    "- Prefer well-known documented defaults from PyTorch or HuggingFace.\n"
    "- If a paper value is explicitly stated, use it exactly."
)
BASE_DEFAULTS = {
    "model": {
        "name": None,
        "family": None,
        "num_layers": None,
        "hidden_dim": None,
        "num_heads": None,
        "input_dim": None,
        "output_dim": None,
        "dropout": 0.1,
    },
    "dataset": {
        "name": None,
        "task_type": None,
        "path": "./data",
        "val_split": 0.1,
        "num_workers": 4,
        "sequence_length": 256,
        "target_length": 32,
    },
    "training": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "seed": 42,
        "gradient_clip": 1.0,
        "eval_every": 1,
        "log_every": 100,
        "save_checkpoint": True,
        "checkpoint_every": 10,
        "device": "cuda",
        "gradient_accumulation_steps": 1,
        "loss": None,
        "optimizer_params": {},
    },
    "scheduler": {
        "type": "none",
        "step_size": 10,
        "gamma": 0.1,
        "warmup_steps": None,
    },
    "logging": {
        "output_dir": "./outputs",
        "checkpoint_dir": "./checkpoints",
    },
    "device": {
        "device": "cuda",
        "num_workers": 4,
        "pin_memory": True,
    },
}
OPTIMIZER_DEFAULTS = {
    "adam": {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0},
    "adamw": {"lr": 0.0001, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.01},
    "sgd": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1.0e-4},
    "rmsprop": {"lr": 0.001, "alpha": 0.99, "eps": 1.0e-8, "weight_decay": 0.0},
}


def _resolve_domain(analysis: PaperAnalysis, paper: Paper, requested_domain: str | None) -> str:
    return (
        (requested_domain or "").strip().lower()
        or str(getattr(analysis, "domain", "") or "").strip().lower()
        or str(getattr(paper, "domain", "") or "").strip().lower()
        or "general"
    )


def _build_domain_agnostic_config_from_analysis(analysis: PaperAnalysis, domain: str) -> tuple[dict, list[dict]]:
    inferred_structure = analysis.inferred_structure if isinstance(analysis.inferred_structure, dict) else {}
    method_items_value = inferred_structure.get("methods") if isinstance(inferred_structure, dict) else []
    if isinstance(method_items_value, dict):
        method_items = method_items_value.get("chunks") or []
    else:
        method_items = method_items_value or []
    result_items = inferred_structure.get("results") if isinstance(inferred_structure, dict) else []
    idea_items = inferred_structure.get("key_ideas") if isinstance(inferred_structure, dict) else []

    config = {
        "domain": domain,
        "hypothesis": (idea_items[0].get("summary") if idea_items and isinstance(idea_items[0], dict) else None)
        or "Define a falsifiable hypothesis based on the paper's central claim.",
        "method_outline": [
            str(item.get("summary", "")).strip()
            for item in method_items[:5]
            if isinstance(item, dict) and str(item.get("summary", "")).strip()
        ]
        or ["Describe the method and required components in reproducible steps."],
        "evaluation_criteria": [
            str(item.get("summary", "")).strip()
            for item in result_items[:5]
            if isinstance(item, dict) and str(item.get("summary", "")).strip()
        ]
        or ["Specify measurable outcomes and acceptance criteria."],
        "expected_outputs": [
            "replication_report.md",
            "results_table.csv",
            "runbook.md",
        ],
        "notes": {
            "paper_id": str(analysis.paper_id),
            "analysis_id": str(analysis.id),
        },
    }
    defaults_used = [
        {"field": "domain", "value": domain},
        {"field": "expected_outputs", "value": config["expected_outputs"]},
    ]
    return config, defaults_used


def _flatten_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(v) for v in value)
    return str(value)


def _append_tracking(target: list[dict], field: str, value, source: str | None = None) -> None:
    entry = {"field": field, "value": value}
    if source:
        entry["source"] = source
    target.append(entry)


def _parse_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _normalize_optimizer_name(value: str | None) -> str:
    lowered = (value or "adam").lower()
    if "adamw" in lowered:
        return "adamw"
    if "sgd" in lowered:
        return "sgd"
    if "rmsprop" in lowered:
        return "rmsprop"
    return "adam"


def _python_identifier(name: str) -> str:
    normalized = re.split(r"[:.,;]\s*", name.strip(), maxsplit=1)[0]
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", normalized).title().replace(" ", "")
    return cleaned or "GeneratedModel"


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _extract_primary_architecture(analysis: PaperAnalysis) -> str | None:
    architectures = analysis.architectures if isinstance(analysis.architectures, dict) else {}
    proposed = architectures.get("proposed", []) if isinstance(architectures, dict) else []
    if proposed:
        return str(proposed[0])
    return analysis.model_architecture


def _extract_primary_optimizer(analysis: PaperAnalysis) -> str | None:
    if isinstance(analysis.optimizers, dict) and analysis.optimizers.get("primary"):
        return str(analysis.optimizers["primary"])
    return analysis.optimizer


def _extract_primary_loss(analysis: PaperAnalysis) -> str | None:
    if isinstance(analysis.losses, dict) and analysis.losses.get("primary"):
        return str(analysis.losses["primary"])
    return analysis.loss_function


def _infer_output_dim(analysis: PaperAnalysis, training_text: str) -> int | None:
    explicit_vocab = _parse_int(training_text, r"(\d+)\s*(?:wpm|word pieces|wordpiece|vocabulary)")
    if explicit_vocab:
        return explicit_vocab
    metrics = _flatten_text(analysis.evaluation_metrics)
    architecture = (_extract_primary_architecture(analysis) or "").lower()
    if "speech" in metrics.lower() or "asr" in architecture:
        return None
    return 10


def _build_experiment_loss_confirmation_prompt(analysis: PaperAnalysis, current_loss: str | None) -> str:
    return (
        "You are preparing a replication experiment from a research paper analysis.\n"
        "Confirm the most likely primary training loss conservatively.\n"
        "If the evidence is ambiguous, return Unknown instead of guessing.\n\n"
        f"Architecture: {_extract_primary_architecture(analysis) or 'Unknown'}\n"
        f"Dataset: {analysis.dataset or 'Unknown'}\n"
        f"Training objective: {getattr(analysis, 'training_objective', None) or 'Unknown'}\n"
        f"Training details: {json.dumps(analysis.training_details, default=str)}\n"
        "A previous pass may have guessed a loss already. Do not anchor on that guess unless the evidence independently supports it.\n\n"
        "Return JSON with keys:\n"
        "- loss: one of [CTC, RNNT, CrossEntropy, BCEWithLogits, MSE, Unknown]\n"
        "- confidence: float between 0 and 1\n"
        "- justification: short string\n"
        "- explicitly_stated: boolean"
    )


def _confirm_experiment_loss(analysis: PaperAnalysis, current_loss: str | None) -> tuple[str | None, dict | None]:
    losses = analysis.losses if isinstance(analysis.losses, dict) else {}
    if current_loss and not bool(losses.get("inferred", False)):
        return normalize_loss_name(current_loss), None
    try:
        confirmation = generate_json_with_reasoning_fallback(_build_experiment_loss_confirmation_prompt(analysis, current_loss))
    except Exception:
        logger.exception("experiment_loss_confirmation_failed paper_analysis_id=%s", getattr(analysis, "id", None))
        return normalize_loss_name(current_loss) if current_loss else None, None

    candidate = str(confirmation.get("loss") or "").strip()
    confidence = confirmation.get("confidence")
    explicitly_stated = bool(confirmation.get("explicitly_stated", False))
    if not candidate or candidate.lower() == "unknown":
        return normalize_loss_name(current_loss) if current_loss else None, {
            "field": "training.loss_confirmation",
            "value": confirmation,
            "source": "LLM",
        }
    if explicitly_stated or float(confidence or 0.0) >= 0.8:
        return normalize_loss_name(candidate), {
            "field": "training.loss_confirmation",
            "value": confirmation,
            "source": "LLM",
        }
    return normalize_loss_name(current_loss) if current_loss else None, {
        "field": "training.loss_confirmation",
        "value": confirmation,
        "source": "LLM",
    }


def _build_config_from_analysis(analysis: PaperAnalysis) -> tuple[dict, list[dict]]:
    config = copy.deepcopy(BASE_DEFAULTS)
    defaults_used: list[dict] = []
    training_text = _flatten_text(analysis.training_details)
    architecture_name = _extract_primary_architecture(analysis)
    task_type = detect_task_type(analysis.dataset, getattr(analysis, "training_objective", None), architecture_name)
    model_family = detect_model_family(architecture_name, analysis.model_architecture)

    config["model"]["name"] = architecture_name
    config["model"]["family"] = model_family
    config["dataset"]["name"] = analysis.dataset
    config["dataset"]["task_type"] = task_type
    config["training"]["optimizer"] = _normalize_optimizer_name(_extract_primary_optimizer(analysis))
    confirmed_loss, loss_confirmation = _confirm_experiment_loss(analysis, _extract_primary_loss(analysis))
    config["training"]["loss"] = confirmed_loss

    explicit_learning_rate = _parse_float(training_text, r"(?:learning rate|lr)\s*[=:]?\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)")
    explicit_batch_size = _parse_int(training_text, r"batch size\s*[=:]?\s*(\d+)")
    explicit_epochs = _parse_int(training_text, r"(\d+)\s+epochs?")
    explicit_warmup = _parse_int(training_text, r"(\d+)\s*(?:warm[- ]?up)\s*steps")
    explicit_num_layers = _parse_int(training_text, r"(\d+)\s*(?:layer|layers)")
    explicit_hidden_dim = _parse_int(training_text, r"(?:hidden dim|model dimension|d_model|hidden size)\s*[=:]?\s*(\d+)")
    explicit_num_heads = _parse_int(training_text, r"(\d+)\s*(?:attention heads|heads)")
    explicit_input_dim = _parse_int(training_text, r"(?:input dim|feature dim|mel bins)\s*[=:]?\s*(\d+)")

    config["model"]["num_layers"] = explicit_num_layers
    config["model"]["hidden_dim"] = explicit_hidden_dim
    config["model"]["num_heads"] = explicit_num_heads
    config["model"]["input_dim"] = explicit_input_dim
    config["model"]["output_dim"] = _infer_output_dim(analysis, training_text)

    optimizer_name = config["training"]["optimizer"]
    config["training"]["learning_rate"] = explicit_learning_rate or OPTIMIZER_DEFAULTS[optimizer_name]["lr"]
    config["training"]["batch_size"] = explicit_batch_size or BASE_DEFAULTS["training"]["batch_size"]
    config["training"]["epochs"] = explicit_epochs or BASE_DEFAULTS["training"]["epochs"]
    config["training"]["optimizer_params"] = copy.deepcopy(OPTIMIZER_DEFAULTS[optimizer_name])
    config["scheduler"]["warmup_steps"] = explicit_warmup
    if "cosine" in training_text.lower():
        config["scheduler"]["type"] = "cosine"
    elif "step" in training_text.lower() and "schedule" in training_text.lower():
        config["scheduler"]["type"] = "steplr"

    if task_type == "speech_asr":
        config["dataset"]["target_length"] = 32

    for field_path, value in [
        ("model.name", config["model"]["name"]),
        ("model.family", config["model"]["family"]),
        ("model.num_layers", config["model"]["num_layers"]),
        ("model.hidden_dim", config["model"]["hidden_dim"]),
        ("model.num_heads", config["model"]["num_heads"]),
        ("model.input_dim", config["model"]["input_dim"]),
        ("model.output_dim", config["model"]["output_dim"]),
        ("model.dropout", config["model"]["dropout"]),
        ("dataset.name", config["dataset"]["name"]),
        ("dataset.task_type", config["dataset"]["task_type"]),
        ("dataset.path", config["dataset"]["path"]),
        ("dataset.val_split", config["dataset"]["val_split"]),
        ("dataset.num_workers", config["dataset"]["num_workers"]),
        ("dataset.sequence_length", config["dataset"]["sequence_length"]),
        ("dataset.target_length", config["dataset"]["target_length"]),
        ("training.optimizer", config["training"]["optimizer"]),
        ("training.learning_rate", config["training"]["learning_rate"]),
        ("training.batch_size", config["training"]["batch_size"]),
        ("training.epochs", config["training"]["epochs"]),
        ("training.seed", config["training"]["seed"]),
        ("training.gradient_clip", config["training"]["gradient_clip"]),
        ("training.eval_every", config["training"]["eval_every"]),
        ("training.log_every", config["training"]["log_every"]),
        ("training.save_checkpoint", config["training"]["save_checkpoint"]),
        ("training.checkpoint_every", config["training"]["checkpoint_every"]),
        ("training.device", config["training"]["device"]),
        ("training.gradient_accumulation_steps", config["training"]["gradient_accumulation_steps"]),
        ("training.loss", config["training"]["loss"]),
        ("scheduler.type", config["scheduler"]["type"]),
        ("scheduler.step_size", config["scheduler"]["step_size"]),
        ("scheduler.gamma", config["scheduler"]["gamma"]),
        ("scheduler.warmup_steps", config["scheduler"]["warmup_steps"]),
        ("logging.output_dir", config["logging"]["output_dir"]),
        ("logging.checkpoint_dir", config["logging"]["checkpoint_dir"]),
        ("device.device", config["device"]["device"]),
        ("device.num_workers", config["device"]["num_workers"]),
        ("device.pin_memory", config["device"]["pin_memory"]),
    ]:
        _append_tracking(defaults_used, field_path, value)

    if loss_confirmation is not None:
        defaults_used.append(loss_confirmation)

    for key, value in config["training"]["optimizer_params"].items():
        _append_tracking(defaults_used, f"training.optimizer_params.{key}", value)

    return config, defaults_used


def load_analysis_step(state: dict) -> dict:
    logger.info("[load_analysis] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    db = state["db"]
    paper_id = uuid.UUID(state["paper_id"])
    analysis = (
        db.execute(
            select(PaperAnalysis)
            .where(PaperAnalysis.paper_id == paper_id)
            .order_by(PaperAnalysis.created_at.desc())
        )
        .scalars()
        .first()
    )
    if analysis is None:
        raise ValueError(f"No paper_analysis found for paper_id={state['paper_id']}")

    paper = db.get(Paper, paper_id)
    if paper is None:
        raise ValueError(f"Paper not found for paper_id={state['paper_id']}")

    domain = _resolve_domain(analysis, paper, state.get("domain"))
    if domain == "ml" and not analysis.model_architecture:
        chunks = db.execute(
            select(PaperChunk)
            .where(PaperChunk.paper_id == paper_id)
            .order_by(PaperChunk.chunk_index)
        ).scalars().all()
        chunk_payloads = [
            {
                **build_chunk_structure(
                    chunk_id=str(chunk.id),
                    chunk_index=chunk.chunk_index,
                    section_name=chunk.section_name,
                    subsection_name=chunk.subsection_name,
                    content=chunk.content,
                    total_chunks=max(1, len(chunks)),
                ),
                "content": chunk.content,
            }
            for chunk in chunks
        ]
        ml_view = ml_adapter(
            chunk_payloads,
            inferred_structure=analysis.inferred_structure if isinstance(analysis.inferred_structure, dict) else None,
        )
        analysis.model_architecture = analysis.model_architecture or ml_view.get("model_architecture")
        analysis.architectures = analysis.architectures or ml_view.get("architectures")
        analysis.dataset = analysis.dataset or ml_view.get("dataset")
        analysis.loss_function = analysis.loss_function or ml_view.get("loss_function")
        analysis.losses = analysis.losses or ml_view.get("losses")
        analysis.training_objective = analysis.training_objective or ml_view.get("training_objective")
        analysis.optimizer = analysis.optimizer or ml_view.get("optimizer")
        analysis.optimizers = analysis.optimizers or ml_view.get("optimizers")
        analysis.training_details = analysis.training_details or ml_view.get("training_details")
        analysis.evaluation_metrics = analysis.evaluation_metrics or ml_view.get("evaluation_metrics")
        analysis.contributions = analysis.contributions or ml_view.get("contributions")

    logger.info(
        "[load_analysis] paper_id=%s experiment_id=%s domain=%s status=exit",
        state["paper_id"],
        state["experiment_id"],
        domain,
    )
    return {**state, "analysis": analysis, "paper": paper, "domain": domain}


def verify_repositories_step(state: dict) -> dict:
    logger.info("[verify_repositories] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    paper_id = uuid.UUID(state["paper_id"])
    repository_recommendation = get_repository_recommendation(
        db=state["db"],
        paper_id=paper_id,
        paper=state["paper"],
        analysis=state["analysis"],
    )
    logger.info(
        "[verify_repositories] paper_id=%s experiment_id=%s primary_repo=%s recommended_action=%s status=exit",
        state["paper_id"],
        state["experiment_id"],
        repository_recommendation.get("primary_repo"),
        repository_recommendation.get("recommended_action"),
    )
    return {**state, "repository_recommendation": repository_recommendation}


def apply_defaults_step(state: dict) -> dict:
    logger.info("[apply_defaults] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    if state.get("domain") == "ml":
        config, defaults_used = _build_config_from_analysis(state["analysis"])
    else:
        config, defaults_used = _build_domain_agnostic_config_from_analysis(
            state["analysis"], state.get("domain", "general")
        )
    logger.info("[apply_defaults] paper_id=%s experiment_id=%s status=exit", state["paper_id"], state["experiment_id"])
    return {
        **state,
        "config": config,
        "defaults_used": defaults_used,
        "inferred_fields": [],
        "validation": {"errors": [], "warnings": []},
    }


def infer_missing_fields_step(state: dict) -> dict:
    logger.info("[infer_missing_fields] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    if state.get("domain") != "ml":
        logger.info(
            "[infer_missing_fields] paper_id=%s experiment_id=%s skipped_for_domain=%s",
            state["paper_id"],
            state["experiment_id"],
            state.get("domain"),
        )
        return state

    analysis = state["analysis"]
    config = copy.deepcopy(state["config"])
    inferred_fields = list(state["inferred_fields"])
    defaults_used = list(state["defaults_used"])

    missing_fields = []
    for path in [
        "model.num_layers",
        "model.hidden_dim",
        "model.num_heads",
        "model.input_dim",
        "model.output_dim",
    ]:
        section, key = path.split(".", 1)
        if config[section].get(key) is None:
            missing_fields.append(path)

    if missing_fields:
        prompt = (
            f"{CONTROLLED_INFERENCE_RULES}\n\n"
            "Return JSON only.\n"
            f"Paper analysis:\n{json.dumps({'model_architecture': analysis.model_architecture, 'architectures': analysis.architectures, 'dataset': analysis.dataset, 'losses': analysis.losses, 'optimizers': analysis.optimizers, 'training_details': analysis.training_details, 'training_objective': getattr(analysis, 'training_objective', None)}, default=str)}\n\n"
            f"Current config:\n{json.dumps(config, default=str)}\n\n"
            f"Missing fields:\n{missing_fields}"
        )
        try:
            inferred = generate_json_with_reasoning_fallback(prompt)
        except Exception:
            logger.exception("[infer_missing_fields] paper_id=%s experiment_id=%s llm_failed", state["paper_id"], state["experiment_id"])
            inferred = {}

        for section_name in ["model", "training", "scheduler", "dataset"]:
            updates = inferred.get(section_name, {}) if isinstance(inferred.get(section_name), dict) else {}
            for key, value in updates.items():
                if value in (None, "", [], {}):
                    continue
                if section_name == "training" and key == "loss":
                    value = normalize_loss_name(str(value))
                config[section_name][key] = value
                _append_tracking(inferred_fields, f"{section_name}.{key}", value, "LLM")
                logger.info(
                    "[infer_missing_fields] paper_id=%s experiment_id=%s field=%s value=%s",
                    state["paper_id"],
                    state["experiment_id"],
                    f"{section_name}.{key}",
                    value,
                )

        for field in missing_fields:
            section_name, key = field.split(".", 1)
            if config[section_name].get(key) is None:
                _append_tracking(inferred_fields, field, None, "LLM")

    if config["training"]["loss"] in (None, ""):
        config["training"]["loss"] = "cross_entropy"
        _append_tracking(defaults_used, "training.loss", "cross_entropy")
    else:
        config["training"]["loss"] = normalize_loss_name(str(config["training"]["loss"]))

    logger.info("[infer_missing_fields] paper_id=%s experiment_id=%s status=exit", state["paper_id"], state["experiment_id"])
    return {**state, "config": config, "defaults_used": defaults_used, "inferred_fields": inferred_fields}


def generate_config_step(state: dict) -> dict:
    logger.info("[generate_config] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    config = copy.deepcopy(state["config"])
    config_yaml = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    logger.info("[generate_config] paper_id=%s experiment_id=%s status=exit", state["paper_id"], state["experiment_id"])
    return {**state, "config": config, "config_yaml": config_yaml}


def generate_code_step(state: dict) -> dict:
    logger.info("[generate_code] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    if state.get("domain") != "ml":
        domain = state.get("domain", "general")
        config = state["config"]
        generic_plan = (
            "from pathlib import Path\n\n"
            "def main() -> None:\n"
            "    Path('replication_report.md').write_text('# Replication Report\\n\\nFill in observations and evidence.\\n', encoding='utf-8')\n"
            "    Path('runbook.md').write_text('# Runbook\\n\\nDocument setup, commands, and environment details.\\n', encoding='utf-8')\n"
            "    Path('results_table.csv').write_text('metric,value\\n', encoding='utf-8')\n"
            "    print('Base scaffold generated for domain:', {!r})\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ).format(domain)
        placeholder_model = (
            "class DomainScaffold:\n"
            "    def __init__(self, config: dict) -> None:\n"
            "        self.config = config\n\n"
            "    def summarize(self) -> str:\n"
            "        return f\"Domain scaffold for {self.config.get('domain', 'general')}\"\n"
        )
        dataset_code = (
            "def load_artifacts() -> dict:\n"
            "    return {\n"
            "        'hypothesis': None,\n"
            "        'method_outline': [],\n"
            "        'evaluation_criteria': [],\n"
            "    }\n"
        )
        utils_code = build_utils_code()
        requirements_txt = "pyyaml>=6.0\n"
        return {
            **state,
            "model_code": placeholder_model,
            "dataset_code": dataset_code,
            "train_code": generic_plan,
            "utils_code": utils_code,
            "requirements_txt": requirements_txt,
            "config": config,
        }

    config = state["config"]
    model_class = _python_identifier(config["model"]["name"] or "generated_model")
    model_family = config["model"]["family"] or "mlp"
    task_type = config["dataset"]["task_type"] or "generic_supervised"

    model_code = build_model_code(model_class=model_class, model_family=model_family, task_type=task_type)
    dataset_code = build_dataset_code(task_type=task_type)
    utils_code = build_utils_code()
    train_code = build_train_code(model_class=model_class, task_type=task_type)
    requirements_txt = build_requirements_txt()

    logger.info(
        "[generate_code] paper_id=%s experiment_id=%s model_family=%s task_type=%s status=exit",
        state["paper_id"],
        state["experiment_id"],
        model_family,
        task_type,
    )
    return {
        **state,
        "model_code": model_code,
        "dataset_code": dataset_code,
        "train_code": train_code,
        "utils_code": utils_code,
        "requirements_txt": requirements_txt,
    }


def write_files_step(state: dict) -> dict:
    logger.info("[write_files] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    artifact_path = Path(ARTIFACTS_DIR).resolve() / "experiments" / str(state["paper_id"]) / state["experiment_id"]
    os.makedirs(artifact_path / "utils", exist_ok=True)

    _write_text_atomic(artifact_path / "config.yaml", state["config_yaml"])
    _write_text_atomic(artifact_path / "model.py", state["model_code"])
    _write_text_atomic(artifact_path / "dataset.py", state["dataset_code"])
    _write_text_atomic(artifact_path / "train.py", state["train_code"])
    _write_text_atomic(artifact_path / "utils" / "config_loader.py", state["utils_code"])
    _write_text_atomic(artifact_path / "requirements.txt", state["requirements_txt"])

    logger.info(
        "[write_files] paper_id=%s experiment_id=%s artifact_path=%s status=exit",
        state["paper_id"],
        state["experiment_id"],
        str(artifact_path),
    )
    return {**state, "artifact_path": str(artifact_path)}


def validate_artifact_step(state: dict) -> dict:
    logger.info("[validate_artifact] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    if state.get("domain") != "ml":
        validation = {"errors": [], "warnings": ["Generated base domain-agnostic scaffold."]}
        return {**state, "validation": validation}

    validation = validate_generated_artifact(
        config_yaml=state["config_yaml"],
        model_code=state["model_code"],
        dataset_code=state["dataset_code"],
        train_code=state["train_code"],
        utils_code=state["utils_code"],
    )
    if validation["errors"]:
        raise ValueError("; ".join(validation["errors"]))
    logger.info(
        "[validate_artifact] paper_id=%s experiment_id=%s warnings=%s status=exit",
        state["paper_id"],
        state["experiment_id"],
        validation["warnings"],
    )
    return {**state, "validation": validation}


def store_metadata_step(state: dict) -> dict:
    logger.info("[store_metadata] paper_id=%s experiment_id=%s status=enter", state["paper_id"], state["experiment_id"])
    metadata = {
        "paper_id": str(state["paper_id"]),
        "experiment_id": state["experiment_id"],
        "domain": state.get("domain", "general"),
        "architecture": ((state["config"].get("model") or {}).get("name") if isinstance(state.get("config"), dict) else None),
        "dataset": ((state["config"].get("dataset") or {}).get("name") if isinstance(state.get("config"), dict) else None),
        "loss": ((state["config"].get("training") or {}).get("loss") if isinstance(state.get("config"), dict) else None),
        "optimizer": ((state["config"].get("training") or {}).get("optimizer") if isinstance(state.get("config"), dict) else None),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inferred_fields": state["inferred_fields"],
        "defaults_used": state["defaults_used"],
        "repository_recommendation": state.get("repository_recommendation"),
        "validation": state.get("validation"),
        "recommended_action": (state.get("repository_recommendation") or {}).get("recommended_action", "use_generated_scaffold"),
    }
    _write_text_atomic(Path(state["artifact_path"]) / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))

    db = state["db"]
    record = db.get(GeneratedExperiment, state["experiment_id"])
    if record is None:
        raise ValueError(f"GeneratedExperiment record missing for {state['experiment_id']}")
    record.artifact_path = state["artifact_path"]
    record.generation_status = "completed"
    record.error_message = None
    record.model_code = state["model_code"]
    record.train_code = state["train_code"]
    record.dataset_code = state["dataset_code"]
    record.config_yaml = state["config_yaml"]
    db.commit()
    logger.info(
        "[store_metadata] paper_id=%s experiment_id=%s artifact_path=%s status=exit",
        state["paper_id"],
        state["experiment_id"],
        state["artifact_path"],
    )
    return {**state, "metadata": metadata, "generation_status": "completed"}


def generate_experiment(paper_id: str, domain: str | None = None) -> dict:
    experiment_id = str(uuid.uuid4())
    paper_uuid = uuid.UUID(paper_id)
    artifact_path = str(Path(ARTIFACTS_DIR).resolve() / "experiments" / paper_id / experiment_id)

    with SessionLocal() as db:
        record = GeneratedExperiment(
            id=experiment_id,
            paper_id=paper_uuid,
            artifact_path=artifact_path,
            generation_status="pending",
            error_message=None,
        )
        db.add(record)
        db.commit()

        from research_agent.agents.graphs.experiment_generation_graph import build_experiment_generation_graph

        graph = build_experiment_generation_graph()
        state = {
            "paper_id": paper_id,
            "experiment_id": experiment_id,
            "artifact_path": artifact_path,
            "generation_status": "pending",
            "domain": domain,
            "db": db,
            "config": {},
            "defaults_used": [],
            "inferred_fields": [],
            "validation": {"errors": [], "warnings": []},
        }

        try:
            final_state = graph.invoke(state)
            repository_recommendation = final_state.get("repository_recommendation") or {}
            return {
                "experiment_id": experiment_id,
                "artifact_path": final_state["artifact_path"],
                "generation_status": final_state["generation_status"],
                "recommended_action": repository_recommendation.get("recommended_action", "use_generated_scaffold"),
                "primary_repo": repository_recommendation.get("primary_repo"),
                "repositories": repository_recommendation.get("repositories", []),
                "validation": final_state.get("validation", {}),
            }
        except Exception as exc:
            db.rollback()
            failed_record = db.get(GeneratedExperiment, experiment_id)
            if failed_record is not None:
                failed_record.generation_status = "failed"
                failed_record.error_message = str(exc)
                db.commit()
            logger.error(
                "[generate_experiment] paper_id=%s experiment_id=%s artifact_path=%s generation_status=failed error=%s",
                paper_id,
                experiment_id,
                artifact_path,
                str(exc),
            )
            return {
                "experiment_id": experiment_id,
                "artifact_path": artifact_path,
                "generation_status": "failed",
                "error_message": str(exc),
            }
