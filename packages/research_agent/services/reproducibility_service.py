import json
import logging
import re
from collections.abc import Mapping

from research_agent.tools.gemini_client import generate_json


logger = logging.getLogger(__name__)

KNOWN_PUBLIC_DATASETS = {
    "librispeech",
    "imagenet",
    "coco",
    "cifar",
    "mnist",
    "wmt",
    "squad",
}
EXPECTED_HYPERPARAMETERS = ["learning_rate", "batch_size", "optimizer", "epochs", "scheduler"]


def _flatten_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _contains_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def _dataset_availability(dataset) -> bool | None:
    dataset_text = _flatten_text(dataset).strip()
    if not dataset_text:
        return None
    lowered = dataset_text.lower()
    return any(name in lowered for name in KNOWN_PUBLIC_DATASETS)


def _ml_hyperparameter_completeness(paper_analysis: dict) -> tuple[float, dict]:
    training_text = _flatten_text(paper_analysis.get("training_details"))
    optimizer_text = _flatten_text(paper_analysis.get("optimizers") or paper_analysis.get("optimizer"))
    detected = {
        "learning_rate": _contains_any(training_text, ["learning rate", "lr ", "warm-up", "peak learning rate"]),
        "batch_size": _contains_any(training_text, ["batch size", "batchsize"]),
        "optimizer": bool(optimizer_text.strip()),
        "epochs": _contains_any(training_text, ["epoch", "epochs", "steps"]),
        "scheduler": _contains_any(training_text, ["schedule", "scheduler", "warm-up", "decay"]),
    }
    score = sum(1 for value in detected.values() if value) / len(EXPECTED_HYPERPARAMETERS)
    return round(score, 4), detected


def _ml_training_detail_score(paper_analysis: dict) -> tuple[float, dict]:
    training_text = _flatten_text(paper_analysis.get("training_details"))
    dataset_text = _flatten_text(paper_analysis.get("dataset"))
    signals = {
        "optimizer_detected": bool(_flatten_text(paper_analysis.get("optimizers") or paper_analysis.get("optimizer")).strip()),
        "loss_detected": bool(_flatten_text(paper_analysis.get("losses") or paper_analysis.get("loss_function")).strip()),
        "training_pipeline_described": _contains_any(training_text, ["train", "optimizer", "regularization", "schedule"]),
        "data_splits_mentioned": _contains_any(dataset_text + " " + training_text, ["train", "dev", "validation", "test", "split"]),
        "augmentation_described": _contains_any(training_text, ["augment", "specaugment", "crop", "mask", "flip"]),
    }
    score = sum(1 for value in signals.values() if value) / len(signals)
    return round(score, 4), signals


def _ml_evaluation_protocol_score(paper_analysis: dict) -> tuple[float, dict]:
    metrics_text = _flatten_text(paper_analysis.get("evaluation_metrics"))
    contributions_text = _flatten_text(paper_analysis.get("contributions"))
    dataset_text = _flatten_text(paper_analysis.get("dataset"))
    architecture_text = _flatten_text(paper_analysis.get("architectures"))
    signals = {
        "metrics_present": bool(metrics_text.strip()),
        "test_dataset_specified": _contains_any(dataset_text + " " + contributions_text, ["test", "benchmark", "librispeech", "imagenet", "coco", "wmt", "squad"]),
        "comparison_baselines_present": _contains_any(architecture_text + " " + contributions_text, ["baseline", "compare", "outperform", "transformer", "contextnet", "cnn"]),
        "benchmark_dataset_mentioned": _contains_any(dataset_text, list(KNOWN_PUBLIC_DATASETS)),
    }
    score = sum(1 for value in signals.values() if value) / len(signals)
    return round(score, 4), signals


def _artifact_availability_score(repository_info: dict, paper_analysis: dict) -> tuple[float, dict]:
    code_available = bool(repository_info.get("primary_repo"))
    dataset_available = _dataset_availability(paper_analysis.get("dataset"))
    artifact_score = (
        (1.0 if code_available else 0.0) * 0.6
        + (1.0 if dataset_available is True else 0.5 if dataset_available is None else 0.0) * 0.4
    )
    return round(artifact_score, 4), {
        "code_available": code_available,
        "dataset_available": dataset_available,
    }


def _methodology_completeness_score(paper_analysis: dict) -> tuple[float, dict]:
    inferred_structure = paper_analysis.get("inferred_structure") if isinstance(paper_analysis, dict) else None
    methods = []
    key_ideas = []
    discussion = []
    if isinstance(inferred_structure, Mapping):
        methods_value = inferred_structure.get("methods")
        if isinstance(methods_value, Mapping):
            methods = methods_value.get("chunks") or []
        else:
            methods = methods_value or []
        key_ideas = inferred_structure.get("key_ideas") or []
        discussion = inferred_structure.get("discussion") or []

    training_text = _flatten_text(paper_analysis.get("training_details"))
    signals = {
        "problem_or_idea_present": bool(key_ideas),
        "method_present": bool(methods) or _contains_any(training_text, ["method", "algorithm", "pipeline", "system"]),
        "implementation_clues": _contains_any(training_text, ["implementation", "module", "component", "setup"]),
        "limitations_or_discussion": bool(discussion) or _contains_any(training_text, ["limitation", "threat", "future work"]),
    }
    score = sum(1 for value in signals.values() if value) / len(signals)
    return round(score, 4), signals


def _result_reproducibility_score(paper_analysis: dict, context: str) -> tuple[float, dict]:
    evaluation_text = _flatten_text(paper_analysis.get("evaluation_metrics"))
    results_text = f"{_flatten_text(paper_analysis.get('contributions'))} {context[:2500]}".lower()
    numeric_mentions = len(re.findall(r"\b\d+(\.\d+)?%?\b", results_text))
    signals = {
        "metrics_present": bool(evaluation_text.strip()),
        "comparisons_present": _contains_any(results_text, ["baseline", "compare", "improve", "outperform"]),
        "quantitative_results_present": numeric_mentions >= 3,
        "benchmark_or_workload_present": _contains_any(results_text, ["benchmark", "dataset", "workload", "testbed"]),
    }
    score = sum(1 for value in signals.values() if value) / len(signals)
    return round(score, 4), signals


def infer_reproducibility_details(context: str, paper_analysis: dict) -> dict:
    prompt = (
        "The following information was extracted from a research paper:\n\n"
        f"Architecture: {_flatten_text(paper_analysis.get('architectures') or paper_analysis.get('model_architecture'))}\n"
        f"Dataset: {_flatten_text(paper_analysis.get('dataset'))}\n"
        f"Optimizer: {_flatten_text(paper_analysis.get('optimizers') or paper_analysis.get('optimizer'))}\n"
        f"Loss: {_flatten_text(paper_analysis.get('losses') or paper_analysis.get('loss_function'))}\n\n"
        "Based on the paper context, determine:\n"
        "1. whether the training setup appears reproducible\n"
        "2. whether hyperparameters are sufficiently described\n"
        "3. whether implementation would likely require missing details\n\n"
        "Return JSON with keys:\n"
        "- training_setup_reproducible: bool\n"
        "- hyperparameters_sufficient: bool\n"
        "- missing_details_likely: bool\n"
        "- confidence: float\n"
        "- summary: string\n\n"
        f"Context:\n{context[:5000]}"
    )
    try:
        return generate_json(prompt)
    except Exception:
        logger.exception("reproducibility_reasoning_failed")
        return {}


def compute_reproducibility_score(
    paper_analysis: dict,
    repository_info: dict,
    context: str,
    domain: str | None = None,
) -> dict:
    normalized_domain = (domain or paper_analysis.get("domain") or "general").lower()

    artifact_score, artifact_signals = _artifact_availability_score(repository_info, paper_analysis)
    methodology_score, methodology_signals = _methodology_completeness_score(paper_analysis)
    results_score, results_signals = _result_reproducibility_score(paper_analysis, context)

    dataset_available = artifact_signals["dataset_available"]
    code_available = bool(artifact_signals["code_available"])

    hyperparameter_completeness = methodology_score
    training_detail_score = methodology_score
    evaluation_protocol_score = results_score

    ml_plugin_signals: dict = {}
    reasoning = {}

    if normalized_domain == "ml":
        ml_hyper, ml_hyper_signals = _ml_hyperparameter_completeness(paper_analysis)
        ml_training, ml_training_signals = _ml_training_detail_score(paper_analysis)
        ml_eval, ml_eval_signals = _ml_evaluation_protocol_score(paper_analysis)
        ml_plugin_signals = {
            "hyperparameter_signals": ml_hyper_signals,
            "training_signals": ml_training_signals,
            "evaluation_signals": ml_eval_signals,
        }

        hyperparameter_completeness = round(0.7 * ml_hyper + 0.3 * methodology_score, 4)
        training_detail_score = round(0.7 * ml_training + 0.3 * methodology_score, 4)
        evaluation_protocol_score = round(0.6 * ml_eval + 0.4 * results_score, 4)

        if hyperparameter_completeness < 0.6 or training_detail_score < 0.6:
            reasoning = infer_reproducibility_details(context, paper_analysis)
            if reasoning.get("hyperparameters_sufficient") is True:
                hyperparameter_completeness = max(hyperparameter_completeness, 0.65)
            if reasoning.get("training_setup_reproducible") is True:
                training_detail_score = max(training_detail_score, 0.7)

    dataset_score = 1.0 if dataset_available is True else 0.5 if dataset_available is None else 0.0
    code_score = 1.0 if code_available else 0.0
    overall_score = round(
        dataset_score * 0.2
        + code_score * 0.25
        + hyperparameter_completeness * 0.2
        + training_detail_score * 0.2
        + evaluation_protocol_score * 0.15,
        4,
    )

    summary_parts = []
    summary_parts.append(
        "Dataset appears public." if dataset_available is True else
        "Dataset availability is unclear." if dataset_available is None else
        "Dataset appears unavailable."
    )
    summary_parts.append("Code repository found." if code_available else "No likely code repository found.")
    summary_parts.append(
        f"Hyperparameter completeness is {hyperparameter_completeness:.2f}, training detail score is {training_detail_score:.2f}, "
        f"and evaluation protocol score is {evaluation_protocol_score:.2f}."
    )
    summary_parts.append(f"Domain interpreted as {normalized_domain}.")
    if reasoning.get("summary"):
        summary_parts.append(str(reasoning["summary"]))

    return {
        "dataset_available": dataset_available,
        "code_available": code_available,
        "hyperparameter_completeness": hyperparameter_completeness,
        "training_detail_score": training_detail_score,
        "evaluation_protocol_score": evaluation_protocol_score,
        "overall_score": overall_score,
        "summary": " ".join(summary_parts),
        "evidence": {
            "base_signals": {
                "artifact": artifact_signals,
                "methodology": methodology_signals,
                "results": results_signals,
            },
            "ml_plugin_signals": ml_plugin_signals,
            "repository_info": repository_info,
            "artifact_score": artifact_score,
            "domain": normalized_domain,
            "reasoning": reasoning,
        },
    }


def format_reproducibility_answer(score_record) -> str:
    dataset_value = score_record.dataset_available
    dataset_text = (
        "available"
        if dataset_value is True
        else "unknown"
        if dataset_value is None
        else "not publicly confirmed"
    )
    return (
        f"Overall reproducibility score: {score_record.overall_score:.2f}. "
        f"Dataset availability: {dataset_text}. "
        f"Code available: {'yes' if score_record.code_available else 'no'}. "
        f"Hyperparameter completeness: {score_record.hyperparameter_completeness:.2f}. "
        f"Training detail score: {score_record.training_detail_score:.2f}. "
        f"Evaluation protocol score: {score_record.evaluation_protocol_score:.2f}. "
        f"{score_record.summary or ''}".strip()
    )
