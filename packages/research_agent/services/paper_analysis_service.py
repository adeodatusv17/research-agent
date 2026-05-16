import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from research_agent.domain.models.paper import Paper
from research_agent.domain.models.paper_analysis import PaperAnalysis
from research_agent.domain.models.paper_chunk import PaperChunk
from research_agent.domain.models.paper_repository import PaperRepository
from research_agent.domain.models.paper_section import PaperSection
from research_agent.domain.models.reproducibility_score import ReproducibilityScore
from research_agent.infrastructure.db.session import SessionLocal
from research_agent.services.chunk_structure import (
    build_chunk_structure,
    build_inferred_structure,
    is_quality_chunk,
)
from research_agent.services.domain_adapters import derive_domain_fields, ml_adapter
from research_agent.services.domain_detector import detect_domain
from research_agent.services.repository_discovery_service import discover_repositories
from research_agent.services.reproducibility_service import (
    compute_reproducibility_score,
    format_reproducibility_answer,
)
from research_agent.tools.embedder import generate_embedding, get_tokenizer
from research_agent.tools.gemini_client import generate_json, generate_json_with_reasoning_fallback
from research_agent.tools.vector_store import semantic_search


logger = logging.getLogger(__name__)

MAX_ANALYSIS_CONTEXT_TOKENS = 2500
SYNTHESIS_SECTION_TITLES = {
    "key_ideas": "Key Ideas",
    "methods": "Methods & Approach",
    "results": "Results & Evaluation",
    "discussion": "Discussion & Limitations",
}
DISCUSSION_FALLBACK_SECTION_NAMES = {"discussion", "conclusion"}
DISCUSSION_FALLBACK_PATTERNS = [
    "future work",
    "future direction",
    "limitation",
    "limitations",
    "threats to validity",
    "in conclusion",
    "we conclude",
    "we presented",
    "we introduced",
]
SECTION_AGENT_KEYS = tuple(SYNTHESIS_SECTION_TITLES.keys())
SECTION_RETRIEVAL_QUERIES = {
    "key_ideas": "abstract introduction conclusion main contribution novelty motivation outcome",
    "methods": "architecture design approach algorithm",
    "results": "evaluation benchmark metric performance comparison",
    "discussion": "limitation future work conclusion",
}
SECTION_AGENT_MAX_WORKERS = int(os.getenv("SECTION_AGENT_MAX_WORKERS", "1"))
SECTION_AGENT_MAX_RETRIEVAL_ROUNDS = 1
SECTION_AGENT_MAX_REWRITE_ROUNDS = 1
SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing pre-processed chunks from a research paper parsing pipeline.

The input is organized into four sections:
- Key Ideas
- Methods & Approach
- Results & Evaluation
- Discussion & Limitations

Within each section, chunks are sorted by importance and confidence scores and have
already been deduplicated and quality-filtered. These scores reflect how salient and
reliable each chunk is — prefer higher-importance chunks when consolidating or deciding
what to keep. Role inference is imperfect, so occasional off-topic or noisy chunks may
appear within a section; if a chunk seems clearly misplaced or incoherent with the others,
treat it as low-signal and deprioritize or discard it.

Your task is to produce a clean, structured summary that preserves the four-section layout
while synthesizing, compressing, and denoising the content within each section. The output
should let a researcher quickly grasp the paper's core contribution and findings without
reading the full paper.

For the Key Ideas, Methods & Approach, and Discussion & Limitations sections, you may:
- Merge similar or overlapping chunks into a single coherent statement
- Discard low-signal, redundant, or clearly mis-tagged content
- Rewrite or compress verbose text for clarity and concision
- Introduce sub-structure (e.g., bullet points, short headers) if it aids readability
Prefer a small number of high-signal items per section (typically 3–6), rather than many granular points.
For the Methods & Approach section specifically, apply this additional constraint:

Focus on architectural decisions and design choices - what components were used,
how they relate to each other, and why the design is novel or different from prior work.

Do NOT include:
- Specific hyperparameter values (e.g., learning rates, dropout rates, beta values,
  epsilon, mask parameters, warmup steps)
- Low-level training configuration (e.g., optimizer settings, augmentation parameters,
  batch sizes, stride values)
- Dataset statistics or feature extraction details unless they are central to the
  method's novelty

These details belong in the raw chunk evidence beneath the synthesis view, not in
the overview. A researcher skimming for relevance needs to understand the approach,
not reproduce the training run.
For the Results & Evaluation section, apply the following format rules:
1. If the input contains numerical comparisons, benchmark scores, or metric values,
   present them in a markdown table with clear column headers. If baseline or competing
   method names are available, include them as rows for direct comparison.
2. If results are qualitative or describe behavioral findings rather than numbers,
   present them as concise bullet points.
3. If a result only makes sense in the context of a specific experimental condition,
   include that condition inline rather than stripping it for brevity.
If numerical data is present but insufficient for a clean table, summarize it in bullets rather than forcing a table.
4. Always end the Results & Evaluation section with a one-line scope caveat if the
   evaluation appears limited in any of these ways:
   - Only one dataset was used
   - No ablation study was reported
   - Comparison baselines are weak or absent
   - Results are on a narrow or domain-specific benchmark
   Format it as: ⚠ Scope: [your caveat here]
Only include the scope caveat if there is clear evidence of limitation in the input. Do not assume missing details imply weakness.
   If evaluation appears thorough and well-rounded, omit the caveat entirely.
When multiple chunks overlap, prioritize the ones with higher importance scores and broader explanatory coverage.
Primarily keep content within its section, but if a chunk clearly expresses a core idea that is misplaced, you may reflect it in the appropriate section while avoiding duplication.

Focus on expressing underlying ideas and contributions, not just compressing individual chunks.

Avoid across all sections:
- Blindly copying chunks verbatim without synthesis
- Over-generalizing or abstracting beyond what the content actually supports
- Inventing claims, conclusions, or details not present in the input
- Fabricating metric values or experimental conditions not present in the chunks

Output a clean, structured result using the same four section headers. Within each section,
aim for the level of detail a researcher would need to assess the paper's relevance and
validity at a glance."""
REFERENCE_SECTION_NAMES = {"reference", "references", "bibliography", "works cited", "cited works"}
STRUCTURED_FIELD_MAP = {
    "training_objective": ["training objective", "training formulation"],
    "dataset": ["dataset", "corpus", "benchmark"],
    "optimizers": ["optimizer", "optimizers", "adam", "sgd", "learning rate"],
    "losses": ["loss", "losses", "objective", "cross-entropy", "mse", "rnnt", "ctc"],
    "architectures": ["architecture", "architectures", "model", "encoder", "decoder", "block"],
}
REPRODUCIBILITY_QUERY_KEYWORDS = [
    "reproduc",
    "reproduce",
    "replicate",
    "code available",
    "implementation",
    "repository",
    "repo",
]
INFERENCE_RULES = [
    (("transducer", "rnnt"), "RNNT", 0.9, "Transducer architectures are typically optimized with RNNT loss."),
    (("gan", "generator", "discriminator"), "adversarial loss", 0.75, "GAN architectures commonly use adversarial objectives."),
    (("diffusion", "denoising"), "denoising loss", 0.75, "Diffusion models are commonly trained with denoising objectives."),
]
SECTION_KEYWORDS = {
    "architecture": ["architecture", "model", "network", "layer", "encoder", "decoder"],
    "dataset": ["dataset", "corpus", "benchmark", "train set", "test set"],
    "loss": ["loss", "objective", "cross-entropy", "mse", "rnnt", "ctc"],
    "optimizer": ["optimizer", "adam", "sgd", "learning rate"],
    "training": ["training", "epoch", "batch size", "regularization", "dropout"],
    "evaluation": ["evaluation", "metric", "accuracy", "f1", "bleu", "rouge", "wer"],
    "contribution": ["contribution", "we propose", "our main", "novel", "introduce"],
}
METRIC_CANDIDATES = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "bleu",
    "rouge",
    "wer",
    "latency",
    "throughput",
]
SECTION_TARGETS = {
    "architecture": ["method"],
    "dataset": ["experiments", "results", "method", "abstract"],
    "training": ["method", "experiments", "abstract"],
    "metrics": ["results", "experiments", "abstract"],
    "contributions": ["abstract", "introduction", "conclusion"],
}
AUGMENTATION_KEYWORDS = ["augmentation", "masking", "noise injection", "mixup", "cutmix", "specaugment", "mask"]
CANONICAL_ARCHITECTURES = {
    "transformer model": "Transformer",
    "transformer": "Transformer",
    "conformer": "Conformer",
    "rnn": "RNN",
    "recurrent neural network": "RNN",
    "cnn": "CNN",
    "convolutional neural network": "CNN",
}


def _normalize_synthesis_section(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        lines = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(f"- {line}" for line in lines)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, indent=2)
    return str(value).strip()


def _format_synthesis_input(inferred_structure: dict | list | None) -> str:
    if not isinstance(inferred_structure, Mapping):
        return ""

    sections: list[str] = []
    for key, title in SYNTHESIS_SECTION_TITLES.items():
        sections.append(f"## {title}")
        entries = inferred_structure.get(key) if isinstance(inferred_structure, Mapping) else None
        if key == "methods" and isinstance(entries, Mapping):
            entries = entries.get("chunks")
        if not isinstance(entries, list) or not entries:
            sections.append("- No high-confidence evidence available.")
            sections.append("")
            continue

        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, Mapping):
                continue
            text = str(entry.get("text") or entry.get("summary") or "").strip()
            if not text:
                continue
            importance = float(entry.get("importance") or 0.0)
            confidence = float(entry.get("confidence") or 0.0)
            role = str(entry.get("role") or "other")
            text = text[:1800].strip()
            sections.append(
                f"- [{index}] role={role}; importance={importance:.2f}; confidence={confidence:.2f}; text={text}"
            )
        sections.append("")

    return "\n".join(sections).strip()


def _normalize_equation_items(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    normalized: list[dict] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        latex = str(item.get("latex") or "").strip()
        description = str(item.get("description") or "").strip()
        latex = " ".join(latex.split())
        latex = re.split(r"(?=\b(?:We|Figure|Fig\.|Table|Algorithm)\b)", latex, maxsplit=1)[0].strip(" ,;:.")
        if "=" in latex:
            lhs, rhs = latex.split("=", 1)
            lhs_norm = re.sub(r"\W+", "", lhs)
            rhs_norm = re.sub(r"[\s,.;:()]+", "", rhs)
            if len(lhs_norm) <= 3 and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", rhs_norm or ""):
                continue
        if any(marker in latex.lower() for marker in ["mask parameter", "time-mask ratio", "warm-up", "dropout"]):
            continue
        alpha_chars = len(re.sub(r"[^A-Za-z]", "", latex))
        if not latex or len(latex) < 12 or alpha_chars < 4:
            continue
        normalized.append(
            {
                "latex": latex,
                "description": description,
                "id": item.get("id"),
                "chunk_index": item.get("chunk_index"),
                "text": item.get("text"),
            }
        )
    return normalized


def _select_discussion_fallback_chunks(chunk_payloads: list[dict], *, max_items: int = 3) -> list[dict]:
    candidates: list[dict] = []
    for payload in chunk_payloads:
        content = str(payload.get("content") or payload.get("text") or "").strip()
        if not content:
            continue
        normalized_section = _normalize_section_name(payload.get("section_name"))
        lowered = content.lower()
        if normalized_section in DISCUSSION_FALLBACK_SECTION_NAMES or any(
            pattern in lowered for pattern in DISCUSSION_FALLBACK_PATTERNS
        ):
            candidates.append(
                {
                    "id": payload.get("chunk_id"),
                    "text": content,
                    "summary": str(payload.get("summary") or payload.get("content_excerpt") or "").strip(),
                    "role": payload.get("role") or "discussion",
                    "importance": float(payload.get("importance") or 0.0),
                    "confidence": float(payload.get("confidence") or 0.0),
                    "source": payload.get("source") or "extracted",
                    "section_name": payload.get("section_name"),
                    "chunk_index": payload.get("chunk_index"),
                }
            )

    ranked = sorted(
        candidates,
        key=lambda item: (float(item.get("importance") or 0.0), float(item.get("confidence") or 0.0)),
        reverse=True,
    )
    deduped: list[dict] = []
    seen: set[str] = set()
    for item in ranked:
        summary_key = str(item.get("summary") or item.get("text") or "").strip().lower()
        if not summary_key or summary_key in seen:
            continue
        seen.add(summary_key)
        deduped.append(item)
        if len(deduped) >= max_items:
            break
    return deduped


def _build_synthesis_structure(inferred_structure: dict, chunk_payloads: list[dict]) -> dict:
    synthesis_structure: dict[str, object] = {
        "key_ideas": list(inferred_structure.get("key_ideas", []))
        if isinstance(inferred_structure.get("key_ideas"), list)
        else [],
        "methods": {
            "chunks": [],
            "equations": {
                "source": None,
                "items": [],
            },
        },
        "results": list(inferred_structure.get("results", []))
        if isinstance(inferred_structure.get("results"), list)
        else [],
        "discussion": list(inferred_structure.get("discussion", []))
        if isinstance(inferred_structure.get("discussion"), list)
        else [],
    }
    methods_value = inferred_structure.get("methods")
    if isinstance(methods_value, Mapping):
        equations_value = methods_value.get("equations")
        synthesis_structure["methods"] = {
            "chunks": list(methods_value.get("chunks", []))
            if isinstance(methods_value.get("chunks"), list)
            else [],
            "equations": {
                "source": equations_value.get("source") if isinstance(equations_value, Mapping) else None,
                "items": list(equations_value.get("items", []))
                if isinstance(equations_value, Mapping) and isinstance(equations_value.get("items"), list)
                else [],
            },
        }
    elif isinstance(methods_value, list):
        synthesis_structure["methods"] = {
            "chunks": list(methods_value),
            "equations": {
                "source": None,
                "items": [],
            },
        }

    if synthesis_structure.get("discussion"):
        return synthesis_structure

    fallback_discussion = _select_discussion_fallback_chunks(chunk_payloads)
    if fallback_discussion:
        synthesis_structure["discussion"] = fallback_discussion
    return synthesis_structure


def _default_section_result(
    synthesis: str = "",
    *,
    confidence: str = "low",
    warning: str | None = None,
    fabrication_flagged: bool = False,
    retrieval_rounds: int = 0,
    rewrite_rounds: int = 0,
    review_score: int = 0,
    review_issues: list[str] | None = None,
    evidence_chunk_count: int = 0,
) -> dict:
    return {
        "synthesis": synthesis.strip(),
        "confidence": confidence,
        "warning": warning,
        "fabrication_flagged": fabrication_flagged,
        "retrieval_rounds": min(retrieval_rounds, SECTION_AGENT_MAX_RETRIEVAL_ROUNDS),
        "rewrite_rounds": min(rewrite_rounds, SECTION_AGENT_MAX_REWRITE_ROUNDS),
        "review_score": max(0, min(10, int(review_score))),
        "review_issues": review_issues or [],
        "evidence_chunk_count": evidence_chunk_count,
    }


def _get_section_chunks(inferred_structure: dict | None, section_key: str) -> list[dict]:
    if not isinstance(inferred_structure, Mapping):
        return []
    value = inferred_structure.get(section_key)
    if section_key == "methods" and isinstance(value, Mapping):
        value = value.get("chunks")
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _format_section_chunks_for_prompt(section_chunks: list[dict]) -> str:
    if not section_chunks:
        return "[]"

    lines: list[str] = []
    for index, chunk in enumerate(section_chunks, start=1):
        text = str(chunk.get("text") or chunk.get("summary") or "").strip()
        if not text:
            continue
        lines.append(
            json.dumps(
                {
                    "index": index,
                    "id": chunk.get("id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "section_name": chunk.get("section_name"),
                    "role": chunk.get("role"),
                    "importance": round(float(chunk.get("importance") or 0.0), 4),
                    "confidence": round(float(chunk.get("confidence") or 0.0), 4),
                    "text": text[:2200],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) if lines else "[]"


def _merge_section_chunks(section_chunks: list[dict], additional_chunks: list[dict], *, max_items: int = 10) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for chunk in sorted(
        [*section_chunks, *additional_chunks],
        key=lambda item: (
            float(item.get("importance") or 0.0),
            float(item.get("confidence") or 0.0),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    ):
        key = str(chunk.get("id") or f"chunk-{chunk.get('chunk_index')}")
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(chunk))
        if len(merged) >= max_items:
            break
    return merged


def retrieve_additional_chunks(section_key: str, paper_id: uuid.UUID, existing_chunks: list[dict] | None = None) -> list[dict]:
    query = SECTION_RETRIEVAL_QUERIES.get(section_key, section_key)
    seen_ids = {str(chunk.get("id")) for chunk in (existing_chunks or []) if chunk.get("id")}
    query_embedding = generate_embedding(query)

    with SessionLocal() as session:
        retrieved = semantic_search(
            session,
            query_embedding,
            paper_id=paper_id,
            top_k=12,
        )

    additional_chunks: list[dict] = []
    total_chunks = max(1, len(retrieved))
    for candidate in retrieved:
        chunk_id = str(candidate.get("chunk_id") or "")
        if chunk_id and chunk_id in seen_ids:
            continue
        structure = build_chunk_structure(
            chunk_id=chunk_id or None,
            chunk_index=int(candidate.get("chunk_index") or 0),
            section_name=candidate.get("section_name"),
            subsection_name=candidate.get("subsection_name"),
            content=str(candidate.get("content") or ""),
            total_chunks=total_chunks,
        )
        additional_chunks.append(
            {
                **structure,
                "id": chunk_id or None,
                "paper_id": str(candidate.get("paper_id") or paper_id),
                "content": str(candidate.get("content") or ""),
                "text": str(candidate.get("content") or ""),
                "page_number": candidate.get("page_number"),
                "score": float(candidate.get("score") or 0.0),
                "source": "extracted",
            }
        )
        if chunk_id:
            seen_ids.add(chunk_id)
        if len(additional_chunks) >= 5:
            break

    return additional_chunks


def _run_evidence_assessment(section_key: str, section_chunks: list[dict]) -> dict:
    issues: list[str] = []
    if not section_chunks:
        return {
            "sufficient": False,
            "trustworthy": False,
            "issues": ["No evidence chunks available."],
            "retrieve_more": True,
            "low_confidence_reason": "No evidence chunks available.",
        }

    chunk_count = len(section_chunks)
    avg_confidence = sum(float(chunk.get("confidence") or 0.0) for chunk in section_chunks) / chunk_count
    avg_importance = sum(float(chunk.get("importance") or 0.0) for chunk in section_chunks) / chunk_count
    unique_sections = {
        _normalize_section_name(chunk.get("section_name")) or "unknown"
        for chunk in section_chunks
    }
    text_blob = "\n".join(str(chunk.get("text") or chunk.get("content") or "") for chunk in section_chunks).lower()
    contradiction_markers = ["however", "but", "although", "in contrast", "whereas", "limitation", "limitations"]
    contradiction_hits = sum(text_blob.count(marker) for marker in contradiction_markers)

    required_keywords = {
        "key_ideas": ["propose", "novel", "contribution", "motivation", "outperform"],
        "methods": ["model", "architecture", "algorithm", "module", "approach"],
        "results": ["results", "benchmark", "metric", "accuracy", "evaluation"],
        "discussion": ["limitation", "future work", "conclusion", "caveat", "discussion"],
    }.get(section_key, [])
    keyword_hits = sum(1 for keyword in required_keywords if keyword in text_blob)

    sufficient = chunk_count >= 2 and avg_confidence >= 0.45 and avg_importance >= 0.25
    if section_key == "results":
        sufficient = sufficient and keyword_hits >= 1
    elif section_key == "methods":
        sufficient = sufficient and keyword_hits >= 1
    elif section_key == "discussion":
        sufficient = sufficient and (
            keyword_hits >= 1
            or "conclusion" in unique_sections
            or "results" in unique_sections
        )

    trustworthy = avg_confidence >= 0.4 and contradiction_hits <= max(3, chunk_count * 2)
    retrieve_more = (not sufficient or keyword_hits == 0) and chunk_count < 6

    if chunk_count < 2:
        issues.append("Too few evidence chunks for stable synthesis.")
    if avg_confidence < 0.45:
        issues.append("Average chunk confidence is low.")
    if avg_importance < 0.25:
        issues.append("Retrieved chunks are low-importance.")
    if keyword_hits == 0 and required_keywords:
        issues.append(f"Evidence lacks strong {section_key} cues.")
    if contradiction_hits > max(3, chunk_count * 2):
        issues.append("Evidence contains many contradiction or caveat markers.")

    low_confidence_reason = "; ".join(issues) if issues else None
    return {
        "sufficient": sufficient,
        "trustworthy": trustworthy,
        "issues": issues,
        "retrieve_more": retrieve_more,
        "low_confidence_reason": low_confidence_reason,
    }


def run_section_synthesis(
    section_key: str,
    evidence_chunks: list[dict],
    paper_id: uuid.UUID,
    retry_instructions: str | None = None,
) -> str:
    if not evidence_chunks:
        return ""

    section_guidance = {
        "key_ideas": (
            "For this section, treat 'key_ideas' as a quick executive summary for the reader. "
            "It is allowed to use evidence drawn from any part of the paper if it helps explain "
            "the problem, novelty, main contribution, and top-line outcome. Prefer a crisp, high-level summary "
            "over implementation detail."
        ),
        "discussion": (
            "For this section, if the paper has no explicit discussion/limitations section, say so implicitly "
            "through a cautious synthesis based on conclusions, future work statements, result caveats, or "
            "other grounded evidence from the paper. Do not fabricate limitations; infer them only when the "
            "source chunks support them."
        ),
    }.get(section_key, "")

    prompt = (
        f"{SYNTHESIS_SYSTEM_PROMPT}\n\n"
        "You are synthesizing exactly one section. Focus only on the requested section below.\n"
        f"Section key: {section_key}\n"
        f"Section title: {SYNTHESIS_SECTION_TITLES.get(section_key, section_key)}\n\n"
        f"{section_guidance}\n\n"
        "Return JSON only using this exact schema:\n"
        '{"synthesis": "markdown string"}\n\n'
        f"Paper ID: {paper_id}\n\n"
        "Source chunks:\n"
        f"{_format_section_chunks_for_prompt(evidence_chunks)}"
    )
    if retry_instructions:
        prompt += f"\n\nRewrite instructions:\n{retry_instructions.strip()}"

    response = generate_json(prompt)
    return _normalize_synthesis_section(response.get("synthesis"))


def _review_section_synthesis(section_key: str, evidence_chunks: list[dict], synthesis_output: str) -> dict:
    section_guidance = {
        "key_ideas": (
            "Important: do not penalize evidence solely because it comes from methods or results. "
            "'key_ideas' is a synthetic reader-summary section and may legitimately draw from across the paper."
        ),
        "discussion": (
            "Important: if explicit discussion chunks are sparse, a cautious summary grounded in conclusion, "
            "future work, or result caveats is acceptable. Only flag unsupported claims, not reasonable synthesis "
            "from cross-section evidence."
        ),
    }.get(section_key, "")
    prompt = (
        "You are reviewing an AI-generated summary of one section of a research paper.\n\n"
        f"Section: {section_key}\n\n"
        f"{section_guidance}\n\n"
        "Source chunks (ground truth):\n"
        f"{_format_section_chunks_for_prompt(evidence_chunks)}\n\n"
        "Synthesis output:\n"
        f"{synthesis_output}\n\n"
        "Choose exactly one strategy:\n"
        '"pass"     - synthesis is accurate, coherent, and appropriately abstracted\n'
        '"rewrite"  - synthesis has fixable quality issues; provide specific rewrite instructions\n'
        '"retrieve" - synthesis is weak because evidence was sparse; more retrieval may help\n'
        '"warn"     - synthesis contains fabricated claims, invented metrics, or unsupported statements\n\n'
        "Section-specific criteria:\n"
        "key_ideas:\n"
        "- Captures core contribution without over-generalizing\n"
        "- Grounded in chunk content from anywhere in the paper\n"
        "- Should not reject valid evidence because it originated in methods or results\n\n"
        "methods:\n"
        "- Describes architecture and design decisions\n"
        "- Free of unnecessary hyperparameter detail\n"
        "- No invented values\n\n"
        "results:\n"
        "- All metrics grounded in source chunks\n"
        "- Table accurate if present\n"
        "- Scope caveat present if evaluation is narrow\n\n"
        "discussion:\n"
        "- Reflects available content\n"
        "- If explicit discussion is sparse, may cautiously synthesize from conclusions, future work, or caveats in other sections\n"
        "- If empty, confirm there was not enough grounded evidence even after considering broader paper evidence\n\n"
        "Respond ONLY with JSON:\n"
        '{'
        '"strategy": "pass", '
        '"score": 0, '
        '"issues": ["issue 1"], '
        '"rewrite_instructions": null, '
        '"warning_message": null'
        "}"
    )
    response = generate_json(prompt)
    strategy = str(response.get("strategy") or "warn").strip().lower()
    if strategy not in {"pass", "rewrite", "retrieve", "warn"}:
        strategy = "warn"
    issues = response.get("issues") if isinstance(response.get("issues"), list) else []
    try:
        score = int(float(response.get("score") or 0))
    except (TypeError, ValueError):
        score = 0
    return {
        "strategy": strategy,
        "score": max(0, min(10, score)),
        "issues": [str(issue).strip() for issue in issues if str(issue).strip()],
        "rewrite_instructions": (
            str(response.get("rewrite_instructions")).strip() if response.get("rewrite_instructions") else None
        ),
        "warning_message": (
            str(response.get("warning_message")).strip() if response.get("warning_message") else None
        ),
    }


def run_section_agent(
    section_key: str,
    section_chunks: list[dict],
    paper_id: uuid.UUID,
    full_inferred_structure: dict | None,
) -> dict:
    start_time = time.perf_counter()
    llm_calls = 0
    retrieval_rounds = 0
    rewrite_rounds = 0
    confidence = "high"
    fabrication_flagged = False
    warning: str | None = None

    evidence_chunks = _merge_section_chunks(section_chunks, [])
    if section_key == "key_ideas":
        cross_section_seed = (
            _get_section_chunks(full_inferred_structure, "key_ideas")
            + _get_section_chunks(full_inferred_structure, "methods")
            + _get_section_chunks(full_inferred_structure, "results")
            + _get_section_chunks(full_inferred_structure, "discussion")
        )
        evidence_chunks = _merge_section_chunks(evidence_chunks, cross_section_seed, max_items=10)
    elif not evidence_chunks and section_key == "discussion":
        discussion_seed = _get_section_chunks(full_inferred_structure, "discussion")
        if not discussion_seed:
            discussion_seed = _merge_section_chunks(
                _get_section_chunks(full_inferred_structure, "results"),
                _get_section_chunks(full_inferred_structure, "key_ideas")
                + _get_section_chunks(full_inferred_structure, "methods"),
                max_items=8,
            )
        evidence_chunks = _merge_section_chunks(evidence_chunks, discussion_seed, max_items=8)

    assessment = _run_evidence_assessment(section_key, evidence_chunks)
    llm_calls += 1

    if not assessment["sufficient"]:
        confidence = "low"
        warning = assessment["low_confidence_reason"] or "; ".join(assessment["issues"]) or warning
        if assessment["retrieve_more"] and retrieval_rounds < SECTION_AGENT_MAX_RETRIEVAL_ROUNDS:
            retrieved_chunks = retrieve_additional_chunks(section_key, paper_id, evidence_chunks)
            retrieval_rounds += 1
            evidence_chunks = _merge_section_chunks(evidence_chunks, retrieved_chunks)
            assessment = _run_evidence_assessment(section_key, evidence_chunks)
            llm_calls += 1
            if not assessment["sufficient"]:
                confidence = "low"
                warning = assessment["low_confidence_reason"] or "; ".join(assessment["issues"]) or warning

    if not assessment["trustworthy"]:
        confidence = "low"
        warning = assessment["low_confidence_reason"] or "; ".join(assessment["issues"]) or warning

    synthesis = run_section_synthesis(section_key, evidence_chunks, paper_id)
    llm_calls += 1
    review = _review_section_synthesis(section_key, evidence_chunks, synthesis)
    llm_calls += 1

    best_output = synthesis
    best_review = review
    final_output = synthesis
    final_review = review

    if review["strategy"] == "rewrite":
        rewritten_output = run_section_synthesis(
            section_key,
            evidence_chunks,
            paper_id,
            retry_instructions=review.get("rewrite_instructions"),
        )
        rewrite_rounds = 1
        llm_calls += 1
        rewritten_review = _review_section_synthesis(section_key, evidence_chunks, rewritten_output)
        llm_calls += 1
        if rewritten_review["score"] >= best_review["score"]:
            best_output = rewritten_output
            best_review = rewritten_review
        if rewritten_review["strategy"] == "pass":
            final_output = rewritten_output
            final_review = rewritten_review
            if confidence != "low":
                confidence = "high"
        else:
            final_output = best_output
            final_review = best_review
            if confidence != "low":
                confidence = "medium"
            if final_review["strategy"] == "warn":
                fabrication_flagged = True
                warning = final_review.get("warning_message") or warning
                confidence = "low"
    elif review["strategy"] == "retrieve":
        if retrieval_rounds < SECTION_AGENT_MAX_RETRIEVAL_ROUNDS:
            retrieved_chunks = retrieve_additional_chunks(section_key, paper_id, evidence_chunks)
            retrieval_rounds += 1
            evidence_chunks = _merge_section_chunks(evidence_chunks, retrieved_chunks)
            retrieved_output = run_section_synthesis(section_key, evidence_chunks, paper_id)
            llm_calls += 1
            retrieved_review = _review_section_synthesis(section_key, evidence_chunks, retrieved_output)
            llm_calls += 1
            if retrieved_review["score"] >= best_review["score"]:
                best_output = retrieved_output
                best_review = retrieved_review
            final_output = best_output
            final_review = best_review
            if final_review["strategy"] == "pass":
                if confidence != "low":
                    confidence = "high"
            elif final_review["strategy"] == "warn":
                fabrication_flagged = True
                warning = final_review.get("warning_message") or warning
                confidence = "low"
            else:
                confidence = "low"
        else:
            confidence = "low"
    elif review["strategy"] == "warn":
        fabrication_flagged = True
        warning = review.get("warning_message") or warning
        confidence = "low"
    else:
        if confidence != "low":
            confidence = "high"

    if fabrication_flagged:
        logger.warning(
            "section_fabrication_flagged paper_id=%s section_key=%s warning=%s",
            paper_id,
            section_key,
            warning,
        )

    result = _default_section_result(
        final_output,
        confidence=confidence,
        warning=warning,
        fabrication_flagged=fabrication_flagged,
        retrieval_rounds=retrieval_rounds,
        rewrite_rounds=rewrite_rounds,
        review_score=final_review.get("score") or 0,
        review_issues=final_review.get("issues") or [],
        evidence_chunk_count=len(evidence_chunks),
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        "section_agent_complete paper_id=%s section_key=%s strategy=%s score=%s confidence=%s retrieval_rounds=%s rewrite_rounds=%s llm_calls=%s elapsed_ms=%s",
        paper_id,
        section_key,
        final_review.get("strategy"),
        final_review.get("score"),
        confidence,
        retrieval_rounds,
        rewrite_rounds,
        llm_calls,
        elapsed_ms,
    )
    return {
        "section_key": section_key,
        "result": result,
        "llm_calls": llm_calls,
        "elapsed_ms": elapsed_ms,
        "strategy": final_review.get("strategy"),
    }


def _compute_analysis_status(section_results: dict[str, dict | None]) -> dict:
    successful_sections = [key for key, value in section_results.items() if isinstance(value, Mapping)]
    failed_sections = [key for key, value in section_results.items() if not isinstance(value, Mapping)]
    if failed_sections and successful_sections:
        status = "partial_failure"
        message = "Some section agents failed."
    elif failed_sections and not successful_sections:
        status = "failed"
        message = "All section agents failed."
    else:
        status = "success"
        message = None
    return {
        "status": status,
        "message": message,
        "successful_sections": successful_sections,
        "failed_sections": failed_sections,
    }


def run_equation_fallback(methods_synthesis: str, paper_id) -> list[dict]:
    if not methods_synthesis.strip():
        return []
    prompt = (
        "The following is a synthesis of the methods section of a research paper.\n"
        "Extract or reconstruct the key mathematical expressions, formulas, or\n"
        "equations that underpin this method. Output them as JSON with schema:\n"
        '{"items": [{"latex": "string", "description": "string"}]}\n'
        "If no meaningful equations can be inferred, return {\"items\": []}.\n\n"
        f"Paper ID: {paper_id}\n\n"
        "Methods synthesis:\n"
        f"{methods_synthesis}"
    )
    start_time = time.perf_counter()
    response = generate_json(prompt)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    items = _normalize_equation_items(response.get("items"))
    logger.info(
        "equation_fallback_complete paper_id=%s latency_ms=%s equation_count=%s",
        paper_id,
        latency_ms,
        len(items),
    )
    return items


def _ensure_method_equations(inferred_structure: dict, methods_synthesis: str, paper_id) -> dict:
    methods_value = inferred_structure.get("methods")
    if not isinstance(methods_value, Mapping):
        return inferred_structure

    equations_value = methods_value.get("equations")
    if not isinstance(equations_value, Mapping):
        equations_value = {"source": None, "items": []}

    items = _normalize_equation_items(equations_value.get("items"))
    if items:
        methods_copy = dict(methods_value)
        methods_copy["equations"] = {"source": "extracted", "items": items}
        updated = dict(inferred_structure)
        updated["methods"] = methods_copy
        return updated

    generated_items = run_equation_fallback(methods_synthesis, paper_id)
    methods_copy = dict(methods_value)
    methods_copy["equations"] = {
        "source": "llm_generated" if generated_items else None,
        "items": generated_items,
    }
    updated = dict(inferred_structure)
    updated["methods"] = methods_copy
    return updated


def run_synthesis_pass(inferred_structure, paper_id) -> dict[str, str]:
    prompt_input = _format_synthesis_input(inferred_structure)
    if not prompt_input:
        return {key: "" for key in SYNTHESIS_SECTION_TITLES}
    prompt = (
        f"{SYNTHESIS_SYSTEM_PROMPT}\n\n"
        "Return JSON only using this exact schema:\n"
        '{'
        '"key_ideas": "markdown string", '
        '"methods": "markdown string", '
        '"results": "markdown string", '
        '"discussion": "markdown string"'
        "}\n\n"
        f"Paper ID: {paper_id}\n\n"
        f"Input:\n{prompt_input}"
    )

    start_time = time.perf_counter()
    response = generate_json(prompt)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    normalized = {
        key: _normalize_synthesis_section(response.get(key))
        for key in SYNTHESIS_SECTION_TITLES
    }
    logger.info(
        "synthesis_pass_complete paper_id=%s latency_ms=%s generated_sections=%s",
        paper_id,
        latency_ms,
        [key for key, value in normalized.items() if value],
    )
    return normalized


def _normalize_section_name(section_name: str | None) -> str | None:
    if not section_name:
        return None
    normalized = section_name.strip().lower().replace("_", " ")
    normalized = " ".join(normalized.split())
    if normalized in REFERENCE_SECTION_NAMES:
        return "references"
    return normalized


def _score_chunk(content: str) -> int:
    # Backward-compatible function name; scoring is now domain-agnostic and structural.
    technical_tokens = re.findall(r"\b[a-zA-Z]{7,}\b", content)
    equation_markers = len(re.findall(r"[=<>+\-*/^]|O\(", content))
    pseudocode_markers = len(
        re.findall(r"\b(for each|while|if|algorithm|procedure|return)\b", content.lower())
    )
    result_markers = len(
        re.findall(r"\b(results?|benchmark|comparison|ablation|table|figure)\b", content.lower())
    )
    return len(technical_tokens) + equation_markers + pseudocode_markers * 2 + result_markers * 2


def _select_relevant_chunks(chunks: list[PaperChunk]) -> list[PaperChunk]:
    evidence_chunks = [chunk for chunk in chunks if _normalize_section_name(chunk.section_name) != "references"]
    total_chunks = len(evidence_chunks)
    quality_chunks: list[PaperChunk] = []
    for chunk in evidence_chunks:
        structure = build_chunk_structure(
            chunk_id=str(chunk.id),
            chunk_index=chunk.chunk_index,
            section_name=chunk.section_name,
            subsection_name=chunk.subsection_name,
            content=chunk.content,
            total_chunks=max(1, total_chunks),
        )
        if is_quality_chunk(structure):
            quality_chunks.append(chunk)

    candidate_chunks = quality_chunks or evidence_chunks
    total_chunks = len(candidate_chunks)
    ranked = sorted(
        candidate_chunks,
        key=lambda chunk: (
            build_chunk_structure(
                chunk_id=str(chunk.id),
                chunk_index=chunk.chunk_index,
                section_name=chunk.section_name,
                subsection_name=chunk.subsection_name,
                content=chunk.content,
                total_chunks=max(1, total_chunks),
            )["importance"],
            _score_chunk(chunk.content),
            -chunk.chunk_index,
        ),
        reverse=True,
    )
    selected = ranked[:12] if ranked else candidate_chunks[:12]
    return sorted(selected, key=lambda chunk: chunk.chunk_index)


def _build_prompt(paper: Paper, chunks: list[PaperChunk]) -> str:
    tokenizer = get_tokenizer()
    current_tokens = 0
    context_parts: list[str] = []

    for chunk in chunks:
        part = f"[Chunk {chunk.chunk_index} | Section {chunk.section_name or 'unknown'}]\n{chunk.content}"
        part_tokens = len(
            tokenizer(
                part,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )["input_ids"]
        )
        if current_tokens + part_tokens > MAX_ANALYSIS_CONTEXT_TOKENS:
            break
        context_parts.append(part)
        current_tokens += part_tokens

    context = "\n\n".join(context_parts)
    return (
        "You are analyzing a machine learning research paper.\n"
        "Use only the provided paper context. Ignore references, citations, bibliography entries, and paper titles in citations as evidence.\n"
        "Return valid JSON only.\n\n"
        "Extract these fields:\n"
        "- model_architecture: short primary architecture summary\n"
        "- architectures: {proposed: list[str], baseline: list[str]}\n"
        "- dataset: string or null\n"
        "- loss_function: short primary loss summary or null\n"
        "- losses: {primary: str|null, auxiliary: list[str], baseline: list[str], inferred: bool, confidence: float}\n"
        "- training_objective: short natural language description of the overall training formulation\n"
        "- optimizer: short primary optimizer summary or null\n"
        "- optimizers: {primary: str|null, baseline: list[str]}\n"
        "- training_details: object\n"
        "- evaluation_metrics: list[str]\n"
        "- contributions: list[str]\n\n"
        "Loss extraction hints:\n"
        "- Primary loss clues: 'we train using', 'training objective', 'our model optimizes', 'loss function'\n"
        "- Auxiliary loss clues: 'auxiliary loss', 'multi-task loss', 'additional loss'\n"
        "- Baseline loss clues: 'baseline uses', 'previous work uses', 'compared with'\n"
        "- Regularization, augmentation, schedules, and optimization tricks belong in training_details, not losses\n\n"
        f"Paper title:\n{paper.title}\n\n"
        f"Context:\n{context}"
    )


def _build_section_context(section_rows: list[PaperSection], allowed_section_names: list[str]) -> str:
    allowed = {_normalize_section_name(name) for name in allowed_section_names}
    selected = [row for row in section_rows if _normalize_section_name(row.section_name) in allowed]
    ordered = sorted(selected, key=lambda row: row.section_order)
    return "\n\n".join(
        f"[Section {row.section_name}]\n{row.content}"
        for row in ordered
        if _normalize_section_name(row.section_name) != "references"
    )[:12000]


def _build_section_target_prompts(paper: Paper, section_rows: list[PaperSection]) -> dict[str, str]:
    architecture_context = _build_section_context(section_rows, SECTION_TARGETS["architecture"])
    dataset_context = _build_section_context(section_rows, SECTION_TARGETS["dataset"])
    training_context = _build_section_context(section_rows, SECTION_TARGETS["training"])
    metrics_context = _build_section_context(section_rows, SECTION_TARGETS["metrics"])
    contributions_context = _build_section_context(section_rows, SECTION_TARGETS["contributions"])
    return {
        "architecture": (
            "Extract architecture information from the paper sections below. Ignore references. "
            "Return JSON only with keys model_architecture and architectures, where architectures = {proposed: list[str], baseline: list[str]}.\n\n"
            f"Paper title: {paper.title}\n\nContext:\n{architecture_context}"
        ),
        "training": (
            "Extract training setup information from the paper sections below. Ignore references. Return JSON only with keys "
            "dataset, loss_function, losses, training_objective, optimizer, optimizers, training_details.\n"
            "Primary loss signals: 'we train using', 'training objective', 'our model optimizes', 'loss function'.\n"
            "Auxiliary loss signals: 'auxiliary loss', 'multi-task loss', 'additional loss'.\n"
            "Baseline loss signals: 'baseline uses', 'previous work uses', 'compared with'.\n"
            "Regularization methods, learning rate schedules, decoder structure, optimization parameters, and augmentation methods belong in training_details.\n\n"
            f"Context:\n{training_context}\n\nDataset hints:\n{dataset_context}"
        ),
        "metrics": (
            "Extract evaluation metrics from the paper sections below. Return JSON only with key evaluation_metrics as a list of strings.\n\n"
            f"Context:\n{metrics_context}"
        ),
        "contributions": (
            "Extract the paper's main contributions from the sections below. Return JSON only with key contributions as a list of strings.\n\n"
            f"Context:\n{contributions_context}"
        ),
    }


def _parse_json_like(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_list_field(value) -> list | None:
    value = _parse_json_like(value)
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, Mapping):
        return [json.dumps(dict(value), ensure_ascii=False)]
    return [str(value)]


def _normalize_training_details(value):
    value = _parse_json_like(value)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return {"summary": str(value)}


def _extract_augmentations(training_details: dict | None, paper_text: str) -> list[str]:
    text = f"{json.dumps(training_details, ensure_ascii=False) if training_details else ''}\n{paper_text}".lower()
    found: list[str] = []
    for keyword in AUGMENTATION_KEYWORDS:
        if keyword in text:
            canonical = "SpecAugment" if keyword == "specaugment" else keyword.title()
            if canonical not in found:
                found.append(canonical)
    return found


def _canonicalize_architecture_name(name: str) -> str:
    lowered = name.strip().lower().replace(" model", "").strip()
    return CANONICAL_ARCHITECTURES.get(lowered, name.strip())


def _dedupe_architecture_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        canonical = _canonicalize_architecture_name(name)
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(canonical)
    return deduped


def _normalize_architectures(value, fallback: str | None) -> dict:
    value = _parse_json_like(value)
    normalized = {"proposed": [], "baseline": []}
    if isinstance(value, Mapping):
        normalized["proposed"] = _normalize_list_field(value.get("proposed")) or []
        normalized["baseline"] = _normalize_list_field(value.get("baseline")) or []
    elif value:
        normalized["proposed"] = _normalize_list_field(value) or []
    if fallback and fallback not in normalized["proposed"]:
        normalized["proposed"].insert(0, fallback)
    normalized["proposed"] = _dedupe_architecture_names(normalized["proposed"])
    normalized["baseline"] = _dedupe_architecture_names(normalized["baseline"])
    return normalized


def _normalize_losses(value, fallback: str | None) -> dict:
    value = _parse_json_like(value)
    normalized = {
        "primary": fallback,
        "auxiliary": [],
        "baseline": [],
        "inferred": False,
        "confidence": 1.0 if fallback else 0.0,
    }
    if isinstance(value, Mapping):
        normalized["primary"] = value.get("primary") or normalized["primary"]
        normalized["auxiliary"] = _normalize_list_field(value.get("auxiliary")) or []
        normalized["baseline"] = _normalize_list_field(value.get("baseline")) or []
        normalized["inferred"] = bool(value.get("inferred", normalized["inferred"]))
        try:
            normalized["confidence"] = float(value.get("confidence", normalized["confidence"]))
        except (TypeError, ValueError):
            pass
    elif value:
        normalized["primary"] = str(value)
        normalized["confidence"] = 1.0
    return normalized


def _normalize_optimizers(value, fallback: str | None) -> dict:
    value = _parse_json_like(value)
    normalized = {"primary": fallback, "baseline": []}
    if isinstance(value, Mapping):
        normalized["primary"] = value.get("primary") or normalized["primary"]
        normalized["baseline"] = _normalize_list_field(value.get("baseline")) or []
    elif value:
        normalized["primary"] = str(value)
    return normalized


def _coerce_confidence(value, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        confidence_map = {
            "very high": 0.95,
            "high": 0.8,
            "medium": 0.6,
            "moderate": 0.6,
            "low": 0.35,
            "very low": 0.2,
        }
        if lowered in confidence_map:
            return confidence_map[lowered]
        try:
            return float(lowered)
        except ValueError:
            return default
    return default


def _build_loss_fallback_prompt(architecture_summary: str) -> str:
    return (
        f"The paper proposes the following architecture and training setup:\n\n{architecture_summary}\n\n"
        "If the training loss is not explicitly stated, infer the most likely loss function used to train this type of model.\n\n"
        "Return JSON with keys: loss, confidence, justification."
    )


def _build_loss_confirmation_prompt(
    architecture_summary: str,
    dataset: str | None,
    training_objective: str | None,
    training_details: dict,
    current_loss: str | None,
) -> str:
    return (
        "You are reviewing a machine learning paper analysis.\n"
        "Your task is to confirm the most likely primary training loss.\n"
        "Be conservative. Only choose a loss if the evidence is reasonably strong.\n"
        "If the evidence is weak or ambiguous, return Unknown.\n\n"
        f"Architecture: {architecture_summary}\n"
        f"Dataset: {dataset or 'Unknown'}\n"
        f"Training objective summary: {training_objective or 'Unknown'}\n"
        f"Training details: {json.dumps(training_details, ensure_ascii=False)}\n"
        f"Current inferred loss: {current_loss or 'None'}\n\n"
        "Return JSON with keys:\n"
        "- loss: one of [CTC, RNNT, CrossEntropy, BCEWithLogits, MSE, Unknown]\n"
        "- confidence: float between 0 and 1\n"
        "- justification: short string\n"
        "- explicitly_stated: boolean"
    )


def _generate_training_objective(paper_analysis: dict) -> str | None:
    if paper_analysis.get("training_objective"):
        return paper_analysis["training_objective"]
    architectures = paper_analysis.get("architectures") or {}
    proposed = architectures.get("proposed", []) if isinstance(architectures, Mapping) else []
    primary_loss = (
        (paper_analysis.get("losses") or {}).get("primary")
        if isinstance(paper_analysis.get("losses"), Mapping)
        else None
    )
    description = " ".join([*(proposed[:1]), paper_analysis.get("dataset") or "", primary_loss or ""]).lower()
    if "speech" in description or "librispeech" in description:
        return "sequence-to-sequence speech recognition training objective"
    if "contrastive" in description:
        return "contrastive representation learning objective"
    if "mask" in description and "language" in description:
        return "masked language modeling objective"
    if primary_loss:
        return f"training objective centered on {primary_loss}"
    return None


def infer_training_details(paper_analysis: dict, paper_text: str) -> dict:
    losses = dict(paper_analysis.get("losses") or {})
    training_details = dict(paper_analysis.get("training_details") or {})

    if losses.get("primary"):
        losses["inferred"] = bool(losses.get("inferred", False))
        losses["confidence"] = float(losses.get("confidence", 1.0) or 1.0)
        augmentations = _extract_augmentations(training_details, paper_text)
        if augmentations:
            training_details["augmentation"] = augmentations
        paper_analysis["losses"] = losses
        paper_analysis["training_objective"] = _generate_training_objective(paper_analysis)
        paper_analysis["training_details"] = training_details or None
        return paper_analysis

    architecture_text = " ".join(
        [
            paper_analysis.get("model_architecture") or "",
            " ".join((paper_analysis.get("architectures") or {}).get("proposed", [])),
            json.dumps(training_details, ensure_ascii=False) if training_details else "",
            paper_text[:2000],
        ]
    ).lower()

    for markers, inferred_loss, confidence, justification in INFERENCE_RULES:
        if any(marker in architecture_text for marker in markers):
            losses["primary"] = inferred_loss
            losses["inferred"] = True
            losses["confidence"] = confidence
            training_details["loss_inference"] = justification
            paper_analysis["loss_function"] = paper_analysis.get("loss_function") or inferred_loss
            augmentations = _extract_augmentations(training_details, paper_text)
            if augmentations:
                training_details["augmentation"] = augmentations
            paper_analysis["losses"] = losses
            paper_analysis["training_objective"] = _generate_training_objective(paper_analysis)
            paper_analysis["training_details"] = training_details
            return paper_analysis

    architecture_summary = paper_analysis.get("model_architecture") or ", ".join(
        (paper_analysis.get("architectures") or {}).get("proposed", [])
    )
    if architecture_summary:
        try:
            fallback = generate_json_with_reasoning_fallback(
                _build_loss_fallback_prompt(
                    f"Architecture: {architecture_summary}\nTraining description: {json.dumps(training_details, ensure_ascii=False)}"
                )
            )
            inferred_loss = fallback.get("loss")
            if inferred_loss:
                losses["primary"] = inferred_loss
                losses["inferred"] = True
                losses["confidence"] = _coerce_confidence(fallback.get("confidence"), 0.4)
                training_details["loss_inference"] = fallback.get("justification")
                paper_analysis["loss_function"] = paper_analysis.get("loss_function") or inferred_loss
        except Exception:
            logger.exception("training_detail_inference_failed architecture_summary=%s", architecture_summary)

    confirmed_loss = losses.get("primary")
    if architecture_summary:
        try:
            confirmation = generate_json_with_reasoning_fallback(
                _build_loss_confirmation_prompt(
                    architecture_summary=architecture_summary,
                    dataset=paper_analysis.get("dataset"),
                    training_objective=paper_analysis.get("training_objective"),
                    training_details=training_details,
                    current_loss=confirmed_loss,
                )
            )
            candidate_loss = str(confirmation.get("loss") or "").strip()
            candidate_confidence = _coerce_confidence(confirmation.get("confidence"), losses.get("confidence", 0.0) or 0.0)
            explicitly_stated = bool(confirmation.get("explicitly_stated", False))
            if candidate_loss and candidate_loss.lower() != "unknown":
                if explicitly_stated or candidate_confidence >= max(0.75, float(losses.get("confidence", 0.0) or 0.0)):
                    losses["primary"] = candidate_loss
                    losses["inferred"] = not explicitly_stated
                    losses["confidence"] = candidate_confidence
                    training_details["loss_inference"] = confirmation.get("justification")
                    paper_analysis["loss_function"] = paper_analysis.get("loss_function") or candidate_loss
        except Exception:
            logger.exception("training_loss_confirmation_failed architecture_summary=%s", architecture_summary)

    augmentations = _extract_augmentations(training_details, paper_text)
    if augmentations:
        training_details["augmentation"] = augmentations
    paper_analysis["losses"] = losses
    paper_analysis["training_objective"] = _generate_training_objective(paper_analysis)
    paper_analysis["training_details"] = training_details or None
    return paper_analysis


def _normalize_analysis_result(result: dict) -> dict:
    model_architecture = result.get("model_architecture")
    loss_function = result.get("loss_function")
    optimizer = result.get("optimizer")
    return {
        "model_architecture": model_architecture,
        "architectures": _normalize_architectures(result.get("architectures"), model_architecture),
        "dataset": result.get("dataset"),
        "loss_function": loss_function,
        "losses": _normalize_losses(result.get("losses"), loss_function),
        "training_objective": result.get("training_objective"),
        "optimizer": optimizer,
        "optimizers": _normalize_optimizers(result.get("optimizers"), optimizer),
        "training_details": _normalize_training_details(result.get("training_details")),
        "evaluation_metrics": _normalize_list_field(result.get("evaluation_metrics")),
        "contributions": _normalize_list_field(result.get("contributions")),
    }


def _coerce_to_mapping(value) -> dict | None:
    """Try to coerce a value that should be a dict into one, returning None if it can't be."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, Mapping):
                    return dict(parsed)
            except json.JSONDecodeError:
                pass
    return None


def _merge_analysis_payloads(*payloads: dict) -> dict:
    merged: dict = {}
    for payload in payloads:
        for key, value in payload.items():
            if value in (None, "", [], {}):
                continue
            if key in {"architectures", "losses", "optimizers"}:
                coerced = _coerce_to_mapping(value)
                if coerced is not None:
                    existing = dict(merged.get(key) or {})
                    for subkey, subvalue in coerced.items():
                        if isinstance(subvalue, list):
                            existing[subkey] = list(dict.fromkeys((existing.get(subkey) or []) + subvalue))
                        elif subvalue not in (None, "", [], {}):
                            existing[subkey] = subvalue
                    merged[key] = existing
                elif key not in merged or merged[key] in (None, "", [], {}):
                    # LLM returned a plain string; store as fallback string only if nothing better exists
                    merged.setdefault(key, {})
            elif key == "training_details":
                coerced = _coerce_to_mapping(value)
                if coerced is not None:
                    existing = dict(merged.get(key) or {})
                    existing.update(coerced)
                    merged[key] = existing
                elif isinstance(value, str) and value.strip():
                    existing = dict(merged.get("training_details") or {})
                    existing.setdefault("summary", value.strip())
                    merged["training_details"] = existing
            elif key in {"evaluation_metrics", "contributions"} and isinstance(value, list):
                merged[key] = list(dict.fromkeys((merged.get(key) or []) + value))
            elif key not in merged or merged[key] in (None, "", [], {}):
                merged[key] = value
    return merged


def _extract_metrics_from_chunks(chunk_payloads: list[dict]) -> list[str]:
    metrics: list[str] = []
    for payload in chunk_payloads:
        text = str(payload.get("content") or "").lower()
        for candidate in METRIC_CANDIDATES:
            if candidate in text and candidate not in metrics:
                metrics.append(candidate)
    return [metric.upper() if metric == "wer" else metric for metric in metrics[:8]]


def _build_chunk_payloads(chunks: list[PaperChunk]) -> list[dict]:
    payloads: list[dict] = []
    total_chunks = len(chunks)
    for chunk in chunks:
        structure = build_chunk_structure(
            chunk_id=str(chunk.id),
            chunk_index=chunk.chunk_index,
            section_name=chunk.section_name,
            subsection_name=chunk.subsection_name,
            content=chunk.content,
            total_chunks=total_chunks,
        )
        if not is_quality_chunk(structure):
            continue
        payloads.append(
            {
                **structure,
                "paper_id": str(chunk.paper_id),
                "content": chunk.content,
                "page_number": None,
                "score": 0.0,
            }
        )
    return payloads


def _build_base_analysis_payload(
    selected_chunk_payloads: list[dict],
    inferred_structure: dict,
    domain: str,
) -> dict:
    role_distribution: dict[str, int] = {}
    for payload in selected_chunk_payloads:
        role = str(payload.get("role") or "other")
        role_distribution[role] = role_distribution.get(role, 0) + 1

    contributions = [
        str(item.get("summary", "")).strip()
        for item in inferred_structure.get("key_ideas", [])
        if isinstance(item, Mapping) and str(item.get("summary", "")).strip()
    ][:6]

    return {
        "model_architecture": None,
        "architectures": None,
        "dataset": None,
        "loss_function": None,
        "losses": None,
        "training_objective": None,
        "optimizer": None,
        "optimizers": None,
        "training_details": {
            "chunk_role_distribution": role_distribution,
            "chunk_count": len(selected_chunk_payloads),
        },
        "evaluation_metrics": _extract_metrics_from_chunks(selected_chunk_payloads),
        "contributions": contributions,
        "domain": domain,
        "inferred_structure": inferred_structure,
        "synthesis_output": None,
        "synthesis_generated_at": None,
    }


def build_domain_view(
    db: Session,
    paper_id: uuid.UUID,
    domain: str | None,
    inferred_structure: dict | list | None = None,
) -> dict:
    normalized_domain = (domain or "general").lower()
    if normalized_domain not in {"ml", "theory", "systems", "security", "networks"}:
        return {}

    chunks = db.scalars(
        select(PaperChunk).where(PaperChunk.paper_id == paper_id).order_by(PaperChunk.chunk_index)
    ).all()
    if not chunks:
        return {}

    chunk_payloads = _build_chunk_payloads(chunks)
    inferred = inferred_structure if isinstance(inferred_structure, Mapping) else None
    return derive_domain_fields(normalized_domain, chunk_payloads, inferred_structure=inferred)


def _apply_ml_compatibility_view(base_payload: dict, domain_view: dict) -> dict:
    ml_payload = domain_view.get("ml") if isinstance(domain_view, Mapping) else None
    if not isinstance(ml_payload, Mapping):
        return base_payload

    merged = dict(base_payload)
    for key in [
        "model_architecture",
        "architectures",
        "dataset",
        "loss_function",
        "losses",
        "training_objective",
        "optimizer",
        "optimizers",
        "training_details",
        "evaluation_metrics",
        "contributions",
    ]:
        value = ml_payload.get(key)
        if value in (None, "", [], {}):
            continue
        merged[key] = value
    return merged


def analyze_paper(db: Session, paper_id: uuid.UUID) -> dict:
    paper = db.get(Paper, paper_id)
    if paper is None:
        raise ValueError(f"paper not found: {paper_id}")

    chunks = db.scalars(
        select(PaperChunk).where(PaperChunk.paper_id == paper_id).order_by(PaperChunk.chunk_index)
    ).all()
    sections = db.scalars(
        select(PaperSection).where(PaperSection.paper_id == paper_id).order_by(PaperSection.section_order)
    ).all()
    if not chunks:
        raise ValueError(f"no chunks found for paper: {paper_id}")

    selected_chunks = _select_relevant_chunks(chunks)
    paper_text = "\n\n".join(chunk.content for chunk in chunks)
    all_chunk_payloads = _build_chunk_payloads(chunks)
    selected_chunk_payloads = _build_chunk_payloads(selected_chunks)
    inferred_structure = build_inferred_structure(selected_chunk_payloads, max_items=6)
    synthesis_structure = _build_synthesis_structure(inferred_structure, all_chunk_payloads)

    domain = (paper.domain or "").strip().lower()
    domain_confidence = float(paper.domain_confidence or 0.0)
    if not domain:
        domain_result = detect_domain([payload["content"] for payload in selected_chunk_payloads[:20]])
        domain = str(domain_result.get("domain") or "general")
        domain_confidence = float(domain_result.get("confidence") or 0.0)
        paper.domain = domain
        paper.domain_confidence = domain_confidence

    section_results: dict[str, dict | None] = {key: None for key in SECTION_AGENT_KEYS}
    section_llm_calls = 0
    agent_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(len(SECTION_AGENT_KEYS), SECTION_AGENT_MAX_WORKERS)) as executor:
        future_map = {
            executor.submit(
                run_section_agent,
                section_key,
                _get_section_chunks(synthesis_structure, section_key),
                paper_id,
                synthesis_structure,
            ): section_key
            for section_key in SECTION_AGENT_KEYS
        }
        for future in as_completed(future_map):
            section_key = future_map[future]
            try:
                agent_result = future.result()
            except Exception:
                logger.exception("section_agent_failed paper_id=%s section_key=%s", paper_id, section_key)
                section_results[section_key] = None
                continue
            section_results[section_key] = agent_result.get("result") if isinstance(agent_result, Mapping) else None
            section_llm_calls += int(agent_result.get("llm_calls") or 0) if isinstance(agent_result, Mapping) else 0

    analysis_status = _compute_analysis_status(section_results)
    section_metadata = {
        key: value
        for key, value in section_results.items()
    }
    methods_section = section_results.get("methods") if isinstance(section_results.get("methods"), Mapping) else {}
    methods_synthesis = str(methods_section.get("synthesis") or "")
    try:
        inferred_structure = _ensure_method_equations(
            inferred_structure,
            methods_synthesis,
            paper_id,
        )
    except Exception:
        logger.exception("equation_fallback_failed paper_id=%s", paper_id)
    agent_elapsed_ms = round((time.perf_counter() - agent_start) * 1000, 2)
    logger.info(
        "section_agent_loop_complete paper_id=%s total_llm_calls=%s wall_time_ms=%s status=%s",
        paper_id,
        section_llm_calls,
        agent_elapsed_ms,
        analysis_status["status"],
    )

    analysis_result = _build_base_analysis_payload(selected_chunk_payloads, inferred_structure, domain)
    if any(value is not None for value in section_metadata.values()):
        analysis_result["synthesis_output"] = section_metadata
        analysis_result["synthesis_generated_at"] = datetime.now(timezone.utc)
    domain_view = derive_domain_fields(domain, selected_chunk_payloads, inferred_structure=inferred_structure)
    compatibility_payload = _apply_ml_compatibility_view(analysis_result, domain_view)

    tokenizer = get_tokenizer()
    context = "\n\n".join(chunk.content for chunk in selected_chunks)
    prompt_length = len(
        tokenizer(
            context,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
    )

    repository_info = discover_repositories(compatibility_payload, paper_text, paper.title)
    reproducibility_result = compute_reproducibility_score(
        compatibility_payload, repository_info, context, domain=domain
    )
    logger.info(
        "paper_analysis_payload_ready paper_id=%s domain=%s domain_confidence=%s chunks_used=%s prompt_length=%s payload=%s",
        paper_id,
        domain,
        domain_confidence,
        len(selected_chunks),
        prompt_length,
        json.dumps(analysis_result, default=str),
    )

    analysis = PaperAnalysis(paper_id=paper_id, **analysis_result)
    db.add(analysis)
    repository_rows = [
        PaperRepository(
            paper_id=paper_id,
            repo_url=repo["url"],
            source=repo["source"],
            confidence=float(repo["confidence"]),
        )
        for repo in repository_info.get("repositories", [])
    ]
    for repository in repository_rows:
        db.add(repository)

    reproducibility_score = ReproducibilityScore(
        paper_id=paper_id,
        dataset_available=reproducibility_result["dataset_available"],
        code_available=reproducibility_result["code_available"],
        hyperparameter_completeness=float(reproducibility_result["hyperparameter_completeness"]),
        training_detail_score=float(reproducibility_result["training_detail_score"]),
        evaluation_protocol_score=float(reproducibility_result["evaluation_protocol_score"]),
        overall_score=float(reproducibility_result["overall_score"]),
        summary=reproducibility_result.get("summary"),
        evidence=reproducibility_result.get("evidence"),
    )
    db.add(reproducibility_score)
    try:
        db.commit()
        db.refresh(analysis)
    except Exception:
        db.rollback()
        logger.exception(
            "paper_analysis_commit_failed paper_id=%s payload=%s",
            paper_id,
            json.dumps(analysis_result, default=str),
        )
        raise

    logger.info(
        "paper_analysis_complete paper_id=%s chunks_used=%s prompt_length=%s analysis_result=%s",
        paper_id,
        len(selected_chunks),
        prompt_length,
        json.dumps(analysis_result, default=str),
    )

    response_payload = compatibility_payload

    return {
        "id": str(analysis.id),
        "paper_id": str(paper_id),
        **response_payload,
        **domain_view,
        "analysis_status": analysis_status,
        "synthesis_output": analysis_result.get("synthesis_output"),
        "synthesis_generated_at": (
            analysis_result["synthesis_generated_at"].isoformat()
            if analysis_result.get("synthesis_generated_at")
            else None
        ),
        "repository_info": repository_info,
        "reproducibility": reproducibility_result,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
    }


def _format_structured_answer(field_name: str, value) -> str:
    if field_name == "dataset":
        return str(value)
    if field_name == "training_objective":
        return str(value)

    if field_name == "losses" and isinstance(value, Mapping):
        parts: list[str] = []
        if value.get("primary"):
            parts.append(f"Primary loss: {value['primary']}")
        if value.get("auxiliary"):
            parts.append(f"Auxiliary losses: {', '.join(value['auxiliary'])}")
        if value.get("baseline"):
            parts.append(f"Baseline losses: {', '.join(value['baseline'])}")
        if value.get("inferred"):
            parts.append(f"Inferred: yes (confidence {float(value.get('confidence', 0.0)):.2f})")
        return "; ".join(parts)

    if field_name == "optimizers" and isinstance(value, Mapping):
        parts: list[str] = []
        if value.get("primary"):
            parts.append(f"Primary optimizer: {value['primary']}")
        if value.get("baseline"):
            parts.append(f"Baseline optimizers: {', '.join(value['baseline'])}")
        return "; ".join(parts)

    if field_name == "architectures" and isinstance(value, Mapping):
        parts: list[str] = []
        if value.get("proposed"):
            parts.append(f"Proposed: {', '.join(value['proposed'])}")
        if value.get("baseline"):
            parts.append(f"Baselines: {', '.join(value['baseline'])}")
        return "; ".join(parts)

    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def get_structured_answer_if_available(db: Session, paper_id: uuid.UUID, query: str) -> dict | None:
    lowered_query = query.lower()
    if any(keyword in lowered_query for keyword in REPRODUCIBILITY_QUERY_KEYWORDS):
        latest_score = (
            db.query(ReproducibilityScore)
            .filter(ReproducibilityScore.paper_id == paper_id)
            .order_by(ReproducibilityScore.created_at.desc())
            .first()
        )
        if latest_score is not None:
            answer = format_reproducibility_answer(latest_score)
            return {
                "answer": answer,
                "sources": [
                    {
                        "paper_id": str(paper_id),
                        "section_name": "reproducibility_scores",
                        "subsection_name": "overall_score",
                        "page_number": None,
                        "score": 1.0,
                        "content": answer,
                        "content_snippet": answer[:400],
                    }
                ],
                "analysis_hits": {
                    "reproducibility": {
                        "overall_score": latest_score.overall_score,
                        "dataset_available": latest_score.dataset_available,
                        "code_available": latest_score.code_available,
                    }
                },
            }

    latest_analysis = (
        db.query(PaperAnalysis)
        .filter(PaperAnalysis.paper_id == paper_id)
        .order_by(PaperAnalysis.created_at.desc())
        .first()
    )
    if latest_analysis is None:
        return None

    domain_view = build_domain_view(
        db,
        paper_id,
        getattr(latest_analysis, "domain", None),
        getattr(latest_analysis, "inferred_structure", None),
    )
    ml_payload = domain_view.get("ml") if isinstance(domain_view, Mapping) else None

    for field_name, keywords in STRUCTURED_FIELD_MAP.items():
        if not any(keyword in lowered_query for keyword in keywords):
            continue

        value = getattr(latest_analysis, field_name, None)
        if not value and isinstance(ml_payload, Mapping):
            value = ml_payload.get(field_name)
        if not value:
            if field_name == "architectures":
                value = latest_analysis.model_architecture
            elif field_name == "losses":
                value = latest_analysis.loss_function
            elif field_name == "optimizers":
                value = latest_analysis.optimizer

        if not value and isinstance(ml_payload, Mapping):
            if field_name == "architectures":
                value = ml_payload.get("model_architecture")
            elif field_name == "losses":
                value = ml_payload.get("loss_function")
            elif field_name == "optimizers":
                value = ml_payload.get("optimizer")

        if not value:
            return None

        answer = _format_structured_answer(field_name, value)
        return {
            "answer": answer,
            "sources": [
                {
                    "paper_id": str(paper_id),
                    "section_name": "paper_analysis",
                    "subsection_name": field_name,
                    "page_number": None,
                    "score": 1.0,
                    "content": answer,
                    "content_snippet": answer[:400],
                }
            ],
            "analysis_hits": {field_name: value},
        }

    return None
