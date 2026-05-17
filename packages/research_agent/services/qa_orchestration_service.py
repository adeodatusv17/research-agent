import logging
import math
import re
from collections import Counter
from typing import Iterable

from research_agent.agents.state.qa_state import (
    AnswerTiers,
    CritiqueReport,
    EvaluationReport,
    EvidenceDiagnostics,
    ExecutionPlan,
    GroundedClaim,
    PlannerSubtask,
    RetrievalAttempt,
    RetrievalParameters,
    VerificationReport,
)
from research_agent.services.retrieval_service import classify_query
from research_agent.tools.embedder import get_tokenizer
from research_agent.tools.gemini_client import generate_json


logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}

EVIDENCE_FIELD_RULES = {
    "methodology": {
        "keywords": ["method", "approach", "architecture", "algorithm", "module", "pipeline"],
        "roles": {"method", "algorithm", "implementation", "theory"},
        "sections": {"method", "methodology", "algorithm", "implementation"},
    },
    "dataset": {
        "keywords": ["dataset", "corpus", "benchmark", "data", "train set", "test set"],
        "roles": {"evaluation", "implementation", "idea"},
        "sections": {"experiments", "results", "method", "abstract"},
    },
    "evaluation": {
        "keywords": ["result", "evaluation", "metric", "benchmark", "ablation", "performance"],
        "roles": {"evaluation", "discussion"},
        "sections": {"results", "evaluation", "experiments", "discussion"},
    },
    "reproducibility": {
        "keywords": ["code", "repository", "reproduce", "training", "hyperparameter", "implementation"],
        "roles": {"implementation", "discussion", "method"},
        "sections": {"method", "experiments", "discussion", "conclusion"},
    },
    "comparison": {
        "keywords": ["baseline", "compare", "versus", "vs", "outperform", "better than"],
        "roles": {"evaluation", "discussion"},
        "sections": {"results", "evaluation", "discussion", "related_work"},
    },
    "limitations": {
        "keywords": ["limitation", "future work", "challenge", "failure", "tradeoff"],
        "roles": {"discussion", "evaluation"},
        "sections": {"discussion", "conclusion", "results"},
    },
}

CONTRADICTION_MARKERS = [
    "however",
    "but",
    "although",
    "worse",
    "fails",
    "failure",
    "limitation",
    "in contrast",
    "whereas",
]

FORMULA_QUERY_KEYWORDS = [
    "formula",
    "formulas",
    "equation",
    "equations",
    "mathematical expression",
    "math expression",
]


def _append_trace(state: dict, event: str) -> list[str]:
    trace = list(state.get("execution_trace", []))
    trace.append(event)
    return trace


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) if token not in STOPWORDS]


def _average(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _is_formula_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in FORMULA_QUERY_KEYWORDS)


def _allows_general_background(query: str, diagnostics: EvidenceDiagnostics) -> bool:
    lowered = query.lower()
    explicit_background_markers = [
        "what is",
        "explain",
        "overview",
        "background",
        "in general",
        "generally",
        "how does",
    ]
    if any(marker in lowered for marker in explicit_background_markers):
        return True
    if _is_formula_query(query):
        return False
    return float(diagnostics.get("retrieval_strength", 0.0)) < 0.35


def _response_mode(query: str) -> str:
    if _is_formula_query(query):
        return "equation_extraction"
    lowered = query.lower()
    if any(marker in lowered for marker in ["list", "show", "provide", "give me", "which are", "what are"]):
        return "literal_extraction"
    return "standard"


def _derive_required_evidence(query: str, query_type: str) -> list[str]:
    lowered = query.lower()
    required: list[str] = []
    if query_type == "method":
        required.append("methodology")
    if query_type == "evaluation":
        required.extend(["evaluation", "dataset"])
    if query_type == "comparison":
        required.extend(["comparison", "evaluation"])
    if "dataset" in lowered or "benchmark" in lowered or "corpus" in lowered:
        required.append("dataset")
    if "reproduc" in lowered or "repository" in lowered or "code" in lowered:
        required.append("reproducibility")
    if "limitation" in lowered or "weakness" in lowered or "future work" in lowered:
        required.append("limitations")
    if not required:
        required.append("methodology" if query_type == "method" else "evaluation")
    return list(dict.fromkeys(required))


def _build_default_subtasks(query: str, required_evidence: list[str]) -> list[PlannerSubtask]:
    field_titles = {
        "methodology": "Methodology Extraction",
        "dataset": "Dataset Inspection",
        "evaluation": "Evaluation Review",
        "reproducibility": "Reproducibility Check",
        "comparison": "Baseline Comparison",
        "limitations": "Limitations Review",
    }
    subtasks: list[PlannerSubtask] = []
    for index, field_name in enumerate(required_evidence, start=1):
        subtasks.append(
            {
                "subtask_id": f"task_{index}",
                "title": field_titles.get(field_name, field_name.title()),
                "question": query,
                "evidence_types": [field_name],
                "priority": "high" if index <= 2 else "medium",
            }
        )
    return subtasks


def _estimate_query_complexity(query: str) -> tuple[int, str]:
    lowered = query.lower()
    clauses = sum(lowered.count(marker) for marker in [" and ", " or ", " while ", " versus ", " vs "])
    comparison_hits = sum(lowered.count(marker) for marker in ["compare", "difference", "baseline", "tradeoff"])
    critique_hits = sum(lowered.count(marker) for marker in ["why", "weakness", "limitation", "critique", "reproduc"])
    structure_hits = len(re.findall(r"[,:;]", query))
    token_count = len(_tokens(query))
    score = 0
    if token_count >= 14:
        score += 1
    if token_count >= 24:
        score += 1
    if clauses >= 1:
        score += 1
    if clauses >= 2:
        score += 1
    if comparison_hits:
        score += 1
    if critique_hits:
        score += 1
    if structure_hits >= 2:
        score += 1
    if score >= 5:
        return score, "complex"
    if score >= 3:
        return score, "medium"
    return score, "simple"


def _build_default_plan(query: str, query_type: str, complexity: str) -> ExecutionPlan:
    required_evidence = _derive_required_evidence(query, query_type)
    return {
        "strategy": "direct" if complexity == "simple" else "guided",
        "complexity": complexity,
        "decomposition_rationale": "Deterministic planner fallback based on query structure and evidence requirements.",
        "required_evidence": required_evidence,
        "missing_information": [],
        "subtasks": _build_default_subtasks(query, required_evidence),
    }


def analyze_query_step(state: dict) -> dict:
    query = state["query"].strip()
    query_type = classify_query(query)
    complexity_score, complexity_label = _estimate_query_complexity(query)
    orchestration_level = 1
    if complexity_score >= 3:
        orchestration_level = 2
    if complexity_score >= 5:
        orchestration_level = 3

    retry_budget = 0 if orchestration_level == 1 else 1 if orchestration_level == 2 else 2
    revision_budget = 0 if orchestration_level < 3 else 1
    should_plan = orchestration_level >= 3
    should_verify = orchestration_level >= 2
    should_critique = orchestration_level >= 3
    query_analysis = {
        "keywords": [token.lower() for token in query.replace("?", " ").split() if token.strip()],
        "length": len(query),
        "token_count": len(_tokens(query)),
        "query_type": query_type,
        "complexity_score": complexity_score,
        "complexity": complexity_label,
    }
    retrieval_parameters: RetrievalParameters = {
        "section_top_k": 3,
        "subsection_top_k": 6,
        "semantic_top_k": 18,
        "final_top_k": 12,
        "include_references": "reference" in query.lower() or "citation" in query.lower(),
        "disable_subsection_filter": False,
        "expanded_scope": False,
    }

    logger.info(
        "qa_orchestration_analysis paper_id=%s query_type=%s complexity=%s level=%s should_plan=%s should_verify=%s",
        state.get("paper_id"),
        query_type,
        complexity_label,
        orchestration_level,
        should_plan,
        should_verify,
    )
    return {
        **state,
        "query": query,
        "active_query": query,
        "query_type": query_type,
        "query_analysis": query_analysis,
        "orchestration_level": orchestration_level,
        "should_plan": should_plan,
        "should_verify": should_verify,
        "should_critique": should_critique,
        "retry_budget": retry_budget,
        "revision_budget": revision_budget,
        "retry_count": int(state.get("retry_count", 0)),
        "revision_count": int(state.get("revision_count", 0)),
        "execution_plan": _build_default_plan(query, query_type, complexity_label),
        "retrieval_parameters": retrieval_parameters,
        "retrieval_attempts": [],
        "execution_trace": _append_trace(state, f"query_analysis:l{orchestration_level}:{complexity_label}"),
    }


def planner_step(state: dict) -> dict:
    if not state.get("should_plan"):
        return {
            **state,
            "execution_trace": _append_trace(state, "planner:skipped"),
        }

    prompt = (
        "Decompose the research question into a compact machine-readable execution plan.\n"
        "Return JSON only with keys: strategy, complexity, decomposition_rationale, required_evidence, "
        "missing_information, subtasks.\n"
        "Each subtask should contain: subtask_id, title, question, evidence_types, priority.\n"
        "Keep the plan bounded to at most 5 subtasks and reuse evidence types from: "
        "[methodology, dataset, evaluation, reproducibility, comparison, limitations].\n\n"
        f"Question: {state['query']}\n"
        f"Query type: {state['query_type']}\n"
        f"Deterministic default plan: {state['execution_plan']}"
    )
    try:
        plan_payload = generate_json(prompt)
    except Exception:
        logger.exception("qa_planner_failed paper_id=%s", state.get("paper_id"))
        plan_payload = {}

    default_plan = state["execution_plan"]
    subtasks = plan_payload.get("subtasks") if isinstance(plan_payload.get("subtasks"), list) else []
    normalized_plan: ExecutionPlan = {
        "strategy": str(plan_payload.get("strategy") or default_plan.get("strategy") or "guided"),
        "complexity": str(plan_payload.get("complexity") or default_plan.get("complexity") or "complex"),
        "decomposition_rationale": str(
            plan_payload.get("decomposition_rationale")
            or default_plan.get("decomposition_rationale")
            or "Planner fallback."
        ).strip(),
        "required_evidence": [
            str(item)
            for item in (plan_payload.get("required_evidence") or default_plan.get("required_evidence") or [])
            if str(item).strip()
        ],
        "missing_information": [
            str(item)
            for item in (plan_payload.get("missing_information") or [])
            if str(item).strip()
        ],
        "subtasks": [],
    }
    for fallback_index, subtask in enumerate(subtasks[:5], start=1):
        if not isinstance(subtask, dict):
            continue
        normalized_plan["subtasks"].append(
            {
                "subtask_id": str(subtask.get("subtask_id") or f"task_{fallback_index}"),
                "title": str(subtask.get("title") or f"Task {fallback_index}"),
                "question": str(subtask.get("question") or state["query"]),
                "evidence_types": [
                    str(item)
                    for item in (subtask.get("evidence_types") or [])
                    if str(item).strip()
                ]
                or ["methodology"],
                "priority": str(subtask.get("priority") or "medium"),
            }
        )
    if not normalized_plan["subtasks"]:
        normalized_plan["subtasks"] = default_plan["subtasks"]
    if not normalized_plan["required_evidence"]:
        normalized_plan["required_evidence"] = default_plan["required_evidence"]

    logger.info(
        "qa_planner_complete paper_id=%s subtasks=%s required_evidence=%s",
        state.get("paper_id"),
        len(normalized_plan["subtasks"]),
        normalized_plan["required_evidence"],
    )
    return {
        **state,
        "execution_plan": normalized_plan,
        "execution_trace": _append_trace(state, f"planner:{len(normalized_plan['subtasks'])}_subtasks"),
    }


def should_run_planner(state: dict) -> str:
    return "planner" if state.get("should_plan") else "retrieve_sections"


def _covers_evidence(field_name: str, chunks: list[dict]) -> bool:
    rule = EVIDENCE_FIELD_RULES.get(field_name)
    if rule is None:
        return False
    keywords = rule["keywords"]
    roles = rule["roles"]
    sections = rule["sections"]
    for chunk in chunks:
        content = str(chunk.get("content") or "").lower()
        section_name = str(chunk.get("section_name") or "").lower()
        role = str(chunk.get("role") or "")
        if role in roles:
            return True
        if section_name in sections:
            return True
        if any(keyword in content for keyword in keywords):
            return True
    return False


def _build_evidence_diagnostics(state: dict) -> EvidenceDiagnostics:
    filtered_chunks = list(state.get("filtered_chunks", []))
    selected_sections = list(state.get("selected_sections", []))
    required_evidence = list((state.get("execution_plan") or {}).get("required_evidence", []))
    chunk_count = len(filtered_chunks)
    avg_score = _average(float(chunk.get("score", 0.0)) for chunk in filtered_chunks)
    avg_words = _average(len(_tokens(str(chunk.get("content") or ""))) for chunk in filtered_chunks)
    density = _clip((avg_score * 0.55) + (_clip(avg_words / 90.0) * 0.45))
    unique_sections = {str(chunk.get("section_name") or "unknown") for chunk in filtered_chunks}
    unique_roles = {str(chunk.get("role") or "other") for chunk in filtered_chunks}
    retrieval_diversity = _clip(
        0.5 * _clip(_safe_div(len(unique_sections), max(1, len(selected_sections) or len(unique_sections) or 1)))
        + 0.5 * _clip(_safe_div(len(unique_roles), 4.0))
    )
    contradiction_hits = 0
    for chunk in filtered_chunks:
        text = str(chunk.get("content") or "").lower()
        contradiction_hits += sum(text.count(marker) for marker in CONTRADICTION_MARKERS)
    contradiction_ratio = _clip(_safe_div(contradiction_hits, max(1, chunk_count * 3)))

    covered_evidence = {field_name: _covers_evidence(field_name, filtered_chunks) for field_name in required_evidence}
    missing_required_fields = [field_name for field_name, covered in covered_evidence.items() if not covered]
    section_coverage = _clip(
        _safe_div(sum(1 for covered in covered_evidence.values() if covered), len(covered_evidence))
        if covered_evidence
        else _safe_div(len(unique_sections), 3.0)
    )
    retrieval_strength = _clip(
        0.6 * avg_score + 0.25 * retrieval_diversity + 0.15 * (1.0 - contradiction_ratio)
    )
    weak_signals: list[str] = []
    if density < 0.45:
        weak_signals.append("low_evidence_density")
    if section_coverage < 0.55:
        weak_signals.append("incomplete_section_coverage")
    if contradiction_ratio > 0.3:
        weak_signals.append("high_contradiction_ratio")
    if chunk_count < 5:
        weak_signals.append("low_chunk_count")
    if missing_required_fields:
        weak_signals.append("missing_required_fields")

    should_retry = (
        int(state.get("retry_count", 0)) < int(state.get("retry_budget", 0))
        and bool(
            density < 0.45
            or section_coverage < 0.55
            or contradiction_ratio > 0.3
            or missing_required_fields
        )
    )
    sufficient = not should_retry and density >= 0.45 and section_coverage >= 0.55
    return {
        "evidence_density": round(density, 4),
        "section_coverage": round(section_coverage, 4),
        "contradiction_ratio": round(contradiction_ratio, 4),
        "retrieval_diversity": round(retrieval_diversity, 4),
        "retrieval_strength": round(retrieval_strength, 4),
        "missing_required_fields": missing_required_fields,
        "covered_evidence": covered_evidence,
        "weak_signals": weak_signals,
        "sufficient": sufficient,
        "should_retry": should_retry,
    }


def evidence_diagnostics_step(state: dict) -> dict:
    diagnostics = _build_evidence_diagnostics(state)
    logger.info(
        "qa_evidence_diagnostics paper_id=%s retry_count=%s should_retry=%s density=%s coverage=%s contradictions=%s missing=%s",
        state.get("paper_id"),
        state.get("retry_count", 0),
        diagnostics["should_retry"],
        diagnostics["evidence_density"],
        diagnostics["section_coverage"],
        diagnostics["contradiction_ratio"],
        diagnostics["missing_required_fields"],
    )
    return {
        **state,
        "evidence_diagnostics": diagnostics,
        "execution_trace": _append_trace(
            state,
            (
                "evidence_diagnostics:"
                f"density={diagnostics['evidence_density']}:coverage={diagnostics['section_coverage']}:retry={diagnostics['should_retry']}"
            ),
        ),
    }


def route_after_diagnostics(state: dict) -> str:
    diagnostics = state.get("evidence_diagnostics", {})
    return "adaptive_retry" if diagnostics.get("should_retry") else "generate_answer"


def adaptive_retry_step(state: dict) -> dict:
    diagnostics = state.get("evidence_diagnostics", {})
    retry_count = int(state.get("retry_count", 0)) + 1
    retrieval_parameters = dict(state.get("retrieval_parameters", {}))
    retrieval_parameters["section_top_k"] = int(retrieval_parameters.get("section_top_k", 3)) + 1
    retrieval_parameters["subsection_top_k"] = int(retrieval_parameters.get("subsection_top_k", 6)) + 2
    retrieval_parameters["semantic_top_k"] = int(retrieval_parameters.get("semantic_top_k", 18)) + 6
    retrieval_parameters["final_top_k"] = int(retrieval_parameters.get("final_top_k", 12)) + 2
    retrieval_parameters["disable_subsection_filter"] = True
    retrieval_parameters["expanded_scope"] = True

    missing = list(diagnostics.get("missing_required_fields", []))
    retry_suffix = " ".join(missing[:3]) if missing else "broader evidence"
    active_query = state["query"]
    if retry_suffix:
        active_query = f"{state['query']} {retry_suffix}"

    attempt: RetrievalAttempt = {
        "attempt": retry_count,
        "query": active_query,
        "reason": ", ".join(diagnostics.get("weak_signals", [])) or "insufficient evidence",
        "expanded_scope": True,
        "missing_required_fields": missing,
    }
    logger.info(
        "qa_adaptive_retry paper_id=%s retry_count=%s active_query=%s params=%s",
        state.get("paper_id"),
        retry_count,
        active_query,
        retrieval_parameters,
    )
    return {
        **state,
        "retry_count": retry_count,
        "active_query": active_query,
        "retrieval_parameters": retrieval_parameters,
        "retrieval_attempts": [*state.get("retrieval_attempts", []), attempt],
        "execution_trace": _append_trace(state, f"adaptive_retry:{retry_count}"),
    }


def _build_chunk_lookup(chunks: list[dict]) -> dict[str, dict]:
    return {str(chunk["chunk_id"]): chunk for chunk in chunks if chunk.get("chunk_id") is not None}


def _claim_support_strength(chunk_ids: list[str], chunk_lookup: dict[str, dict]) -> tuple[float, float]:
    supports = [chunk_lookup[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_lookup]
    if not supports:
        return 0.0, 0.0
    avg_score = _average(float(chunk.get("score", 0.0)) for chunk in supports)
    avg_rerank = _average(float(chunk.get("rerank_score", chunk.get("score", 0.0))) for chunk in supports)
    return avg_score, avg_rerank


def _claim_overlap_ratio(claim_text: str, supporting_chunks: list[dict]) -> float:
    claim_tokens = set(_tokens(claim_text))
    if not claim_tokens or not supporting_chunks:
        return 0.0
    support_tokens: set[str] = set()
    for chunk in supporting_chunks:
        support_tokens.update(_tokens(str(chunk.get("content") or "")))
    overlap = len(claim_tokens & support_tokens)
    return _clip(_safe_div(overlap, len(claim_tokens)))


def _derive_claim_grounding(
    claim_text: str,
    tier: str,
    supporting_chunk_ids: list[str],
    chunk_lookup: dict[str, dict],
    diagnostics: EvidenceDiagnostics,
    claim_index: int,
) -> GroundedClaim:
    supporting_chunks = [chunk_lookup[chunk_id] for chunk_id in supporting_chunk_ids if chunk_id in chunk_lookup]
    avg_score, avg_rerank = _claim_support_strength(supporting_chunk_ids, chunk_lookup)
    overlap_ratio = _claim_overlap_ratio(claim_text, supporting_chunks)
    contradiction_count = math.ceil(len(supporting_chunks) * float(diagnostics.get("contradiction_ratio", 0.0)))
    evidence_coverage = _clip(_safe_div(len(supporting_chunks), max(1, min(3, len(chunk_lookup)))))
    confidence = _clip(
        0.4 * avg_score
        + 0.2 * avg_rerank
        + 0.2 * overlap_ratio
        + 0.1 * float(diagnostics.get("section_coverage", 0.0))
        + 0.1 * (1.0 - float(diagnostics.get("contradiction_ratio", 0.0)))
    )
    support_status = "supported" if supporting_chunks and overlap_ratio >= 0.05 else "weak"
    return {
        "claim_id": f"claim_{claim_index}",
        "claim_text": claim_text.strip(),
        "tier": tier,
        "supporting_chunk_ids": [chunk["chunk_id"] for chunk in supporting_chunks if chunk.get("chunk_id")],
        "confidence": round(confidence, 4),
        "evidence_coverage": round(evidence_coverage, 4),
        "contradiction_count": contradiction_count,
        "support_strength": round(_clip(0.5 * avg_score + 0.5 * avg_rerank), 4),
        "support_status": support_status,
    }


def _render_context(filtered_chunks: list[dict], max_tokens: int = 2200) -> tuple[str, list[dict]]:
    tokenizer = get_tokenizer()
    current_tokens = 0
    parts: list[str] = []
    used_chunks: list[dict] = []
    for chunk in filtered_chunks:
        part = (
            f"[Chunk {chunk['chunk_id']} | Section {chunk.get('section_name') or 'unknown'} | "
            f"Subsection {chunk.get('subsection_name') or 'none'} | score={float(chunk.get('score', 0.0)):.4f}]\n"
            f"{chunk['content']}"
        )
        part_tokens = len(
            tokenizer(
                part,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )["input_ids"]
        )
        if current_tokens + part_tokens > max_tokens:
            break
        current_tokens += part_tokens
        parts.append(part)
        used_chunks.append(chunk)
    return "\n\n".join(parts), used_chunks


def _normalize_answer_tiers(value: object) -> AnswerTiers:
    if not isinstance(value, dict):
        return {
            "evidence_backed": [],
            "inferred_from_evidence": [],
            "general_background": [],
        }

    def normalize_list(key: str) -> list[str]:
        entries = value.get(key)
        if not isinstance(entries, list):
            return []
        return [str(item).strip() for item in entries if str(item).strip()]

    return {
        "evidence_backed": normalize_list("evidence_backed"),
        "inferred_from_evidence": normalize_list("inferred_from_evidence"),
        "general_background": normalize_list("general_background"),
    }


def _format_answer_from_tiers(answer_tiers: AnswerTiers) -> str:
    sections: list[str] = []
    labels = [
        ("evidence_backed", "Supported by Paper"),
        ("inferred_from_evidence", "Inferred From Paper Evidence"),
        ("general_background", "General Background"),
    ]
    for key, label in labels:
        entries = answer_tiers.get(key, [])
        if not entries:
            continue
        sections.append(f"**{label}**")
        sections.extend(f"- {entry}" for entry in entries)
        sections.append("")
    return "\n".join(section for section in sections if section is not None).strip()


def generate_grounded_answer_step(state: dict) -> dict:
    filtered_chunks = list(state.get("filtered_chunks", []))
    diagnostics = state.get("evidence_diagnostics", {})
    context, used_chunks = _render_context(filtered_chunks)
    required_evidence = list((state.get("execution_plan") or {}).get("required_evidence", []))
    allow_general_background = _allows_general_background(state["query"], diagnostics)
    response_mode = _response_mode(state["query"])
    prompt = (
        "Answer the research question using a 3-tier policy.\n"
        "Return JSON with keys: answer_tiers, claims.\n"
        "answer_tiers must contain exactly these keys:\n"
        "- evidence_backed: list[str]\n"
        "- inferred_from_evidence: list[str]\n"
        "- general_background: list[str]\n"
        "claims must be a list of objects with keys: claim_text, supporting_chunk_ids, tier.\n"
        "tier must be either evidence_backed or inferred_from_evidence.\n"
        "Each claim must cite 1-3 chunk ids from the context. Do not invent chunk ids.\n"
        "Do not put chunk-cited claims in general_background.\n"
        "General background is allowed only when explicitly permitted below.\n"
        f"General background allowed: {'yes' if allow_general_background else 'no'}.\n"
        f"Response mode: {response_mode}.\n"
        "If the question asks whether formulas/equations exist, do not stop at yes/no. "
        "List the actual formulas or equation expressions visible in the evidence under evidence_backed whenever possible.\n"
        "For extraction-style questions, prefer literal extraction over abstract summary.\n"
        f"Required evidence types: {required_evidence}\n\n"
        f"Question:\n{state['query']}\n\n"
        f"Context:\n{context}"
    )
    try:
        answer_payload = generate_json(prompt)
    except Exception:
        logger.exception("qa_grounded_generation_failed paper_id=%s", state.get("paper_id"))
        answer_payload = {}

    answer_tiers = _normalize_answer_tiers(answer_payload.get("answer_tiers"))
    raw_claims = answer_payload.get("claims") if isinstance(answer_payload.get("claims"), list) else []
    answer_text = _format_answer_from_tiers(answer_tiers)
    if not answer_text:
        answer_text = "Insufficient grounded evidence to produce a detailed answer."
    chunk_lookup = _build_chunk_lookup(used_chunks)
    grounded_claims: list[GroundedClaim] = []
    for index, raw_claim in enumerate(raw_claims[:8], start=1):
        if not isinstance(raw_claim, dict):
            continue
        claim_text = str(raw_claim.get("claim_text") or "").strip()
        tier = str(raw_claim.get("tier") or "evidence_backed").strip()
        supporting_chunk_ids = [str(item) for item in (raw_claim.get("supporting_chunk_ids") or []) if str(item).strip()]
        if not claim_text:
            continue
        grounded_claims.append(
            _derive_claim_grounding(claim_text, tier, supporting_chunk_ids, chunk_lookup, diagnostics, index)
        )
    if not grounded_claims and used_chunks:
        grounded_claims.append(
            _derive_claim_grounding(
                answer_text,
                "evidence_backed",
                [str(used_chunks[0]["chunk_id"])],
                chunk_lookup,
                diagnostics,
                1,
            )
        )

    used_ids = {chunk_id for claim in grounded_claims for chunk_id in claim.get("supporting_chunk_ids", [])}
    sources = [
        {
            "paper_id": chunk["paper_id"],
            "section_name": chunk.get("section_name"),
            "subsection_name": chunk.get("subsection_name"),
            "page_number": chunk.get("page_number"),
            "content": chunk["content"],
            "content_snippet": chunk["content"][:400],
            "score": chunk["score"],
            "chunk_id": chunk["chunk_id"],
            "rerank_score": chunk.get("rerank_score"),
        }
        for chunk in used_chunks
        if not used_ids or chunk["chunk_id"] in used_ids
    ]
    final_confidence = round(_average(claim["confidence"] for claim in grounded_claims), 4) if grounded_claims else 0.0
    logger.info(
        "qa_grounded_generation paper_id=%s grounded_claims=%s final_confidence=%s",
        state.get("paper_id"),
        len(grounded_claims),
        final_confidence,
    )
    return {
        **state,
        "context": context,
        "answer": answer_text,
        "answer_tiers": answer_tiers,
        "grounded_claims": grounded_claims,
        "final_confidence": final_confidence,
        "sources": sources,
        "execution_trace": _append_trace(state, f"generate_answer:{len(grounded_claims)}_claims"),
    }


def route_after_generation(state: dict) -> str:
    return "verify_answer" if state.get("should_verify") else "evaluate_answer"


def verify_answer_step(state: dict) -> dict:
    claims = list(state.get("grounded_claims", []))
    chunk_lookup = _build_chunk_lookup(state.get("filtered_chunks", []))
    unsupported_claim_ids: list[str] = []
    weak_claim_ids: list[str] = []
    issues: list[str] = []
    for claim in claims:
        chunk_ids = [str(item) for item in claim.get("supporting_chunk_ids", [])]
        supporting_chunks = [chunk_lookup[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_lookup]
        overlap_ratio = _claim_overlap_ratio(str(claim.get("claim_text") or ""), supporting_chunks)
        if not supporting_chunks:
            unsupported_claim_ids.append(str(claim.get("claim_id")))
            issues.append(f"{claim.get('claim_id')} has no valid supporting chunks.")
        elif overlap_ratio < 0.05:
            weak_claim_ids.append(str(claim.get("claim_id")))
            issues.append(f"{claim.get('claim_id')} has weak lexical evidence overlap.")
    supported_count = len(claims) - len(unsupported_claim_ids)
    report: VerificationReport = {
        "used_llm": False,
        "supported_claim_ratio": round(_safe_div(supported_count, len(claims)), 4) if claims else 0.0,
        "unsupported_claim_ids": unsupported_claim_ids,
        "weak_claim_ids": weak_claim_ids,
        "issues": issues,
        "status": "pass" if not unsupported_claim_ids and not weak_claim_ids else "review",
    }

    should_escalate_llm = bool(
        issues and state.get("orchestration_level", 1) >= 3 and len(issues) <= 4
    )
    if should_escalate_llm:
        focused_claims = [
            {
                "claim_id": claim.get("claim_id"),
                "claim_text": claim.get("claim_text"),
                "supporting_chunk_ids": claim.get("supporting_chunk_ids", []),
            }
            for claim in claims
            if claim.get("claim_id") in set(unsupported_claim_ids + weak_claim_ids)
        ]
        prompt = (
            "Verify whether each claim is supported by the cited evidence. Return JSON only with keys: "
            "unsupported_claim_ids, weak_claim_ids, issues.\n"
            "Do not assess style, only grounding.\n\n"
            f"Claims:\n{focused_claims}\n\n"
            f"Evidence excerpts:\n{state.get('sources', [])}"
        )
        try:
            verifier_payload = generate_json(prompt)
        except Exception:
            logger.exception("qa_verifier_llm_failed paper_id=%s", state.get("paper_id"))
            verifier_payload = {}
        report["used_llm"] = True
        report["unsupported_claim_ids"] = [str(item) for item in verifier_payload.get("unsupported_claim_ids", unsupported_claim_ids)]
        report["weak_claim_ids"] = [str(item) for item in verifier_payload.get("weak_claim_ids", weak_claim_ids)]
        report["issues"] = [str(item) for item in verifier_payload.get("issues", issues)]
        report["status"] = "pass" if not report["unsupported_claim_ids"] and not report["weak_claim_ids"] else "review"

    logger.info(
        "qa_verifier_complete paper_id=%s supported_ratio=%s issues=%s used_llm=%s",
        state.get("paper_id"),
        report["supported_claim_ratio"],
        len(report["issues"]),
        report["used_llm"],
    )
    return {
        **state,
        "verifier_report": report,
        "execution_trace": _append_trace(state, f"verify:{report['status']}"),
    }


def route_after_verification(state: dict) -> str:
    report = state.get("verifier_report", {})
    if state.get("should_critique") or report.get("issues"):
        return "critique_answer"
    return "evaluate_answer"


def critique_answer_step(state: dict) -> dict:
    diagnostics = state.get("evidence_diagnostics", {})
    verifier_report = state.get("verifier_report", {})
    issues = list(verifier_report.get("issues", []))
    revision_focus: list[str] = []
    if diagnostics.get("missing_required_fields"):
        missing = ", ".join(diagnostics["missing_required_fields"])
        issues.append(f"Missing evidence coverage for: {missing}.")
        revision_focus.append("add_missing_evidence_scope")
    if float(diagnostics.get("contradiction_ratio", 0.0)) > 0.25:
        issues.append("Answer should acknowledge contradictory or limiting evidence.")
        revision_focus.append("address_contradictions")
    if float(verifier_report.get("supported_claim_ratio", 1.0)) < 0.75:
        issues.append("Several claims are weakly supported.")
        revision_focus.append("tighten_claim_grounding")
    critique: CritiqueReport = {
        "used_llm": False,
        "issues": issues,
        "revision_focus": list(dict.fromkeys(revision_focus)),
        "should_revise": bool(revision_focus) and int(state.get("revision_count", 0)) < int(state.get("revision_budget", 0)),
        "severity": "high" if len(revision_focus) >= 2 else "medium" if revision_focus else "low",
    }
    should_call_llm = bool(
        state.get("should_critique")
        and (issues or state.get("orchestration_level", 1) >= 3)
        and int(state.get("revision_count", 0)) < int(state.get("revision_budget", 0))
    )
    if should_call_llm:
        prompt = (
            "Critique the grounded research answer for logical gaps, unsupported assumptions, and shallow reasoning.\n"
            "Return JSON only with keys: issues, revision_focus, should_revise, severity.\n"
            "Keep revision_focus items short and action-oriented.\n\n"
            f"Question: {state['query']}\n"
            f"Answer: {state.get('answer', '')}\n"
            f"Claims: {state.get('grounded_claims', [])}\n"
            f"Diagnostics: {diagnostics}\n"
            f"Verifier report: {verifier_report}"
        )
        try:
            critique_payload = generate_json(prompt)
        except Exception:
            logger.exception("qa_critic_llm_failed paper_id=%s", state.get("paper_id"))
            critique_payload = {}
        critique["used_llm"] = True
        critique["issues"] = [str(item) for item in critique_payload.get("issues", critique["issues"]) if str(item).strip()]
        critique["revision_focus"] = [
            str(item) for item in critique_payload.get("revision_focus", critique["revision_focus"]) if str(item).strip()
        ]
        critique["should_revise"] = bool(
            critique_payload.get("should_revise", critique["should_revise"])
        ) and int(state.get("revision_count", 0)) < int(state.get("revision_budget", 0))
        critique["severity"] = str(critique_payload.get("severity") or critique["severity"])

    logger.info(
        "qa_critic_complete paper_id=%s issues=%s should_revise=%s used_llm=%s",
        state.get("paper_id"),
        len(critique["issues"]),
        critique["should_revise"],
        critique["used_llm"],
    )
    return {
        **state,
        "critic_report": critique,
        "execution_trace": _append_trace(state, f"critique:{critique['severity']}"),
    }


def route_after_critique(state: dict) -> str:
    critique = state.get("critic_report", {})
    return "revise_answer" if critique.get("should_revise") else "evaluate_answer"


def revise_answer_step(state: dict) -> dict:
    critique = state.get("critic_report", {})
    prompt = (
        "Revise the research answer to improve grounding and completeness.\n"
        "Return JSON only with keys: answer, claims.\n"
        "claims must be a list of objects with keys: claim_text, supporting_chunk_ids.\n"
        "Only cite chunk ids already present in the evidence.\n\n"
        f"Question: {state['query']}\n"
        f"Current answer: {state.get('answer', '')}\n"
        f"Current claims: {state.get('grounded_claims', [])}\n"
        f"Revision focus: {critique.get('revision_focus', [])}\n"
        f"Evidence: {state.get('sources', [])}"
    )
    try:
        revision_payload = generate_json(prompt)
    except Exception:
        logger.exception("qa_revision_failed paper_id=%s", state.get("paper_id"))
        revision_payload = {}

    chunk_lookup = _build_chunk_lookup(state.get("filtered_chunks", []))
    diagnostics = state.get("evidence_diagnostics", {})
    raw_claims = revision_payload.get("claims") if isinstance(revision_payload.get("claims"), list) else []
    revised_claims: list[GroundedClaim] = []
    for index, raw_claim in enumerate(raw_claims[:8], start=1):
        if not isinstance(raw_claim, dict):
            continue
        claim_text = str(raw_claim.get("claim_text") or "").strip()
        support_ids = [str(item) for item in (raw_claim.get("supporting_chunk_ids") or []) if str(item).strip()]
        if not claim_text:
            continue
        revised_claims.append(
            _derive_claim_grounding(claim_text, "inferred_from_evidence", support_ids, chunk_lookup, diagnostics, index)
        )

    answer = str(revision_payload.get("answer") or state.get("answer") or "").strip()
    final_confidence = round(_average(claim["confidence"] for claim in revised_claims), 4) if revised_claims else float(state.get("final_confidence", 0.0))
    revision_count = int(state.get("revision_count", 0)) + 1
    logger.info(
        "qa_revision_complete paper_id=%s revision_count=%s claims=%s",
        state.get("paper_id"),
        revision_count,
        len(revised_claims),
    )
    return {
        **state,
        "answer": answer or state.get("answer", ""),
        "grounded_claims": revised_claims or state.get("grounded_claims", []),
        "final_confidence": final_confidence,
        "revision_count": revision_count,
        "execution_trace": _append_trace(state, f"revise:{revision_count}"),
    }


def _label_score(value: float) -> float:
    return round(_clip(value), 4)


def evaluate_answer_step(state: dict) -> dict:
    diagnostics = state.get("evidence_diagnostics", {})
    verifier = state.get("verifier_report", {})
    critique = state.get("critic_report", {})
    plan = state.get("execution_plan", {})
    grounded_claims = list(state.get("grounded_claims", []))
    required_count = max(1, len(plan.get("required_evidence", [])) if isinstance(plan, dict) else 1)
    completeness = _clip(
        1.0 - _safe_div(len(diagnostics.get("missing_required_fields", [])), required_count)
    )
    supported_claim_ratio = verifier.get("supported_claim_ratio")
    if supported_claim_ratio is None:
        if grounded_claims:
            supported_claim_ratio = _safe_div(
                sum(1 for claim in grounded_claims if claim.get("support_status") == "supported"),
                len(grounded_claims),
            )
        else:
            supported_claim_ratio = 0.0
    grounding_quality = _clip(
        0.6 * float(supported_claim_ratio)
        + 0.4 * float(state.get("final_confidence", 0.0))
    )
    evidence_coverage = float(diagnostics.get("section_coverage", 0.0))
    reasoning_depth = _clip(
        0.35
        + 0.15 * len((plan.get("subtasks", []) if isinstance(plan, dict) else []))
        + 0.25 * float(state.get("orchestration_level", 1)) / 3.0
        + 0.25 * completeness
    )
    contradiction_handling = _clip(
        1.0 - float(diagnostics.get("contradiction_ratio", 0.0))
        if "address_contradictions" in critique.get("revision_focus", [])
        else 0.8 - 0.5 * float(diagnostics.get("contradiction_ratio", 0.0))
    )
    scientific_rigor = _clip(
        0.4 * grounding_quality + 0.35 * evidence_coverage + 0.25 * contradiction_handling
    )
    critique_usefulness = _clip(
        0.0
        if not critique
        else 0.7 if critique.get("issues") and not critique.get("should_revise") else 1.0 if state.get("revision_count", 0) else 0.5
    )
    summary: list[str] = []
    if diagnostics.get("missing_required_fields"):
        summary.append(f"Missing evidence: {', '.join(diagnostics['missing_required_fields'])}.")
    if verifier.get("issues"):
        summary.append(f"Verifier flagged {len(verifier['issues'])} issue(s).")
    elif grounded_claims:
        summary.append(
            f"Claim support ratio from grounded claims: {round(float(supported_claim_ratio) * 100)}%."
        )
    if not summary:
        summary.append("Grounding and evidence coverage are acceptable for the selected orchestration level.")

    overall_status = "strong"
    if grounding_quality < 0.45 or evidence_coverage < 0.45:
        overall_status = "insufficient"
    elif grounding_quality < 0.6 or evidence_coverage < 0.6 or completeness < 0.6:
        overall_status = "needs_review"
    elif grounding_quality < 0.75 or scientific_rigor < 0.7:
        overall_status = "sufficient"

    report: EvaluationReport = {
        "grounding_quality": _label_score(grounding_quality),
        "evidence_coverage": _label_score(evidence_coverage),
        "reasoning_depth": _label_score(reasoning_depth),
        "scientific_rigor": _label_score(scientific_rigor),
        "critique_usefulness": _label_score(critique_usefulness),
        "contradiction_handling": _label_score(contradiction_handling),
        "completeness": _label_score(completeness),
        "overall_status": overall_status,
        "summary": summary,
    }
    logger.info(
        "qa_evaluation_complete paper_id=%s overall_status=%s grounding=%s completeness=%s",
        state.get("paper_id"),
        overall_status,
        report["grounding_quality"],
        report["completeness"],
    )
    return {
        **state,
        "evaluation_report": report,
        "execution_trace": _append_trace(state, f"evaluate:{overall_status}"),
    }
