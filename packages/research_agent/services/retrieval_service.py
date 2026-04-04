import logging
import os
import re
import uuid
from collections import defaultdict

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from research_agent.tools.embedder import generate_embedding
from research_agent.tools.vector_store import (
    semantic_search,
    semantic_search_sections,
    semantic_search_subsections,
)
from research_agent.services.chunk_structure import (
    INTENT_ROLE_WEIGHTS,
    build_chunk_structure,
    is_quality_chunk,
    normalize_section_name,
)


load_dotenv()

logger = logging.getLogger(__name__)
DEFAULT_SECTION_TOP_K = int(os.getenv("RETRIEVAL_SECTION_TOP_K", "3"))
DEFAULT_SUBSECTION_TOP_K = int(os.getenv("RETRIEVAL_SUBSECTION_TOP_K", "6"))
DEFAULT_CHUNK_TOP_K = int(os.getenv("RETRIEVAL_CHUNK_TOP_K", "18"))
DEFAULT_FINAL_TOP_K = int(os.getenv("RETRIEVAL_FINAL_TOP_K", "12"))
CHUNKS_PER_SECTION = int(os.getenv("RETRIEVAL_CHUNKS_PER_SECTION", "3"))
REFERENCE_QUERY_KEYWORDS = [
    "reference",
    "references",
    "citation",
    "cite",
    "bibliography",
    "prior work",
]
REFERENCE_SECTION_NAMES = {
    "reference",
    "references",
    "bibliography",
    "works cited",
    "cited works",
}
QUERY_INTENT_KEYWORDS = {
    "definition": ["what is", "define", "definition", "overview", "motivation", "problem"],
    "method": ["method", "approach", "algorithm", "pipeline", "implementation", "how"],
    "evaluation": ["evaluate", "results", "benchmark", "metric", "performance", "ablation"],
    "theory": ["theorem", "proof", "lemma", "bound", "complexity", "formal"],
    "comparison": ["compare", "difference", "baseline", "versus", "vs", "better than"],
}
INTENT_ROLE_MATCHES = {
    "definition": {"problem", "idea"},
    "method": {"method", "algorithm"},
    "evaluation": {"evaluation"},
    "theory": {"theory"},
    "comparison": {"evaluation"},
}
INTENT_ROLE_NEUTRALS = {
    "definition": {"discussion"},
    "method": {"implementation", "idea"},
    "evaluation": {"discussion", "method"},
    "theory": {"algorithm", "idea"},
    "comparison": {"discussion", "method"},
}
SECTION_PRIORS = {
    "definition": {"abstract": 1.0, "introduction": 0.9, "discussion": 0.5},
    "method": {"method": 1.0, "methodology": 1.0, "algorithm": 0.95, "implementation": 0.8},
    "evaluation": {"results": 1.0, "evaluation": 1.0, "experiment": 0.9, "discussion": 0.6},
    "theory": {"theory": 1.0, "proof": 1.0, "method": 0.6},
    "comparison": {"results": 1.0, "evaluation": 0.95, "discussion": 0.8},
}
SECTION_WEIGHTS = {
    "method": 1.35,
    "experiments": 1.3,
    "results": 1.2,
    "abstract": 1.1,
    "introduction": 1.0,
    "front_matter": 0.95,
    "discussion": 0.95,
    "conclusion": 0.9,
    "related_work": 0.7,
    "references": 0.05,
}
CITATION_PATTERNS = [
    r"\[[0-9,\s]+\]",
    r"et al\.",
    r"arxiv preprint",
    r"proceedings of",
    r"ieee",
    r"journal of",
]


def _normalize_section_name(section_name: str | None) -> str | None:
    return normalize_section_name(section_name)

def _score_chunk(content: str) -> int:
    words = len(re.findall(r"\b\w+\b", content))
    technical = len(re.findall(r"\b[a-zA-Z]{7,}\b", content))
    equations = len(re.findall(r"[=<>+\-*/^]|O\(", content))
    return int(min(100, words * 0.05 + technical * 0.25 + equations * 0.4))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _citation_density(text: str) -> float:
    lowered = text.lower()
    matches = sum(len(re.findall(pattern, lowered)) for pattern in CITATION_PATTERNS)
    return matches / max(1, _word_count(text))


def _table_density(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    noisy_lines = 0
    for line in lines:
        digit_ratio = sum(char.isdigit() for char in line) / max(1, len(line))
        upper_ratio = sum(char.isupper() for char in line if char.isalpha()) / max(
            1, sum(char.isalpha() for char in line)
        )
        if "|" in line or "\t" in line or digit_ratio > 0.35 or (upper_ratio > 0.85 and len(line) < 40):
            noisy_lines += 1
    return noisy_lines / max(1, len(lines))


def _heading_fragment_penalty(text: str) -> float:
    stripped = " ".join(text.split())
    if not stripped:
        return 1.0
    if len(stripped) < 20:
        return 1.0
    if _word_count(stripped) <= 4:
        return 0.8
    return 0.0


def _content_quality_score(text: str) -> float:
    words = _word_count(text)
    if words == 0:
        return -2.0
    lowered = text.lower()
    alpha_chars = sum(char.isalpha() for char in text)
    digit_chars = sum(char.isdigit() for char in text)
    alpha_ratio = alpha_chars / max(1, len(text))
    digit_ratio = digit_chars / max(1, len(text))
    citation_density = _citation_density(text)
    table_density = _table_density(text)
    heading_penalty = _heading_fragment_penalty(text)
    newline_density = text.count("\n") / max(1, words)

    score = 0.0
    if words >= 80:
        score += 1.0
    elif words >= 40:
        score += 0.6
    elif words >= 20:
        score += 0.2
    else:
        score -= 0.4

    if alpha_ratio < 0.45:
        score -= 0.6
    if citation_density > 0.03:
        score -= 1.35
    elif citation_density > 0.015:
        score -= 0.7
    if table_density > 0.6:
        score -= 1.0
    elif table_density > 0.35:
        score -= 0.7
    if digit_ratio > 0.18 and newline_density > 0.12:
        score -= 0.8
    if heading_penalty:
        score -= heading_penalty
    if lowered.startswith("table ") or lowered.startswith("figure ") or lowered.startswith("equation:"):
        score -= 0.8
    if "we propose" in lowered or "our proposed" in lowered or "in this work" in lowered:
        score += 0.3
    return score


def _intent_role_weight(intent: str, role: str) -> float:
    return INTENT_ROLE_WEIGHTS.get(intent, {}).get(role, 0.4)


def _role_match_weight(intent: str, role: str) -> float:
    if role in INTENT_ROLE_MATCHES.get(intent, set()):
        return 1.0
    if role in INTENT_ROLE_NEUTRALS.get(intent, set()):
        return 0.3
    return 0.0


def _annotate_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    scores = [float(candidate["score"]) for candidate in candidates]
    semantic_floor = min(scores)
    semantic_span = max(scores) - semantic_floor
    annotated = []
    for candidate in candidates:
        role_info = build_chunk_structure(
            chunk_id=str(candidate.get("chunk_id") or candidate.get("section_id") or candidate.get("subsection_id") or ""),
            chunk_index=int(candidate.get("chunk_index") or candidate.get("section_order") or 0),
            section_name=candidate.get("section_name"),
            subsection_name=candidate.get("subsection_name"),
            content=str(candidate.get("content") or ""),
            total_chunks=max(1, len(candidates)),
        )
        quality_score = _content_quality_score(candidate.get("content", ""))
        semantic_norm = (float(candidate["score"]) - semantic_floor) / semantic_span if semantic_span > 0 else 1.0
        annotated.append(
            {
                **candidate,
                "role": role_info["role"],
                "importance": role_info["importance"],
                "role_confidence": role_info["confidence"],
                "summary": role_info["summary"],
                "quality_score": round(quality_score, 4),
                "semantic_rank_score": round(semantic_norm, 4),
                "quality_pass": is_quality_chunk(role_info),
            }
        )
    return annotated


def _filter_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    annotated = _annotate_candidates(candidates)
    quality_kept = [candidate for candidate in annotated if bool(candidate.get("quality_pass"))]
    if quality_kept:
        annotated = quality_kept
    kept: list[dict] = []
    best_score = max(float(candidate["score"]) for candidate in annotated)
    for candidate in annotated:
        quality_score = float(candidate["quality_score"])
        semantic_score = float(candidate["score"])
        if quality_score <= -1.0 and semantic_score < best_score - 0.01:
            continue
        if quality_score <= -0.5 and semantic_score < best_score - 0.01:
            continue
        if quality_score <= -0.2 and semantic_score < best_score - 0.05:
            continue
        kept.append(candidate)
    return kept or annotated


def _chunk_rerank_score(candidate: dict, query_intent: str) -> float:
    cosine_similarity = float(candidate["score"])
    role_match_weight = _role_match_weight(query_intent, str(candidate.get("role") or "other"))
    importance = float(candidate.get("importance", 0.0))
    # NOTE: score ranges will remain low for papers ingested before the Fix 1
    # quality gate was deployed. Re-ingest affected papers using reanalyze_all().
    return round(
        0.5 * cosine_similarity
        + 0.3 * role_match_weight
        + 0.2 * importance,
        4,
    )


def _apply_rerank_scores(candidates: list[dict], query_intent: str) -> list[dict]:
    scored: list[dict] = []
    for candidate in candidates:
        scored.append(
            {
                **candidate,
                "intent_role_weight": round(
                    _intent_role_weight(query_intent, str(candidate.get("role") or "other")),
                    4,
                ),
                "role_match_weight": round(
                    _role_match_weight(query_intent, str(candidate.get("role") or "other")),
                    4,
                ),
                "rerank_score": _chunk_rerank_score(candidate, query_intent),
            }
        )
    return scored


def _section_weight(section_name: str | None) -> float:
    if not section_name:
        return 1.0
    return SECTION_WEIGHTS.get(_normalize_section_name(section_name), 1.0)


def is_reference_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in REFERENCE_QUERY_KEYWORDS)


def classify_query(query: str) -> str:
    lowered = query.lower()
    best_type = "method"
    best_score = 0
    for query_type, keywords in QUERY_INTENT_KEYWORDS.items():
        score = sum(lowered.count(keyword) for keyword in keywords)
        if score > best_score:
            best_score = score
            best_type = query_type
    return best_type


def _section_prior(query_type: str, section_name: str | None) -> float:
    if not section_name:
        return 0.0
    return SECTION_PRIORS.get(query_type, {}).get(_normalize_section_name(section_name), 0.1)


def semantic_retrieve_sections(
    db: Session,
    query_embedding: list[float],
    paper_id: uuid.UUID,
    query_type: str = "method",
    include_references: bool = False,
    section_top_k: int = DEFAULT_SECTION_TOP_K,
) -> list[dict]:
    semantic_sections = semantic_search_sections(
        db,
        query_embedding,
        paper_id=paper_id,
        top_k=max(section_top_k * 4, DEFAULT_SECTION_TOP_K),
    )
    filtered_sections = _filter_candidates(semantic_sections)
    weighted_sections = sorted(
        [
            section
            for section in filtered_sections
            if include_references
            or _normalize_section_name(section["section_name"]) != "references"
        ],
        key=lambda section: (
            0.7 * section["score"] + 0.3 * _section_prior(query_type, section["section_name"]),
            section.get("quality_score", 0.0),
            _section_weight(section["section_name"]),
            -section["section_order"],
        ),
        reverse=True,
    )
    selected_sections: list[dict] = []
    seen_section_names: set[str] = set()
    for section in weighted_sections:
        normalized_name = _normalize_section_name(section["section_name"]) or str(section["section_name"])
        if normalized_name in seen_section_names:
            continue
        selected_sections.append(section)
        seen_section_names.add(normalized_name)
        if len(selected_sections) >= section_top_k:
            break
    return selected_sections


def semantic_retrieve_subsections(
    db: Session,
    query_embedding: list[float],
    paper_id: uuid.UUID,
    section_names: list[str],
    subsection_top_k: int = DEFAULT_SUBSECTION_TOP_K,
) -> list[dict]:
    subsection_results = semantic_search_subsections(
        db,
        query_embedding,
        paper_id=paper_id,
        section_names=section_names,
        top_k=max(subsection_top_k * 4, DEFAULT_SUBSECTION_TOP_K),
    )
    subsection_results = _filter_candidates(subsection_results)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for subsection in subsection_results:
        grouped[subsection.get("section_name") or "unknown"].append(subsection)

    selected: list[dict] = []
    per_section_limit = max(1, subsection_top_k // max(1, len(section_names or ["_"])))
    ordered_sections = section_names or list(grouped.keys())
    for section_name in ordered_sections:
        ranked = sorted(
            grouped.get(section_name, []),
            key=lambda subsection: (
                subsection["score"] * _section_weight(subsection["section_name"]),
                subsection.get("quality_score", 0.0),
                -(subsection["page_number"] or 10**6),
            ),
            reverse=True,
        )
        selected.extend(ranked[:per_section_limit])

    if len(selected) < subsection_top_k:
        seen_ids = {subsection["subsection_id"] for subsection in selected}
        remainder = sorted(
            subsection_results,
            key=lambda subsection: (
                subsection["score"] * _section_weight(subsection["section_name"]),
                subsection.get("quality_score", 0.0),
                -(subsection["page_number"] or 10**6),
            ),
            reverse=True,
        )
        selected.extend([subsection for subsection in remainder if subsection["subsection_id"] not in seen_ids])

    return selected[:subsection_top_k]


def semantic_retrieve_chunks(
    db: Session,
    query_embedding: list[float],
    paper_id: uuid.UUID | None = None,
    section_names: list[str] | None = None,
    subsection_names: list[str | None] | None = None,
    query_intent: str = "method",
    semantic_top_k: int = DEFAULT_CHUNK_TOP_K,
) -> list[dict]:
    chunk_results = semantic_search(
        db,
        query_embedding,
        paper_id=paper_id,
        section_names=section_names,
        subsection_names=subsection_names,
        top_k=max(semantic_top_k * 4, 40),
    )
    filtered_chunks = _filter_candidates(chunk_results)
    scored_chunks = _apply_rerank_scores(filtered_chunks, query_intent)
    ranked_chunks = sorted(
        scored_chunks,
        key=lambda chunk: (
            float(chunk.get("rerank_score", chunk["score"])),
            float(chunk.get("score", 0.0)),
            float(chunk.get("importance", 0.0)),
            -(chunk.get("page_number") or 10**6),
        ),
        reverse=True,
    )
    return ranked_chunks[: max(semantic_top_k * 2, semantic_top_k)]


def rerank_chunks(
    chunks: list[dict],
    top_k: int = DEFAULT_FINAL_TOP_K,
    query_intent: str = "method",
) -> list[dict]:
    annotated = _filter_candidates(chunks)
    if not annotated:
        return []
    scored = _apply_rerank_scores(annotated, query_intent)
    ranked = sorted(
        scored,
        key=lambda chunk: (
            float(chunk.get("rerank_score", chunk["score"])),
            float(chunk.get("score", 0.0)),
            float(chunk.get("importance", 0.0)),
            -int(chunk["chunk_index"]),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def balanced_select_chunks(
    chunks: list[dict],
    selected_sections: list[dict],
    query_intent: str = "method",
    top_k: int = DEFAULT_FINAL_TOP_K,
    chunks_per_section: int = CHUNKS_PER_SECTION,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.get("section_name") or "unknown"].append(chunk)

    for section_name in grouped:
        grouped[section_name] = rerank_chunks(
            grouped[section_name],
            top_k=chunks_per_section,
            query_intent=query_intent,
        )

    merged: list[dict] = []
    for section in selected_sections:
        section_candidates = grouped.get(section["section_name"], [])
        if not section_candidates:
            continue
        best_section_score = max(float(candidate.get("rerank_score", candidate["score"])) for candidate in section_candidates)
        merged.extend(
            [
                candidate
                for candidate in section_candidates
                if float(candidate.get("rerank_score", candidate["score"])) >= best_section_score - 0.12
                and float(candidate.get("quality_score", 0.0)) > -0.2
            ]
        )

    if len(merged) < top_k:
        seen_ids = {chunk["chunk_id"] for chunk in merged}
        remainder = [
            chunk
            for chunk in rerank_chunks(chunks, top_k=len(chunks), query_intent=query_intent)
            if chunk["chunk_id"] not in seen_ids
            and float(chunk.get("quality_score", 0.0)) > -0.1
            and float(chunk.get("importance", 0.0)) > 0.1
        ]
        merged.extend(remainder)

    return rerank_chunks(merged, top_k=top_k, query_intent=query_intent)


def retrieve_relevant_chunks(
    db: Session,
    query: str,
    top_k: int = DEFAULT_FINAL_TOP_K,
    paper_id: uuid.UUID | None = None,
    section_top_k: int = DEFAULT_SECTION_TOP_K,
    subsection_top_k: int = DEFAULT_SUBSECTION_TOP_K,
    semantic_top_k: int = DEFAULT_CHUNK_TOP_K,
) -> dict[str, object]:
    query_type = classify_query(query)
    reference_query = is_reference_query(query)
    query_embedding = generate_embedding(query)

    selected_sections = (
        semantic_retrieve_sections(
            db,
            query_embedding,
            paper_id,
            query_type=query_type,
            include_references=reference_query,
            section_top_k=section_top_k,
        )
        if paper_id is not None
        else []
    )
    section_names = [section["section_name"] for section in selected_sections]

    selected_subsections = (
        semantic_retrieve_subsections(
            db,
            query_embedding,
            paper_id,
            section_names=section_names,
            subsection_top_k=subsection_top_k,
        )
        if paper_id is not None and section_names
        else []
    )
    subsection_names = [
        subsection["subsection_name"] for subsection in selected_subsections
    ] or None

    semantic_results = semantic_retrieve_chunks(
        db,
        query_embedding,
        paper_id=paper_id,
        section_names=section_names or None,
        subsection_names=subsection_names,
        query_intent=query_type,
        semantic_top_k=semantic_top_k,
    )
    selected_chunks = balanced_select_chunks(
        semantic_results,
        selected_sections=selected_sections,
        query_intent=query_type,
        top_k=top_k,
        chunks_per_section=CHUNKS_PER_SECTION,
    )
    retrieval_confidence = (
        sum(float(chunk["score"]) for chunk in selected_chunks) / len(selected_chunks)
        if selected_chunks
        else 0.0
    )

    logger.info(
        "retrieval_complete query=%s query_type=%s paper_id=%s query_embedding_dimension=%s reference_query=%s "
        "selected_sections=%s selected_subsections=%s semantic_chunks_retrieved=%s "
        "post_filter_chunks_selected=%s retrieval_confidence=%s top_similarity_scores=%s",
        query,
        query_type,
        paper_id,
        len(query_embedding),
        reference_query,
        section_names,
        subsection_names,
        len(semantic_results),
        len(selected_chunks),
        round(retrieval_confidence, 4),
        [round(float(result["score"]), 4) for result in semantic_results[:5]],
    )

    return {
        "query_type": query_type,
        "query_embedding": query_embedding,
        "retrieved_sections": selected_sections,
        "retrieved_subsections": selected_subsections,
        "retrieved_chunks": semantic_results,
        "filtered_chunks": selected_chunks,
        "retrieval_confidence": retrieval_confidence,
    }
