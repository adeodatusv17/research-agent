import re
from collections.abc import Mapping
from typing import TypedDict

from research_agent.tools.embedder import normalize_text_for_embedding


CHUNK_ROLES = [
    "problem",
    "idea",
    "method",
    "algorithm",
    "equation",
    "formula",
    "theory",
    "evaluation",
    "implementation",
    "discussion",
    "other",
]

INTENT_ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "definition": {"problem": 1.0, "idea": 0.9, "discussion": 0.6, "other": 0.4},
    "method": {"method": 1.0, "algorithm": 0.95, "equation": 0.85, "formula": 0.85, "implementation": 0.75, "idea": 0.5},
    "evaluation": {"evaluation": 1.0, "discussion": 0.7, "method": 0.5},
    "theory": {"theory": 1.0, "equation": 0.8, "formula": 0.8, "algorithm": 0.65, "idea": 0.5},
    "comparison": {"evaluation": 1.0, "discussion": 0.8, "method": 0.45},
}

SECTION_ROLE_HINTS: dict[str, str] = {
    "abstract": "idea",
    "introduction": "problem",
    "background": "problem",
    "related_work": "other",
    "method": "method",
    "methodology": "method",
    "approach": "method",
    "algorithm": "algorithm",
    "theory": "theory",
    "proof": "theory",
    "implementation": "implementation",
    "system": "implementation",
    "experiment": "evaluation",
    "results": "evaluation",
    "evaluation": "evaluation",
    "discussion": "discussion",
    "conclusion": "discussion",
}

SECTION_ROLE_PRIORS: dict[str, tuple[str, ...]] = {
    "abstract": ("idea", "problem"),
    "introduction": ("problem", "idea"),
    "method": ("method", "algorithm"),
    "methodology": ("method", "algorithm"),
    "approach": ("method", "algorithm"),
    "results": ("evaluation",),
    "evaluation": ("evaluation",),
    "experiment": ("evaluation",),
    "conclusion": ("discussion",),
}

ROLE_KEYWORDS: dict[str, list[str]] = {
    "problem": ["problem", "challenge", "limitation", "goal", "objective", "motivation"],
    "idea": ["we propose", "key idea", "insight", "novel", "framework", "approach"],
    "method": ["method", "pipeline", "procedure", "stage", "module", "component"],
    "algorithm": ["algorithm", "step", "pseudocode", "complexity", "runtime", "procedure"],
    "equation": ["equation", "=", "[equation]"],
    "formula": ["formula", "objective", "loss is defined", "mathematically"],
    "theory": ["theorem", "lemma", "proof", "corollary", "bound", "proposition"],
    "evaluation": ["experiment", "results", "benchmark", "comparison", "ablation", "accuracy", "latency"],
    "implementation": ["implementation", "system", "deployment", "engineering", "prototype", "infrastructure"],
    "discussion": ["discussion", "limitation", "future work", "threats to validity", "conclusion"],
}

TECHNICAL_TOKEN_RE = re.compile(r"\b[a-zA-Z]{7,}\b")
WORD_RE = re.compile(r"\b\w+\b")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")
UPPER_OR_DIGIT_RE = re.compile(r"^[A-Z0-9\[]")
ONLY_SYMBOLS_RE = re.compile(r"^[\d\s+\-*/^=<>()[\]{}|.,:;%]+$")
EQUATION_MARKER_RE = re.compile(r"\bEQUATION:\s*", re.IGNORECASE)
BANNED_SUMMARY_STARTS = (
    "figure",
    "fig.",
    "table",
    "eq.",
    "equation:",
    "index terms:",
)
BANNED_SUMMARY_PATTERNS = (
    re.compile(r"\bms rate\b", re.IGNORECASE),
    re.compile(r"\bx\s*n\s*\+", re.IGNORECASE),
    re.compile(r"\b1/2\s*x\b", re.IGNORECASE),
    re.compile(r"\bsubsampling\b", re.IGNORECASE),
)
VERB_LIKE_RE = re.compile(
    r"\b("
    r"is|are|was|were|be|been|being|am|has|have|had|do|does|did|"
    r"can|could|may|might|will|would|should|"
    r"propose|proposes|proposed|introduce|introduces|introduced|"
    r"show|shows|showed|demonstrate|demonstrates|demonstrated|"
    r"contain|contains|contained|consist|consists|consisted|"
    r"include|includes|included|improve|improves|improved|"
    r"achieve|achieves|achieved|evaluate|evaluates|evaluated|"
    r"use|uses|used|perform|performs|performed|"
    r"yield|yields|yielded|reduce|reduces|reduced|"
    r"increase|increases|increased|outperform|outperforms|outperformed|"
    r"appear|appears|appeared|allow|allows|allowed|"
    r"provide|provides|provided"
    r")\b",
    re.IGNORECASE,
)
SUMMARY_MAX_CHARS = 300


class ChunkStructure(TypedDict):
    chunk_id: str | None
    chunk_index: int
    section_name: str | None
    subsection_name: str | None
    role: str
    importance: float
    confidence: float
    content_excerpt: str
    summary: str


class PreparedIndexChunk(ChunkStructure, total=False):
    content: str
    token_count: int


def _normalize_equation_latex(value: str) -> str:
    collapsed = " ".join(value.split())
    collapsed = collapsed.replace("[equation]", "").strip(" ,;:")
    return collapsed


def _extract_equation_items(chunk: Mapping[str, object]) -> list[dict]:
    text = str(chunk.get("content") or chunk.get("text") or "").strip()
    if not text:
        return []

    patterns = [
        r"[A-Za-z0-9˜′'`_]+(?:\s*[A-Za-z0-9˜′'`_()]*)?\s*=\s*[^.;\n]{6,180}",
        r"O\([^)]{1,80}\)",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))

    deduped: list[dict] = []
    seen: set[str] = set()
    description = str(chunk.get("summary") or text).strip()
    for match in matches:
        latex = _normalize_equation_latex(match)
        if not latex or latex.lower() in seen:
            continue
        seen.add(latex.lower())
        deduped.append(
            {
                "id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "latex": latex,
                "description": description[:220],
                "text": text,
            }
        )
    return deduped


def normalize_section_name(section_name: str | None) -> str | None:
    if not section_name:
        return None
    normalized = section_name.strip().lower().replace("_", " ")
    normalized = " ".join(normalized.split())
    if normalized in {"reference", "references", "bibliography", "works cited", "cited works"}:
        return "references"
    if normalized in {"related work", "related works", "prior work"}:
        return "related_work"
    return normalized.replace(" ", "_")


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _technical_density(text: str) -> float:
    words = _word_count(text)
    if words == 0:
        return 0.0
    technical_tokens = len(TECHNICAL_TOKEN_RE.findall(text))
    symbol_count = len(re.findall(r"[=<>+\-*/^]|O\(|\\[a-zA-Z]+", text))
    return min(1.0, (technical_tokens + symbol_count * 2) / max(1, words))


def _has_equation_signal(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(theorem|lemma|proof|corollary|equation)\b", lowered)) or (
        text.count("=") >= 2 or "O(" in text
    )


def _has_pseudocode_signal(text: str) -> bool:
    lowered = text.lower()
    patterns = ["algorithm", "procedure", "for each", "while ", "if ", "return "]
    return sum(1 for pattern in patterns if pattern in lowered) >= 2


def _has_result_signal(text: str) -> bool:
    lowered = text.lower()
    patterns = ["benchmark", "experiment", "results", "table", "figure", "comparison", "improve", "%"]
    return sum(1 for pattern in patterns if pattern in lowered) >= 2


def _normalize_chunk_whitespace(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def _starts_with_banned_prefix(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in BANNED_SUMMARY_STARTS)


def _contains_banned_fragment(text: str) -> bool:
    return any(pattern.search(text) for pattern in BANNED_SUMMARY_PATTERNS)


def _is_equation_or_diagram_heavy(text: str) -> bool:
    lowered = text.lower()
    if ONLY_SYMBOLS_RE.fullmatch(text):
        return True
    if _contains_banned_fragment(text):
        return True
    equation_markers = len(re.findall(r"(\[equation\]|=|O\(|\\[a-zA-Z]+|\bfrac\b)", lowered))
    bracket_chars = sum(text.count(char) for char in "()[]{}")
    alpha_chars = sum(char.isalpha() for char in text)
    if equation_markers >= 3 and alpha_chars < 50:
        return True
    if bracket_chars >= 12 and alpha_chars < 80:
        return True
    return False


def _advance_to_sentence_boundary(text: str) -> str | None:
    if UPPER_OR_DIGIT_RE.match(text):
        return text
    match = SENTENCE_BOUNDARY_RE.search(text)
    if not match:
        return None
    advanced = text[match.end() :].lstrip()
    return advanced if advanced and UPPER_OR_DIGIT_RE.match(advanced) else None


def sanitize_chunk_content(content: str) -> str | None:
    cleaned = _normalize_chunk_whitespace(content)
    if not cleaned:
        return None
    if _starts_with_banned_prefix(cleaned):
        return None

    cleaned = EQUATION_MARKER_RE.sub("[equation] ", cleaned)
    cleaned = _normalize_chunk_whitespace(cleaned)
    if _is_equation_or_diagram_heavy(cleaned):
        return None

    advanced = _advance_to_sentence_boundary(cleaned)
    if not advanced:
        return None
    cleaned = advanced.strip()
    if _starts_with_banned_prefix(cleaned) or _is_equation_or_diagram_heavy(cleaned):
        return None
    return cleaned


def build_chunk_summary(content: str, *, max_chars: int = SUMMARY_MAX_CHARS) -> str | None:
    cleaned = sanitize_chunk_content(content)
    if not cleaned:
        return None

    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(cleaned) if sentence.strip()]
    if not sentences:
        return None

    summary_parts: list[str] = []
    truncated = False
    for sentence in sentences:
        candidate = " ".join(summary_parts + [sentence]).strip()
        if len(candidate) > max_chars:
            truncated = True
            break
        summary_parts.append(sentence)
        if len(candidate) >= 140:
            break

    if not summary_parts:
        first_sentence = sentences[0]
        if len(first_sentence) > max_chars - 3:
            trimmed = first_sentence[: max_chars - 3].rsplit(" ", 1)[0].strip()
            return f"{trimmed}..." if trimmed else None
        return first_sentence

    summary = " ".join(summary_parts).strip()
    if len(summary) < 40:
        for sentence in sentences[len(summary_parts) :]:
            candidate = f"{summary} {sentence}".strip()
            if len(candidate) > max_chars:
                truncated = True
                break
            summary = candidate
            if len(summary) >= 40:
                break

    if len(summary_parts) < len(sentences):
        truncated = True

    if truncated:
        if len(summary) > max_chars - 3:
            summary = summary[: max_chars - 3].rsplit(" ", 1)[0].strip()
        summary = f"{summary}..." if summary and not summary.endswith("...") else summary

    return summary[:max_chars] if summary else None


def _looks_like_short_noun_phrase(summary: str) -> bool:
    words = re.findall(r"\b[A-Za-z][A-Za-z\-]*\b", summary)
    if len(words) <= 1:
        return True
    if len(words) > 5:
        return False
    if VERB_LIKE_RE.search(summary):
        return False
    return True


def is_quality_chunk(chunk: dict) -> bool:
    confidence = float(chunk.get("confidence") or 0.0)
    role = str(chunk.get("role") or "other")
    summary = str(
        chunk.get("summary")
        or chunk.get("content_excerpt")
        or chunk.get("content")
        or ""
    ).strip()

    if confidence < 0.35:
        return False
    if role == "other":
        return False
    if len(summary) < 40:
        return False
    if not UPPER_OR_DIGIT_RE.match(summary):
        return False
    if _starts_with_banned_prefix(summary):
        return False
    if ONLY_SYMBOLS_RE.fullmatch(summary):
        return False
    if _contains_banned_fragment(summary):
        return False

    alpha_only = re.sub(r"[\d\s\W_]+", "", summary)
    if len(alpha_only) < 20:
        return False
    if _looks_like_short_noun_phrase(summary):
        return False
    return True


def infer_chunk_role(content: str, section_name: str | None, subsection_name: str | None = None) -> tuple[str, float]:
    lowered = content.lower()
    normalized_section = normalize_section_name(section_name)
    normalized_subsection = normalize_section_name(subsection_name)
    effective_section = normalized_subsection or normalized_section

    if effective_section == "related_work":
        return "other", 0.25

    role_scores: dict[str, float] = {role: 0.0 for role in CHUNK_ROLES}
    section_hint = SECTION_ROLE_HINTS.get(effective_section or "")
    if section_hint:
        role_scores[section_hint] += 1.0

    for role, keywords in ROLE_KEYWORDS.items():
        role_scores[role] += sum(lowered.count(keyword) for keyword in keywords) * 0.18

    if _has_equation_signal(content):
        role_scores["equation"] += 0.75
        role_scores["formula"] += 0.55
        role_scores["theory"] += 0.35
        role_scores["algorithm"] += 0.12
    if _has_pseudocode_signal(content):
        role_scores["algorithm"] += 0.35
        role_scores["implementation"] += 0.15
    if _has_result_signal(content):
        role_scores["evaluation"] += 0.35
        role_scores["discussion"] += 0.1

    role, score = max(role_scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(role_scores.values(), reverse=True)
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = max(0.0, score - second)
    confidence = max(0.2, min(1.0, 0.35 + margin + min(0.35, score / 4.0)))
    if score <= 0.01:
        return "other", 0.25

    if confidence < 0.6 and effective_section:
        priors = SECTION_ROLE_PRIORS.get(effective_section, ())
        if priors:
            role = max(priors, key=lambda candidate_role: role_scores.get(candidate_role, 0.0))
            confidence = max(confidence, 0.45)

    return role, round(confidence, 4)


def infer_chunk_importance(
    content: str,
    section_name: str | None,
    chunk_index: int,
    total_chunks: int,
) -> float:
    normalized_section = normalize_section_name(section_name) or ""
    position = chunk_index / max(1, total_chunks - 1)
    importance = 0.25

    if normalized_section in {"abstract", "introduction", "conclusion", "discussion"}:
        importance += 0.2
    elif normalized_section in {"method", "methodology", "algorithm", "evaluation", "results", "experiment"}:
        importance += 0.12

    if position <= 0.15:
        importance += 0.15
    elif position >= 0.9:
        importance += 0.12

    technical_density = _technical_density(content)
    importance += min(0.3, technical_density * 0.6)

    if _has_equation_signal(content):
        importance += 0.08
    if _has_pseudocode_signal(content):
        importance += 0.08
    if _has_result_signal(content):
        importance += 0.08

    return round(max(0.0, min(1.0, importance)), 4)


def build_chunk_structure(
    *,
    chunk_id: str | None,
    chunk_index: int,
    section_name: str | None,
    subsection_name: str | None,
    content: str,
    total_chunks: int,
) -> ChunkStructure:
    summary = build_chunk_summary(content) or ""
    role, confidence = infer_chunk_role(content, section_name, subsection_name)
    importance = infer_chunk_importance(content, section_name, chunk_index, total_chunks)
    return {
        "chunk_id": chunk_id,
        "chunk_index": int(chunk_index),
        "section_name": section_name,
        "subsection_name": subsection_name,
        "role": role,
        "importance": importance,
        "confidence": confidence,
        "content_excerpt": summary,
        "summary": summary,
    }


def prepare_chunk_for_indexing(
    chunk: Mapping[str, object],
    *,
    total_chunks: int,
) -> PreparedIndexChunk | None:
    cleaned_content = sanitize_chunk_content(str(chunk.get("content") or ""))
    if not cleaned_content:
        return None

    normalized_content, token_count = normalize_text_for_embedding(cleaned_content)
    if not normalized_content:
        return None

    prepared = build_chunk_structure(
        chunk_id=None,
        chunk_index=int(chunk.get("chunk_index") or 0),
        section_name=str(chunk.get("section_name")) if chunk.get("section_name") else None,
        subsection_name=str(chunk.get("subsection_name")) if chunk.get("subsection_name") else None,
        content=normalized_content,
        total_chunks=total_chunks,
    )
    if not is_quality_chunk(prepared):
        return None

    return {
        **prepared,
        "chunk_index": int(chunk.get("chunk_index") or 0),
        "section_name": str(chunk.get("section_name")) if chunk.get("section_name") else None,
        "subsection_name": str(chunk.get("subsection_name")) if chunk.get("subsection_name") else None,
        "content": normalized_content,
        "token_count": token_count,
    }


def prepare_chunks_for_indexing(chunks: list[Mapping[str, object]]) -> list[PreparedIndexChunk]:
    prepared: list[PreparedIndexChunk] = []
    total_chunks = max(1, len(chunks))
    for chunk in chunks:
        prepared_chunk = prepare_chunk_for_indexing(chunk, total_chunks=total_chunks)
        if prepared_chunk is not None:
            prepared.append(prepared_chunk)
    return prepared


def build_inferred_structure(chunks: list[ChunkStructure], *, max_items: int = 6) -> dict[str, object]:
    grouped: dict[str, object] = {
        "key_ideas": [],
        "methods": {
            "chunks": [],
            "equations": {
                "source": None,
                "items": [],
            },
        },
        "results": [],
        "discussion": [],
    }
    for chunk in sorted(chunks, key=lambda row: (row["importance"], row["confidence"]), reverse=True):
        summary = str(chunk.get("summary") or chunk.get("content_excerpt") or "").strip()
        full_text = str(chunk.get("content") or summary).strip()
        payload = {
            "id": chunk.get("chunk_id"),
            "text": full_text,
            "summary": summary,
            "role": chunk["role"],
            "importance": chunk["importance"],
            "confidence": chunk["confidence"],
            "source": "extracted" if chunk.get("chunk_id") else "inferred",
            "section_name": chunk["section_name"],
            "chunk_index": chunk["chunk_index"],
        }
        if chunk["role"] in {"equation", "formula"}:
            equation_items = _extract_equation_items(chunk)
            if equation_items:
                equations_bucket = grouped["methods"]["equations"]["items"]  # type: ignore[index]
                equations_bucket.extend(equation_items)
        if not is_quality_chunk(payload):
            continue
        if chunk["role"] in {"problem", "idea"}:
            grouped["key_ideas"].append(payload)  # type: ignore[union-attr]
        elif chunk["role"] in {"method", "algorithm", "implementation", "theory"}:
            grouped["methods"]["chunks"].append(payload)  # type: ignore[index]
        elif chunk["role"] == "evaluation":
            grouped["results"].append(payload)  # type: ignore[union-attr]
        else:
            grouped["discussion"].append(payload)  # type: ignore[union-attr]

    for key in ("key_ideas", "results", "discussion"):
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in grouped[key]:  # type: ignore[index]
            norm = item["summary"].lower().strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(item)
            if len(deduped) >= max_items:
                break
        grouped[key] = deduped

    method_deduped: list[dict] = []
    method_seen: set[str] = set()
    for item in grouped["methods"]["chunks"]:  # type: ignore[index]
        norm = item["summary"].lower().strip()
        if not norm or norm in method_seen:
            continue
        method_seen.add(norm)
        method_deduped.append(item)
        if len(method_deduped) >= max_items:
            break
    grouped["methods"]["chunks"] = method_deduped  # type: ignore[index]

    equation_deduped: list[dict] = []
    equation_seen: set[str] = set()
    for item in grouped["methods"]["equations"]["items"]:  # type: ignore[index]
        latex = str(item.get("latex") or "").strip().lower()
        if not latex or latex in equation_seen:
            continue
        equation_seen.add(latex)
        equation_deduped.append(item)
        if len(equation_deduped) >= max_items:
            break
    grouped["methods"]["equations"]["items"] = equation_deduped  # type: ignore[index]
    if equation_deduped:
        grouped["methods"]["equations"]["source"] = "extracted"  # type: ignore[index]
    return grouped
