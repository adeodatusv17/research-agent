import re
from collections.abc import Sequence


METRIC_KEYWORDS = ("wer", "cer", "bleu", "rouge", "f1", "accuracy", "latency", "perplexity", "precision", "recall")
DATASET_KEYWORDS = (
    "mnli",
    "sst-2",
    "mrpc",
    "cola",
    "qnli",
    "qqp",
    "rte",
    "sts-b",
    "librispeech",
    "wikisql",
    "webnlg",
    "dart",
    "samsum",
)
TABLE_LABEL_RE = re.compile(r"\btable\s+(\d+)\b", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.replace("\x00", "").splitlines()).strip()


def _find_caption(blocks: Sequence[dict], block_index: int) -> str | None:
    candidate_indexes = [
        block_index - 4,
        block_index - 3,
        block_index - 2,
        block_index - 1,
        block_index + 1,
        block_index + 2,
        block_index + 3,
        block_index + 4,
    ]
    for index in candidate_indexes:
        if index < 0 or index >= len(blocks):
            continue
        text = str(blocks[index].get("text") or "").strip()
        if TABLE_LABEL_RE.search(text):
            return _normalize_text(text)
    return None


def _looks_like_table_payload(text: str) -> bool:
    stripped = _normalize_text(text)
    if not stripped:
        return False
    if stripped.startswith("TABLE:"):
        return True

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False

    if len(lines) < 6:
        return False

    short_line_count = sum(1 for line in lines if len(line) <= 18)
    digit_count = sum(1 for char in stripped if char.isdigit())

    if short_line_count >= max(4, len(lines) // 2) and digit_count >= 4:
        return True

    first_line = lines[0].lower()
    if TABLE_LABEL_RE.search(first_line) and short_line_count >= max(4, len(lines) // 2):
        return True

    return False


def _infer_table_type(caption: str | None, table_text: str) -> str:
    lowered = f"{caption or ''}\n{table_text}".lower()
    if any(keyword in lowered for keyword in ("learning rate", "batch size", "max seq", "dropout", "optimizer")):
        return "hyperparameter"
    if any(keyword in lowered for keyword in ("ablation", "remove", "without", "effect of")):
        return "ablation"
    if any(keyword in lowered for keyword in ("accuracy", "wer", "cer", "bleu", "rouge", "benchmark", "results")):
        return "results"
    if any(keyword in lowered for keyword in ("layer", "component", "module", "parameter", "architecture")):
        return "architecture"
    return "other"


def _extract_metric_names(text: str) -> list[str]:
    lowered = text.lower()
    return [keyword.upper() if len(keyword) <= 4 else keyword.title() for keyword in METRIC_KEYWORDS if keyword in lowered]


def _extract_dataset_names(text: str) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in DATASET_KEYWORDS if keyword in lowered]


def _extract_model_names(table_text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    for line in lines[1:6]:
        first_cell = line.split("|", 1)[0].strip()
        if not first_cell or len(first_cell) < 2:
            continue
        if re.search(r"\d", first_cell) and len(first_cell.split()) <= 2:
            continue
        if first_cell.lower() in seen:
            continue
        seen.add(first_cell.lower())
        names.append(first_cell)
    return names


def _linked_chunk_indexes(page_number: int | None, section_segments: list[dict[str, object]], *, span: int = 2) -> list[int]:
    if page_number is None:
        return []
    matching = [
        int(segment.get("section_order") or 0)
        for segment in section_segments
        if int(segment.get("page_number") or 0) == page_number
    ]
    if not matching:
        return []
    indexes: set[int] = set()
    for base in matching:
        for offset in range(-span, span + 1):
            indexes.add(max(0, base + offset))
    return sorted(indexes)


def _locate_section(page_number: int | None, section_segments: list[dict[str, object]]) -> tuple[str | None, str | None]:
    if page_number is None:
        return None, None
    same_page = [segment for segment in section_segments if int(segment.get("page_number") or 0) == page_number]
    if same_page:
        segment = same_page[0]
        return (
            str(segment.get("section_name")) if segment.get("section_name") else None,
            str(segment.get("subsection_name")) if segment.get("subsection_name") else None,
        )
    earlier = [
        segment
        for segment in section_segments
        if segment.get("page_number") is not None and int(segment.get("page_number")) <= page_number
    ]
    if not earlier:
        return None, None
    segment = earlier[-1]
    return (
        str(segment.get("section_name")) if segment.get("section_name") else None,
        str(segment.get("subsection_name")) if segment.get("subsection_name") else None,
    )


def extract_table_artifacts(
    document: str | list[dict[str, object]],
    section_segments: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(document, list):
        return []

    artifacts: list[dict[str, object]] = []
    seen_signatures: set[tuple[str | None, int | None, str]] = set()
    table_index = 0
    for page in document:
        page_number = int(page.get("page_number") or 0) or None
        blocks = page.get("blocks") or []
        for block_index, block in enumerate(blocks):
            block_text = _normalize_text(str(block.get("text") or ""))
            caption = _find_caption(blocks, block_index)
            label_match = TABLE_LABEL_RE.search(caption or "")
            if label_match:
                table_label = f"Table {label_match.group(1)}"
            else:
                first_line = block_text.splitlines()[0] if block_text.splitlines() else ""
                first_label_match = TABLE_LABEL_RE.search(first_line)
                table_label = f"Table {first_label_match.group(1)}" if first_label_match else None
            if not (_looks_like_table_payload(block_text) and (caption or table_label)):
                continue
            normalized_table_text = _normalize_text(f"{caption or ''}\n{block_text}".strip())
            signature = (table_label, page_number, normalized_table_text)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            section_name, subsection_name = _locate_section(page_number, section_segments)
            artifacts.append(
                {
                    "table_index": table_index,
                    "table_label": table_label,
                    "caption": caption,
                    "section_name": section_name,
                    "subsection_name": subsection_name,
                    "page_number": page_number,
                    "raw_table_text": block_text,
                    "normalized_table_text": normalized_table_text,
                    "table_type": _infer_table_type(caption, block_text),
                    "metric_names": _extract_metric_names(normalized_table_text),
                    "dataset_names": _extract_dataset_names(normalized_table_text),
                    "model_names": _extract_model_names(block_text),
                    "linked_chunk_indexes": _linked_chunk_indexes(page_number, section_segments),
                }
            )
            table_index += 1
    return artifacts
