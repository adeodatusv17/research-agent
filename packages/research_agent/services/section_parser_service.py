import re


SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "related_work",
    "related work": "related_work",
    "related works": "related_work",
    "prior work": "related_work",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "model": "method",
    "proposed method": "method",
    "architecture": "method",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "implementation details": "experiments",
    "training setup": "experiments",
    "evaluation": "results",
    "results": "results",
    "analysis": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "references": "references",
}


def _sanitize_text(value: str) -> str:
    return value.replace("\x00", "")


def _iter_document_lines(document: str | list[dict]) -> list[dict[str, str | int | None]]:
    if isinstance(document, str):
        return [{"text": _sanitize_text(line).rstrip(), "page_number": None} for line in document.splitlines()]

    lines: list[dict[str, str | int | None]] = []
    for page in document:
        page_number = int(page["page_number"])
        for block in page.get("blocks", []):
            for line in _sanitize_text(str(block["text"])).splitlines():
                lines.append({"text": line.rstrip(), "page_number": page_number})
    return lines


def _normalize_heading(line: str) -> str:
    stripped = re.sub(r"^\s*\d+(\.\d+)*\s*", "", line).strip().lower()
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped


def _detect_primary_section(line: str) -> tuple[str, str, str | None] | None:
    normalized = _normalize_heading(line)
    if normalized in SECTION_ALIASES:
        return SECTION_ALIASES[normalized], line.strip(), None

    lowered = line.strip().lower()
    for alias, canonical in SECTION_ALIASES.items():
        if lowered.startswith(f"{alias}:") or lowered.startswith(f"{alias} -") or lowered.startswith(
            f"{alias} "
        ):
            remainder = line.strip()[len(alias) :].lstrip(" :-\t")
            return canonical, alias.title(), remainder or None
    return None


def _looks_like_subsection(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    if re.match(r"^\d+\.\s+.+", stripped):
        return True
    if re.match(r"^\d+\.\d+(\.\d+)*\s+.+", stripped):
        return True
    if re.match(r"^[IVXLC]+\.?\s+.+", stripped):
        return True
    if re.match(r"^[A-Z]\.\s+.+", stripped):
        return True
    if any(marker in stripped for marker in [",", "[", "]", "\"", "“", "”"]) or "et al" in stripped.lower():
        return False
    if stripped.endswith("."):
        return False
    words = stripped.split()
    if 2 <= len(words) <= 6 and stripped == stripped.title() and ":" not in stripped:
        return True
    return False


def parse_sections(document: str | list[dict]) -> list[dict[str, str | int | None]]:
    lines = _iter_document_lines(document)
    sections: list[dict[str, str | int | None]] = []
    current_section = "front_matter"
    current_heading = "Front Matter"
    current_subsection: str | None = None
    buffer: list[str] = []
    order = 0
    current_page_number: int | None = None

    def flush_buffer() -> None:
        nonlocal order, buffer, current_page_number
        content = _sanitize_text("\n".join(line for line in buffer if line.strip()).strip())
        if not content:
            buffer = []
            return
        sections.append(
            {
                "section_name": current_section,
                "section_heading": current_heading,
                "subsection_name": current_subsection,
                "section_order": order,
                "page_number": current_page_number,
                "content": content,
            }
        )
        order += 1
        buffer = []
        current_page_number = None

    for line_info in lines:
        line = str(line_info["text"])
        page_number = line_info["page_number"]

        if not line.strip():
            if buffer:
                buffer.append("")
            continue

        primary_section = _detect_primary_section(line)
        if primary_section:
            flush_buffer()
            current_section, current_heading, remainder = primary_section
            current_subsection = None
            current_page_number = int(page_number) if page_number is not None else None
            if remainder:
                buffer.append(remainder)
            continue

        if _looks_like_subsection(line):
            flush_buffer()
            current_subsection = line.strip()
            current_page_number = int(page_number) if page_number is not None else None
            continue

        if current_page_number is None and page_number is not None:
            current_page_number = int(page_number)
        buffer.append(line)

    flush_buffer()

    if not sections:
        raw_text = (
            _sanitize_text(document).strip()
            if isinstance(document, str)
            else _sanitize_text("\n".join(str(line["text"]) for line in lines)).strip()
        )
        if not raw_text:
            return []
        sections.append(
            {
                "section_name": "front_matter",
                "section_heading": "Front Matter",
                "subsection_name": None,
                "section_order": 0,
                "page_number": 1,
                "content": raw_text,
            }
        )

    return sections


def build_section_index_entries(
    sections: list[dict[str, str | int | None]]
) -> list[dict[str, str | int | None]]:
    merged: dict[str, dict[str, str | int | None]] = {}

    for section in sections:
        section_name = str(section["section_name"])
        if section_name not in merged:
            merged[section_name] = {
                "section_name": section_name,
                "section_heading": section.get("section_heading"),
                "section_order": section.get("section_order", 0),
                "page_number": section.get("page_number"),
                "content": _sanitize_text(str(section["content"])),
            }
        else:
            merged[section_name]["content"] = (
                _sanitize_text(f"{merged[section_name]['content']}\n\n{section['content']}").strip()
            )

    return sorted(merged.values(), key=lambda section: int(section["section_order"]))


def build_subsection_index_entries(
    sections: list[dict[str, str | int | None]]
) -> list[dict[str, str | int | None]]:
    merged: dict[tuple[str, str | None], dict[str, str | int | None]] = {}

    for section in sections:
        key = (str(section["section_name"]), str(section["subsection_name"]) if section.get("subsection_name") else None)
        if key not in merged:
            merged[key] = {
                "section_name": key[0],
                "subsection_name": key[1],
                "page_number": section.get("page_number"),
                "content": _sanitize_text(str(section["content"])),
                "section_order": section.get("section_order", 0),
            }
        else:
            merged[key]["content"] = (
                _sanitize_text(f"{merged[key]['content']}\n\n{section['content']}").strip()
            )

    return sorted(merged.values(), key=lambda subsection: int(subsection["section_order"]))
