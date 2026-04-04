import re
from typing import TypedDict

import fitz


class Block(TypedDict):
    block_number: int
    block_type: str
    bbox: list[float]
    text: str


class Page(TypedDict):
    page_number: int
    text: str
    blocks: list[Block]


def _normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _is_equation_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    equation_markers = ["=", "∑", "Σ", "argmax", "argmin", "log", "exp", "lambda", "β", "α"]
    return any(marker in stripped for marker in equation_markers) and len(stripped.split()) <= 40


def _format_table_like_block(lines: list[str]) -> str | None:
    row_cells: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in re.split(r"\s{2,}|\t+", line) if cell.strip()]
        if len(cells) >= 2:
            row_cells.append(cells)

    if len(row_cells) < 2:
        return None

    target_width = max(len(row) for row in row_cells)
    normalized_rows = [row + [""] * (target_width - len(row)) for row in row_cells]
    formatted_rows = [" | ".join(row) for row in normalized_rows]
    return "TABLE:\n" + "\n".join(formatted_rows)


def _extract_block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        line_text = "".join(span.get("text", "") for span in spans)
        line_text = _normalize_whitespace(line_text)
        if line_text:
            lines.append(line_text)

    if not lines:
        return ""

    table_text = _format_table_like_block(lines)
    if table_text:
        return table_text

    normalized_lines: list[str] = []
    for line in lines:
        if _is_equation_line(line):
            normalized_lines.append(f"EQUATION: {line}")
        else:
            normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def parse_pdf(file_path: str) -> list[Page]:
    document = fitz.open(file_path)
    pages: list[Page] = []

    try:
        for page in document:
            page_dict = page.get_text("dict", sort=True)
            page_blocks: list[Block] = []
            page_text_parts: list[str] = []

            for block_number, block in enumerate(page_dict.get("blocks", [])):
                if block.get("type") != 0:
                    continue

                block_text = _extract_block_text(block)
                if not block_text:
                    continue

                page_blocks.append(
                    {
                        "block_number": block_number,
                        "block_type": "text",
                        "bbox": [float(value) for value in block.get("bbox", (0, 0, 0, 0))],
                        "text": block_text,
                    }
                )
                page_text_parts.append(block_text)

            pages.append(
                {
                    "page_number": page.number + 1,
                    "text": "\n\n".join(page_text_parts).strip(),
                    "blocks": page_blocks,
                }
            )
    finally:
        document.close()

    return pages


def extract_title_from_pages(pages: list[Page]) -> str | None:
    if not pages:
        return None

    first_page = pages[0]
    for block in first_page.get("blocks", []):
        text = _normalize_whitespace(block["text"])
        if not text:
            continue
        if len(text) < 12 or len(text) > 200:
            continue
        lowered = text.lower()
        if lowered in {"abstract", "introduction"}:
            continue
        if "@" in text or "{" in text or "google.com" in lowered:
            continue
        if sum(1 for char in text if char.isalpha()) < 10:
            continue
        return text
    return None
