import re

from research_agent.tools.embedder import get_max_embed_tokens, get_tokenizer, normalize_text_for_embedding


MAX_CHUNK_TOKENS = get_max_embed_tokens()
CHUNK_OVERLAP_TOKENS = 50


def _split_text_units(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    units: list[str] = []

    for paragraph in paragraphs:
        if paragraph.startswith("TABLE:"):
            units.append(paragraph)
            continue

        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("EQUATION:"):
                units.append(line)
                continue

            sentence_parts = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line)
                if sentence.strip()
            ]
            units.extend(sentence_parts or [line])

    return units


def split_text_into_chunks(text: str) -> list[dict[str, int | str]]:
    tokenizer = get_tokenizer()
    cleaned = " ".join(text.replace("\x00", "").split())
    if not cleaned:
        return []

    chunks: list[dict[str, int | str]] = []
    current_units: list[str] = []
    current_token_count = 0
    chunk_index = 0
    unit_buffer = _split_text_units(text.replace("\x00", "").strip())

    def token_count_for_text(value: str) -> int:
        return len(
            tokenizer(
                value,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )["input_ids"]
        )

    def flush_chunk() -> None:
        nonlocal chunk_index, current_units, current_token_count
        content = "\n".join(current_units).strip()
        normalized_content, normalized_token_count = normalize_text_for_embedding(content)
        if normalized_content:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "content": normalized_content,
                    "token_count": normalized_token_count,
                }
            )
            chunk_index += 1

        overlap_units: list[str] = []
        overlap_tokens = 0
        for unit in reversed(current_units):
            unit_tokens = token_count_for_text(unit)
            if overlap_tokens + unit_tokens > CHUNK_OVERLAP_TOKENS and overlap_units:
                break
            overlap_units.insert(0, unit)
            overlap_tokens += unit_tokens

        current_units = overlap_units
        current_token_count = overlap_tokens

    for unit in unit_buffer:
        unit_tokens = token_count_for_text(unit)
        if unit_tokens >= MAX_CHUNK_TOKENS:
            normalized_content, normalized_token_count = normalize_text_for_embedding(unit)
            if normalized_content:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "content": normalized_content,
                        "token_count": normalized_token_count,
                    }
                )
                chunk_index += 1
            current_units = []
            current_token_count = 0
            continue

        if current_units and current_token_count + unit_tokens > MAX_CHUNK_TOKENS:
            flush_chunk()

        current_units.append(unit)
        current_token_count += unit_tokens

    if current_units:
        flush_chunk()

    return chunks


def split_sections_into_chunks(
    sections: list[dict[str, str | int | None]]
) -> list[dict[str, int | str | None]]:
    all_chunks: list[dict[str, int | str | None]] = []
    global_chunk_index = 0

    for section in sections:
        section_chunks = split_text_into_chunks(str(section["content"]))
        for chunk in section_chunks:
            all_chunks.append(
                {
                    "chunk_index": global_chunk_index,
                    "section_name": section["section_name"],
                    "subsection_name": section["subsection_name"],
                    "content": chunk["content"],
                    "token_count": chunk["token_count"],
                }
            )
            global_chunk_index += 1

    return all_chunks
