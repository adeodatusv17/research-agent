from research_agent.tools.pdf_parser import parse_pdf


def _sanitize_pdf_text(text: str) -> str:
    return text.replace("\x00", "")


def extract_text_from_pdf(file_path: str) -> str:
    pages = parse_pdf(file_path)
    return _sanitize_pdf_text("\n\n".join(page["text"] for page in pages)).strip()
