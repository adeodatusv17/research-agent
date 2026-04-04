from dataclasses import dataclass


@dataclass(slots=True)
class PaperChunk:
    id: str
    paper_id: str
    chunk_index: int
    content: str
