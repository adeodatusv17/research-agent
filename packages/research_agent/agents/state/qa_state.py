from typing import TypedDict


class QAState(TypedDict, total=False):
    db: object
    query: str
    query_type: str
    paper_id: str | None
    query_analysis: dict
    query_embedding: list[float]
    analysis_hits: dict
    retrieved_sections: list[dict]
    selected_sections: list[dict]
    retrieved_subsections: list[dict]
    retrieved_chunks: list[dict]
    filtered_chunks: list[dict]
    retrieval_confidence: float
    context: str
    answer: str
    sources: list[dict]
