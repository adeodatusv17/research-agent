from typing import TypedDict


class AnalysisState(TypedDict, total=False):
    paper_id: str
    extracted_sections: list[str]
    method_summary: dict
    repository_candidates: list[dict]
