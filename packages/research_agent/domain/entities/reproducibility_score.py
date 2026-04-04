from dataclasses import dataclass


@dataclass(slots=True)
class ReproducibilityScore:
    id: str
    paper_id: str
    overall_score: float
    summary: str | None = None
