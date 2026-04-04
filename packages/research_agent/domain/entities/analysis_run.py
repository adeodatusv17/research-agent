from dataclasses import dataclass


@dataclass(slots=True)
class AnalysisRun:
    id: str
    paper_id: str
    run_type: str
    status: str
