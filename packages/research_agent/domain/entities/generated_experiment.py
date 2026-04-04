from dataclasses import dataclass


@dataclass(slots=True)
class GeneratedExperiment:
    id: str
    paper_id: str
    framework: str = "pytorch"
    generation_status: str = "draft"
