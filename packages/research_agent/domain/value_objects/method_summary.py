from dataclasses import dataclass


@dataclass(slots=True)
class MethodSummary:
    architecture: str | None = None
    dataset: str | None = None
    loss_function: str | None = None
