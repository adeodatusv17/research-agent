from dataclasses import dataclass


@dataclass(slots=True)
class TrainingSetup:
    optimizer: str | None = None
    batch_size: int | None = None
    epochs: int | None = None
