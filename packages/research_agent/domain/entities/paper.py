from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Paper:
    id: str
    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    created_at: datetime | None = None
