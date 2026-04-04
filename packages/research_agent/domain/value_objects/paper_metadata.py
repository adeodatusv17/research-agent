from dataclasses import dataclass, field


@dataclass(slots=True)
class PaperMetadata:
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
