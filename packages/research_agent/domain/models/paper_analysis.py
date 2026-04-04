import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from research_agent.infrastructure.db.session import Base


class PaperAnalysis(Base):
    __tablename__ = "paper_analysis"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    model_architecture: Mapped[str | None] = mapped_column(Text)
    architectures: Mapped[dict | list | None] = mapped_column(JSONB)
    dataset: Mapped[str | None] = mapped_column(Text)
    loss_function: Mapped[str | None] = mapped_column(Text)
    losses: Mapped[dict | list | None] = mapped_column(JSONB)
    training_objective: Mapped[str | None] = mapped_column(Text)
    optimizer: Mapped[str | None] = mapped_column(Text)
    optimizers: Mapped[dict | list | None] = mapped_column(JSONB)
    training_details: Mapped[dict | list | None] = mapped_column(JSONB)
    evaluation_metrics: Mapped[list | dict | None] = mapped_column(JSONB)
    contributions: Mapped[list | dict | None] = mapped_column(JSONB)
    domain: Mapped[str | None] = mapped_column(Text)
    inferred_structure: Mapped[dict | list | None] = mapped_column(JSONB)
    synthesis_output: Mapped[dict | list | None] = mapped_column(JSONB)
    synthesis_generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
