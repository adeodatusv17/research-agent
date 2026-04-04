import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Float, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from research_agent.infrastructure.db.session import Base


class ReproducibilityScore(Base):
    __tablename__ = "reproducibility_scores"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    dataset_available: Mapped[bool | None] = mapped_column(Boolean)
    code_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    hyperparameter_completeness: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    training_detail_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    evaluation_protocol_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    summary: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
