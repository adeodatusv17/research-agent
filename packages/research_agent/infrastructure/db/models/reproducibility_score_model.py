from sqlalchemy import DateTime, ForeignKey, JSON, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from research_agent.infrastructure.db.base import Base


class ReproducibilityScoreModel(Base):
    __tablename__ = "reproducibility_scores"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    overall_score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    dataset_availability_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    code_availability_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    hyperparameter_completeness_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    architecture_clarity_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    training_detail_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    summary: Mapped[str | None] = mapped_column(Text)
    strengths: Mapped[list | None] = mapped_column(JSON)
    gaps: Mapped[list | None] = mapped_column(JSON)
    evidence: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
