from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from research_agent.infrastructure.db.base import Base


class GeneratedExperimentModel(Base):
    __tablename__ = "generated_experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    framework: Mapped[str] = mapped_column(String(64), default="pytorch")
    generation_status: Mapped[str] = mapped_column(String(32), default="draft")
    artifact_path: Mapped[str | None] = mapped_column(Text)
    model_code: Mapped[str | None] = mapped_column(Text)
    dataset_code: Mapped[str | None] = mapped_column(Text)
    train_code: Mapped[str | None] = mapped_column(Text)
    config_yaml: Mapped[str | None] = mapped_column(Text)
    requirements_txt: Mapped[str | None] = mapped_column(Text)
    generation_notes: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
