from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from research_agent.infrastructure.db.session import Base


class GeneratedExperiment(Base):
    __tablename__ = "generated_experiments"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    artifact_path: Mapped[str | None] = mapped_column(Text)
    generation_status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)
    model_code: Mapped[str | None] = mapped_column(Text)
    train_code: Mapped[str | None] = mapped_column(Text)
    dataset_code: Mapped[str | None] = mapped_column(Text)
    config_yaml: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
