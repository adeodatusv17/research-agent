from sqlalchemy import JSON, Date, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from research_agent.infrastructure.db.base import Base


class PaperModel(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_type: Mapped[str | None] = mapped_column(String(50))
    source_url: Mapped[str | None] = mapped_column(Text)
    arxiv_id: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list | None] = mapped_column(JSON)
    abstract: Mapped[str | None] = mapped_column(Text)
    publication_date: Mapped[str | None] = mapped_column(Date)
    venue: Mapped[str | None] = mapped_column(String(255))
    pdf_storage_path: Mapped[str | None] = mapped_column(Text)
    full_text: Mapped[str | None] = mapped_column(Text)
    parsed_sections: Mapped[dict | None] = mapped_column(JSON)
    metadata: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
