import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from research_agent.infrastructure.db.session import Base
from research_agent.tools.embedder import get_embedding_dimension


class PaperTable(Base):
    __tablename__ = "paper_tables"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    table_index: Mapped[int] = mapped_column(Integer, nullable=False)
    table_label: Mapped[str | None] = mapped_column(String(100))
    caption: Mapped[str | None] = mapped_column(Text)
    section_name: Mapped[str | None] = mapped_column(Text)
    subsection_name: Mapped[str | None] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    raw_table_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_table_text: Mapped[str] = mapped_column(Text, nullable=False)
    table_type: Mapped[str | None] = mapped_column(String(50))
    metric_names: Mapped[list[str] | None] = mapped_column(JSONB)
    dataset_names: Mapped[list[str] | None] = mapped_column(JSONB)
    model_names: Mapped[list[str] | None] = mapped_column(JSONB)
    linked_chunk_indexes: Mapped[list[int] | None] = mapped_column(JSONB)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(get_embedding_dimension()), nullable=False)
