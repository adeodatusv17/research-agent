"""add hnsw vector indexes

Revision ID: 20260622_0010
Revises: 20260601_0009
Create Date: 2026-06-22 14:55:00
"""

from alembic import op


revision = "20260622_0010"
down_revision = "20260601_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("DROP INDEX IF EXISTS ix_paper_tables_embedding")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_chunks_embedding_hnsw "
        "ON paper_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_sections_embedding_hnsw "
        "ON paper_sections USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_subsections_embedding_hnsw "
        "ON paper_subsections USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_tables_embedding_hnsw "
        "ON paper_tables USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_paper_tables_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_paper_subsections_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_paper_sections_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_paper_chunks_embedding_hnsw")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_tables_embedding "
        "ON paper_tables USING ivfflat (embedding vector_cosine_ops)"
    )
