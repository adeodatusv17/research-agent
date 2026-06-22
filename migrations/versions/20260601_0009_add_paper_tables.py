"""add paper tables

Revision ID: 20260601_0009
Revises: 20260322_0008
Create Date: 2026-06-01 11:05:00
"""

from alembic import op

from research_agent.tools.embedder import get_embedding_dimension


revision = "20260601_0009"
down_revision = "20260322_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    dimension = get_embedding_dimension()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS paper_tables (
            id UUID PRIMARY KEY,
            paper_id UUID NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
            table_index INTEGER NOT NULL,
            table_label VARCHAR(100) NULL,
            caption TEXT NULL,
            section_name TEXT NULL,
            subsection_name TEXT NULL,
            page_number INTEGER NULL,
            raw_table_text TEXT NOT NULL,
            normalized_table_text TEXT NOT NULL,
            table_type VARCHAR(50) NULL,
            metric_names JSONB NULL,
            dataset_names JSONB NULL,
            model_names JSONB NULL,
            linked_chunk_indexes JSONB NULL,
            token_count INTEGER NULL,
            embedding VECTOR({dimension}) NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_paper_tables_paper_id ON paper_tables (paper_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_paper_tables_table_type ON paper_tables (table_type)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_tables_embedding ON paper_tables USING ivfflat (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_paper_tables_embedding")
    op.execute("DROP INDEX IF EXISTS ix_paper_tables_table_type")
    op.execute("DROP INDEX IF EXISTS ix_paper_tables_paper_id")
    op.execute("DROP TABLE IF EXISTS paper_tables")
