"""add paper_subsections table

Revision ID: 20260317_0001
Revises:
Create Date: 2026-03-17 20:10:00
"""

from alembic import op


revision = "20260317_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_subsections (
            id UUID PRIMARY KEY,
            paper_id UUID NOT NULL REFERENCES papers (id),
            section_name TEXT NOT NULL,
            subsection_name TEXT NULL,
            page_number INTEGER NULL,
            content TEXT NOT NULL,
            token_count INTEGER NULL,
            embedding vector(384) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_subsections_paper_id ON paper_subsections (paper_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_paper_subsections_paper_id")
    op.execute("DROP TABLE IF EXISTS paper_subsections")
