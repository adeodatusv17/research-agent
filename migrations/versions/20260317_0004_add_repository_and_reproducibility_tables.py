"""add repository and reproducibility tables

Revision ID: 20260317_0004
Revises: 20260317_0003
Create Date: 2026-03-17 22:05:00
"""

from alembic import op


revision = "20260317_0004"
down_revision = "20260317_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_repositories (
            id UUID PRIMARY KEY,
            paper_id UUID NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
            repo_url TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_paper_repositories_paper_id ON paper_repositories (paper_id)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS reproducibility_scores (
            id UUID PRIMARY KEY,
            paper_id UUID NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
            dataset_available BOOLEAN NULL,
            code_available BOOLEAN NOT NULL DEFAULT FALSE,
            hyperparameter_completeness DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            training_detail_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            evaluation_protocol_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            overall_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            summary TEXT NULL,
            evidence JSONB NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_reproducibility_scores_paper_id ON reproducibility_scores (paper_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_reproducibility_scores_paper_id")
    op.execute("DROP TABLE IF EXISTS reproducibility_scores")
    op.execute("DROP INDEX IF EXISTS ix_paper_repositories_paper_id")
    op.execute("DROP TABLE IF EXISTS paper_repositories")
