"""add generated_experiments table

Revision ID: 20260318_0006
Revises: 20260317_0005
Create Date: 2026-03-18 10:05:00
"""

from alembic import op


revision = "20260318_0006"
down_revision = "20260317_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS generated_experiments (
            id TEXT PRIMARY KEY,
            paper_id UUID NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
            artifact_path TEXT NULL,
            generation_status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT NULL,
            model_code TEXT NULL,
            train_code TEXT NULL,
            dataset_code TEXT NULL,
            config_yaml TEXT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_generated_experiments_paper_id ON generated_experiments (paper_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_generated_experiments_paper_id")
    op.execute("DROP TABLE IF EXISTS generated_experiments")
