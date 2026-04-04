"""add synthesis output to paper analysis

Revision ID: 20260322_0008
Revises: 20260321_0007
Create Date: 2026-03-22 10:15:00
"""

from alembic import op


revision = "20260322_0008"
down_revision = "20260321_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE paper_analysis ADD COLUMN IF NOT EXISTS synthesis_output JSONB")
    op.execute("ALTER TABLE paper_analysis ADD COLUMN IF NOT EXISTS synthesis_generated_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE paper_analysis DROP COLUMN IF EXISTS synthesis_generated_at")
    op.execute("ALTER TABLE paper_analysis DROP COLUMN IF EXISTS synthesis_output")
