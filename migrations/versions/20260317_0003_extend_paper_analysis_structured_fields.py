"""extend paper_analysis structured fields

Revision ID: 20260317_0003
Revises: 20260317_0002
Create Date: 2026-03-17 21:20:00
"""

from alembic import op


revision = "20260317_0003"
down_revision = "20260317_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE paper_analysis ADD COLUMN IF NOT EXISTS architectures JSONB")
    op.execute("ALTER TABLE paper_analysis ADD COLUMN IF NOT EXISTS losses JSONB")
    op.execute("ALTER TABLE paper_analysis ADD COLUMN IF NOT EXISTS optimizers JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE paper_analysis DROP COLUMN IF EXISTS optimizers")
    op.execute("ALTER TABLE paper_analysis DROP COLUMN IF EXISTS losses")
    op.execute("ALTER TABLE paper_analysis DROP COLUMN IF EXISTS architectures")
