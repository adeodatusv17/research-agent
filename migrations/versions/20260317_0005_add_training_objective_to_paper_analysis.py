"""add training_objective to paper_analysis

Revision ID: 20260317_0005
Revises: 20260317_0004
Create Date: 2026-03-17 22:35:00
"""

from alembic import op


revision = "20260317_0005"
down_revision = "20260317_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE paper_analysis ADD COLUMN IF NOT EXISTS training_objective TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE paper_analysis DROP COLUMN IF EXISTS training_objective")
