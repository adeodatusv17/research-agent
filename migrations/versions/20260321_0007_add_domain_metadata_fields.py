"""add domain metadata fields

Revision ID: 20260321_0007
Revises: 20260318_0006
Create Date: 2026-03-21 10:30:00
"""

from alembic import op


revision = "20260321_0007"
down_revision = "20260318_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS domain VARCHAR(50)")
    op.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS domain_confidence DOUBLE PRECISION")
    op.execute("ALTER TABLE paper_analysis ADD COLUMN IF NOT EXISTS domain TEXT")
    op.execute("ALTER TABLE paper_analysis ADD COLUMN IF NOT EXISTS inferred_structure JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE paper_analysis DROP COLUMN IF EXISTS inferred_structure")
    op.execute("ALTER TABLE paper_analysis DROP COLUMN IF EXISTS domain")
    op.execute("ALTER TABLE papers DROP COLUMN IF EXISTS domain_confidence")
    op.execute("ALTER TABLE papers DROP COLUMN IF EXISTS domain")
