"""cascade delete paper relations

Revision ID: 20260317_0002
Revises: 20260317_0001
Create Date: 2026-03-17 20:45:00
"""

from alembic import op


revision = "20260317_0002"
down_revision = "20260317_0001"
branch_labels = None
depends_on = None


PAPER_FK_CONSTRAINTS = [
    ("analysis_runs", "analysis_runs_paper_id_fkey"),
    ("paper_analysis", "paper_analysis_paper_id_fkey"),
    ("paper_chunks", "paper_chunks_paper_id_fkey"),
    ("paper_sections", "paper_sections_paper_id_fkey"),
    ("paper_subsections", "paper_subsections_paper_id_fkey"),
]


def upgrade() -> None:
    for table_name, constraint_name in PAPER_FK_CONSTRAINTS:
        op.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS "{constraint_name}"')
        op.execute(
            f'''
            ALTER TABLE "{table_name}"
            ADD CONSTRAINT "{constraint_name}"
            FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            '''
        )


def downgrade() -> None:
    for table_name, constraint_name in PAPER_FK_CONSTRAINTS:
        op.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS "{constraint_name}"')
        op.execute(
            f'''
            ALTER TABLE "{table_name}"
            ADD CONSTRAINT "{constraint_name}"
            FOREIGN KEY (paper_id) REFERENCES papers (id)
            '''
        )
