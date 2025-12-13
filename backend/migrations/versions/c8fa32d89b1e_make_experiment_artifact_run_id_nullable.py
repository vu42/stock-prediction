"""make_experiment_artifact_run_id_nullable

Revision ID: c8fa32d89b1e
Revises: b6fda344785a
Create Date: 2025-12-13 19:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8fa32d89b1e'
down_revision: Union[str, None] = 'b6fda344785a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make run_id nullable in experiment_ticker_artifacts
    # This allows Airflow DAG runs to store artifacts without an experiment run
    op.alter_column(
        'experiment_ticker_artifacts',
        'run_id',
        existing_type=sa.dialects.postgresql.UUID(as_uuid=True),
        nullable=True
    )
    
    # Drop the existing unique constraint that requires run_id
    op.drop_constraint('uq_artifact_run_stock', 'experiment_ticker_artifacts', type_='unique')
    
    # Create a new unique constraint that allows null run_id
    # For null run_id (Airflow DAG runs), only allow one artifact per stock
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_artifact_stock_null_run 
        ON experiment_ticker_artifacts (stock_id) 
        WHERE run_id IS NULL
    """)
    
    # For non-null run_id (experiment runs), keep the original uniqueness
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_artifact_run_stock_not_null 
        ON experiment_ticker_artifacts (run_id, stock_id) 
        WHERE run_id IS NOT NULL
    """)


def downgrade() -> None:
    # Drop the new partial indexes
    op.execute("DROP INDEX IF EXISTS uq_artifact_stock_null_run")
    op.execute("DROP INDEX IF EXISTS uq_artifact_run_stock_not_null")
    
    # Restore the original unique constraint (will fail if null run_ids exist)
    op.create_unique_constraint(
        'uq_artifact_run_stock',
        'experiment_ticker_artifacts',
        ['run_id', 'stock_id']
    )
    
    # Make run_id not nullable again
    op.alter_column(
        'experiment_ticker_artifacts',
        'run_id',
        existing_type=sa.dialects.postgresql.UUID(as_uuid=True),
        nullable=False
    )

