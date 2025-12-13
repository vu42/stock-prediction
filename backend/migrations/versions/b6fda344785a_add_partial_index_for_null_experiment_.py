"""add_partial_index_for_null_experiment_run

Revision ID: b6fda344785a
Revises: 9ae522b394d5
Create Date: 2025-12-13 07:18:30.640049

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b6fda344785a'
down_revision: Union[str, None] = '9ae522b394d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create partial unique index for predictions without experiment_run_id
    # This prevents duplicate predictions when training via Airflow (no run_id)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_pred_point_null_run 
        ON stock_prediction_points (stock_id, horizon_days, prediction_date) 
        WHERE experiment_run_id IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_pred_point_null_run")

