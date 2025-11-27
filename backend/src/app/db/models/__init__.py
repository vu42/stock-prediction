"""
SQLAlchemy models.
Import all models here for Alembic to detect them.
"""

from app.db.models.pipelines import (
    DAGStatus,
    PipelineDAG,
    PipelineRun,
    PipelineRunLog,
    PipelineRunState,
    PipelineRunTask,
)
from app.db.models.stocks import (
    CrawlMetadata,
    FreshnessState,
    ModelHorizonMetric,
    ModelStatus,
    Stock,
    StockPredictionPoint,
    StockPredictionSummary,
    StockPrice,
)
from app.db.models.training import (
    ExperimentLog,
    ExperimentRun,
    ExperimentState,
    ExperimentTickerArtifact,
    TrainingConfig,
)
from app.db.models.users import AuthToken, User, UserRole

__all__ = [
    # Users
    "User",
    "UserRole",
    "AuthToken",
    # Stocks
    "Stock",
    "StockPrice",
    "StockPredictionSummary",
    "StockPredictionPoint",
    "ModelStatus",
    "ModelHorizonMetric",
    "FreshnessState",
    "CrawlMetadata",
    # Training
    "TrainingConfig",
    "ExperimentRun",
    "ExperimentState",
    "ExperimentLog",
    "ExperimentTickerArtifact",
    # Pipelines
    "PipelineDAG",
    "DAGStatus",
    "PipelineRun",
    "PipelineRunState",
    "PipelineRunTask",
    "PipelineRunLog",
]
