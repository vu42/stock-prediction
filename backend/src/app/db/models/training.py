"""
Training configuration and experiment models.
Maps to SPECS.md Section 6.2: training_configs, experiment_runs,
experiment_logs, experiment_ticker_artifacts
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class TrainingConfig(Base, TimestampMixin):
    """
    Versioned configuration blobs for training.
    Includes stock universe, data window, indicators, targets,
    models, ensemble, and reproducibility settings.
    """

    __tablename__ = "training_configs"
    __table_args__ = (
        UniqueConstraint("owner_user_id", "name", "version", name="uq_config_version"),
        Index("idx_training_configs_owner_active", "owner_user_id", "is_active"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="training_configs",
        foreign_keys=[owner_user_id],
    )
    experiment_runs: Mapped[list["ExperimentRun"]] = relationship(
        "ExperimentRun",
        back_populates="config",
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingConfig(id={self.id}, name={self.name}, version={self.version})>"
        )


class ExperimentState(str, Enum):
    """Experiment run state enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentRun(Base):
    """
    Individual training runs.
    """

    __tablename__ = "experiment_runs"
    __table_args__ = (
        Index("idx_experiment_runs_state_created", "state", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    config_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("training_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    scope: Mapped[str | None] = mapped_column(String(50), nullable=True)
    state: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=ExperimentState.PENDING.value,
    )
    progress_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    eta: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    config: Mapped["TrainingConfig"] = relationship(
        "TrainingConfig",
        back_populates="experiment_runs",
    )
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="experiment_runs",
        foreign_keys=[owner_user_id],
    )
    logs: Mapped[list["ExperimentLog"]] = relationship(
        "ExperimentLog",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    ticker_artifacts: Mapped[list["ExperimentTickerArtifact"]] = relationship(
        "ExperimentTickerArtifact",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ExperimentRun(id={self.id}, state={self.state})>"


class ExperimentLog(Base):
    """
    Ordered log entries per experiment run.
    """

    __tablename__ = "experiment_logs"
    __table_args__ = (Index("idx_experiment_logs_run_ts", "run_id", "ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiment_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    level: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    run: Mapped["ExperimentRun"] = relationship("ExperimentRun", back_populates="logs")

    def __repr__(self) -> str:
        return f"<ExperimentLog(run_id={self.run_id}, level={self.level})>"


class ExperimentTickerArtifact(Base):
    """
    Per-run, per-ticker artifact records.
    Includes metric JSON and URLs to evaluation/future plots,
    model/scaler binaries, and future_predictions CSVs.
    """

    __tablename__ = "experiment_ticker_artifacts"
    __table_args__ = (
        UniqueConstraint("run_id", "stock_id", name="uq_artifact_run_stock"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiment_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    stock_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
    )
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    evaluation_png_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    future_png_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_pkl_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    scaler_pkl_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    future_predictions_csv: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    run: Mapped["ExperimentRun"] = relationship(
        "ExperimentRun", back_populates="ticker_artifacts"
    )

    def __repr__(self) -> str:
        return f"<ExperimentTickerArtifact(run_id={self.run_id}, stock_id={self.stock_id})>"


# Forward reference
from app.db.models.users import User  # noqa: E402, F401
