"""
Pipeline (Airflow) models.
Maps to SPECS.md Section 6.2: pipeline_dags, pipeline_runs,
pipeline_run_tasks, pipeline_run_logs
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class DAGStatus(str, Enum):
    """DAG status enumeration."""

    ACTIVE = "active"
    PAUSED = "paused"


class PipelineDAG(Base, TimestampMixin):
    """
    Metadata and editable settings for Airflow DAGs.
    """

    __tablename__ = "pipeline_dags"

    dag_id: Mapped[str] = mapped_column(String(250), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=DAGStatus.ACTIVE.value,
    )
    schedule_cron: Mapped[str] = mapped_column(String(100), nullable=False)
    schedule_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    timezone: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="Asia/Ho_Chi_Minh",
    )
    catchup: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    max_active_runs: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    default_retries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    default_retry_delay_minutes: Mapped[int] = mapped_column(
        Integer, default=5, nullable=False
    )
    default_owner: Mapped[str | None] = mapped_column(String(100), nullable=True)
    default_tags: Mapped[list[str]] = mapped_column(
        ARRAY(String), default=[], nullable=False
    )

    # Relationships
    runs: Mapped[list["PipelineRun"]] = relationship(
        "PipelineRun",
        back_populates="dag",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<PipelineDAG(dag_id={self.dag_id}, status={self.status})>"


class PipelineRunState(str, Enum):
    """Pipeline run state enumeration."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    QUEUED = "queued"
    UP_FOR_RETRY = "up_for_retry"
    CANCELLED = "cancelled"


class PipelineRun(Base):
    """
    Individual DAG run records.
    """

    __tablename__ = "pipeline_runs"
    __table_args__ = (
        Index("idx_pipeline_runs_dag_state_start", "dag_id", "state", "start_time"),
    )

    run_id: Mapped[str] = mapped_column(String(250), primary_key=True)
    dag_id: Mapped[str] = mapped_column(
        String(250),
        ForeignKey("pipeline_dags.dag_id", ondelete="CASCADE"),
        nullable=False,
    )
    conf: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    state: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=PipelineRunState.QUEUED.value,
    )
    start_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    triggered_by_label: Mapped[str] = mapped_column(String(100), nullable=False)
    triggered_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    dag: Mapped["PipelineDAG"] = relationship("PipelineDAG", back_populates="runs")
    tasks: Mapped[list["PipelineRunTask"]] = relationship(
        "PipelineRunTask",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    logs: Mapped[list["PipelineRunLog"]] = relationship(
        "PipelineRunLog",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<PipelineRun(run_id={self.run_id}, dag_id={self.dag_id}, state={self.state})>"


class PipelineRunTask(Base):
    """
    Per-run task nodes for Graph and Gantt views.
    """

    __tablename__ = "pipeline_run_tasks"
    __table_args__ = (UniqueConstraint("run_id", "task_id", name="uq_run_task"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(250),
        ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id: Mapped[str] = mapped_column(String(250), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[str] = mapped_column(String(50), nullable=False)
    start_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="tasks")

    def __repr__(self) -> str:
        return f"<PipelineRunTask(run_id={self.run_id}, task_id={self.task_id})>"


class PipelineRunLog(Base):
    """
    Detailed log entries per pipeline run.
    """

    __tablename__ = "pipeline_run_logs"
    __table_args__ = (Index("idx_pipeline_run_logs_run_ts", "run_id", "ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(250),
        ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"),
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
    run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="logs")

    def __repr__(self) -> str:
        return f"<PipelineRunLog(run_id={self.run_id}, level={self.level})>"
