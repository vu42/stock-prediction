"""
Pipeline (Airflow) models.
Maps to SPECS.md Section 6.2: pipeline_dags, pipeline_runs,
pipeline_run_tasks, pipeline_run_logs
"""

import uuid
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
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
from sqlalchemy.orm import relationship

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

    dag_id = Column(String(250), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        String(20),
        nullable=False,
        default=DAGStatus.ACTIVE.value,
    )
    schedule_cron = Column(String(100), nullable=False)
    schedule_label = Column(String(100), nullable=True)
    timezone = Column(
        String(50),
        nullable=False,
        default="Asia/Ho_Chi_Minh",
    )
    catchup = Column(Boolean, default=False, nullable=False)
    max_active_runs = Column(Integer, default=1, nullable=False)
    default_retries = Column(Integer, default=0, nullable=False)
    default_retry_delay_minutes = Column(Integer, default=5, nullable=False)
    default_owner = Column(String(100), nullable=True)
    default_tags = Column(ARRAY(String), default=[], nullable=False)

    # Relationships
    runs = relationship(
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

    run_id = Column(String(250), primary_key=True)
    dag_id = Column(
        String(250),
        ForeignKey("pipeline_dags.dag_id", ondelete="CASCADE"),
        nullable=False,
    )
    conf = Column(JSONB, nullable=True)
    state = Column(
        String(20),
        nullable=False,
        default=PipelineRunState.QUEUED.value,
    )
    start_time = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    end_time = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    duration_seconds = Column(Integer, nullable=True)
    triggered_by_label = Column(String(100), nullable=False)
    triggered_by_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    dag = relationship("PipelineDAG", back_populates="runs")
    tasks = relationship(
        "PipelineRunTask",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    logs = relationship(
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

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(
        String(250),
        ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id = Column(String(250), nullable=False)
    label = Column(String(255), nullable=False)
    state = Column(String(50), nullable=False)
    start_time = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    end_time = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    run = relationship("PipelineRun", back_populates="tasks")

    def __repr__(self) -> str:
        return f"<PipelineRunTask(run_id={self.run_id}, task_id={self.task_id})>"


class PipelineRunLog(Base):
    """
    Detailed log entries per pipeline run.
    """

    __tablename__ = "pipeline_run_logs"
    __table_args__ = (Index("idx_pipeline_run_logs_run_ts", "run_id", "ts"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(
        String(250),
        ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)

    # Relationships
    run = relationship("PipelineRun", back_populates="logs")

    def __repr__(self) -> str:
        return f"<PipelineRunLog(run_id={self.run_id}, level={self.level})>"
