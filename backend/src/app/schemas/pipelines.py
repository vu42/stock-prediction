"""
Pipeline (Airflow) schemas for request/response validation.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# DAG Schemas
# ============================================================================


class DAGResponse(BaseModel):
    """DAG response schema."""

    dag_id: str = Field(..., alias="dagId")
    name: str
    description: str | None = None
    status: str  # active, paused
    schedule_cron: str = Field(..., alias="scheduleCron")
    schedule_label: str | None = Field(None, alias="scheduleLabel")
    next_run_at: datetime | None = Field(None, alias="nextRunAt")
    last_run_at: datetime | None = Field(None, alias="lastRunAt")
    last_run_state: str | None = Field(None, alias="lastRunState")

    class Config:
        populate_by_name = True
        from_attributes = True


class DAGDetailResponse(BaseModel):
    """DAG detail response schema."""

    dag_id: str = Field(..., alias="dagId")
    name: str
    description: str | None = None
    status: str
    owner: str | None = None
    tags: list[str] = Field(default_factory=list)
    timezone: str
    schedule_cron: str = Field(..., alias="scheduleCron")
    schedule_label: str | None = Field(None, alias="scheduleLabel")
    catchup: bool
    max_active_runs: int = Field(..., alias="maxActiveRuns")

    class Config:
        populate_by_name = True
        from_attributes = True


class DAGSettingsUpdate(BaseModel):
    """DAG settings update schema."""

    schedule_cron: str | None = Field(None, alias="scheduleCron")
    timezone: str | None = None
    catchup: bool | None = None
    max_active_runs: int | None = Field(None, alias="maxActiveRuns")
    default_args: "DefaultArgsUpdate | None" = Field(None, alias="defaultArgs")

    class Config:
        populate_by_name = True


class DefaultArgsUpdate(BaseModel):
    """Default arguments update schema."""

    retries: int | None = None
    retry_delay_minutes: int | None = Field(None, alias="retryDelayMinutes")
    owner: str | None = None
    tags: list[str] | None = None

    class Config:
        populate_by_name = True


# ============================================================================
# Run Schemas
# ============================================================================


class TriggerRunRequest(BaseModel):
    """Trigger run request schema."""

    conf: dict[str, Any] | None = None


class TriggerRunResponse(BaseModel):
    """Trigger run response schema."""

    run_id: str = Field(..., alias="runId")

    class Config:
        populate_by_name = True


class PauseDAGRequest(BaseModel):
    """Pause DAG request schema."""

    paused: bool


class StopRunRequest(BaseModel):
    """Stop run request schema."""

    run_id: str = Field(..., alias="runId")

    class Config:
        populate_by_name = True


class DAGRunResponse(BaseModel):
    """DAG run response schema."""

    run_id: str = Field(..., alias="runId")
    dag_id: str = Field(..., alias="dagId")
    conf: dict[str, Any] | None = None
    state: str
    start: datetime | None = None
    end: datetime | None = None
    duration_seconds: int | None = Field(None, alias="durationSeconds")
    triggered_by: str = Field(..., alias="triggeredBy")

    class Config:
        populate_by_name = True
        from_attributes = True


class DAGRunsListResponse(BaseModel):
    """DAG runs list response schema."""

    data: list[DAGRunResponse]
    meta: "RunsListMeta"

    class Config:
        populate_by_name = True


class RunsListMeta(BaseModel):
    """Runs list metadata schema."""

    page: int
    page_size: int = Field(..., alias="pageSize")
    total: int

    class Config:
        populate_by_name = True


# ============================================================================
# Graph/Gantt Schemas
# ============================================================================


class GraphNode(BaseModel):
    """Graph node schema."""

    id: str
    label: str
    state: str


class GraphEdge(BaseModel):
    """Graph edge schema."""

    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")

    class Config:
        populate_by_name = True


class GraphResponse(BaseModel):
    """Graph response schema."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GanttTask(BaseModel):
    """Gantt task schema."""

    task_id: str = Field(..., alias="taskId")
    label: str
    start: datetime | None = None
    end: datetime | None = None
    state: str

    class Config:
        populate_by_name = True


class GanttResponse(BaseModel):
    """Gantt response schema."""

    tasks: list[GanttTask]


# ============================================================================
# Log Schemas
# ============================================================================


class LogEntry(BaseModel):
    """Log entry schema."""

    timestamp: datetime
    level: str
    message: str


class LogsResponse(BaseModel):
    """Logs response schema."""

    entries: list[LogEntry]
    next_cursor: str | None = Field(None, alias="nextCursor")

    class Config:
        populate_by_name = True


# Update forward references
DAGSettingsUpdate.model_rebuild()
DAGRunsListResponse.model_rebuild()

