"""
Pipeline (Airflow) API endpoints.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import CurrentUser, RequireDataScientist
from app.db.models import PipelineDAG, PipelineRun, PipelineRunLog, PipelineRunTask
from app.db.session import get_db
from app.schemas import (
    DAGDetailResponse,
    DAGResponse,
    DAGRunResponse,
    DAGRunsListResponse,
    DAGSettingsUpdate,
    GanttResponse,
    GanttTask,
    GraphEdge,
    GraphNode,
    GraphResponse,
    LogEntry,
    LogsResponse,
    PauseDAGRequest,
    RunsListMeta,
    StopRunRequest,
    TriggerRunRequest,
    TriggerRunResponse,
)

router = APIRouter(prefix="/pipeline", tags=["Pipelines"])


# ============================================================================
# DAG Catalog Endpoints
# ============================================================================


@router.get("/dags", response_model=list[DAGResponse])
async def list_dags(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Get list of all DAGs.
    """
    stmt = select(PipelineDAG).order_by(PipelineDAG.name)
    dags = db.execute(stmt).scalars().all()
    
    result = []
    for dag in dags:
        # Get last run
        last_run_stmt = (
            select(PipelineRun)
            .where(PipelineRun.dag_id == dag.dag_id)
            .order_by(PipelineRun.start_time.desc())
            .limit(1)
        )
        last_run = db.execute(last_run_stmt).scalar_one_or_none()
        
        result.append(
            DAGResponse(
                dagId=dag.dag_id,
                name=dag.name,
                description=dag.description,
                status=dag.status,
                scheduleCron=dag.schedule_cron,
                scheduleLabel=dag.schedule_label,
                nextRunAt=None,  # Would need to calculate from cron
                lastRunAt=last_run.start_time if last_run else None,
                lastRunState=last_run.state if last_run else None,
            )
        )
    
    return result


@router.get("/dags/{dag_id}", response_model=DAGDetailResponse)
async def get_dag(
    dag_id: str,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Get DAG details.
    """
    stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
    dag = db.execute(stmt).scalar_one_or_none()
    
    if not dag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DAG not found: {dag_id}",
        )
    
    return DAGDetailResponse(
        dagId=dag.dag_id,
        name=dag.name,
        description=dag.description,
        status=dag.status,
        owner=dag.default_owner,
        tags=dag.default_tags or [],
        timezone=dag.timezone,
        scheduleCron=dag.schedule_cron,
        scheduleLabel=dag.schedule_label,
        catchup=dag.catchup,
        maxActiveRuns=dag.max_active_runs,
    )


@router.post("/dags/{dag_id}/trigger", response_model=TriggerRunResponse)
async def trigger_dag_run(
    dag_id: str,
    request: TriggerRunRequest | None = None,
    current_user: CurrentUser = RequireDataScientist,
    db: Session = Depends(get_db),
):
    """
    Trigger a new DAG run.
    """
    # Check DAG exists
    stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
    dag = db.execute(stmt).scalar_one_or_none()
    
    if not dag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DAG not found: {dag_id}",
        )
    
    # Create run record
    run_id = f"manual__{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    
    run = PipelineRun(
        run_id=run_id,
        dag_id=dag_id,
        conf=request.conf if request else None,
        state="queued",
        triggered_by_label=f"Manual trigger by {current_user.username}",
        triggered_by_user_id=current_user.id,
    )
    db.add(run)
    db.commit()
    
    # TODO: Actually trigger via Airflow API
    # from app.integrations.airflow_client import trigger_dag
    # await trigger_dag(dag_id, run_id, request.conf if request else None)
    
    return TriggerRunResponse(runId=run_id)


@router.post("/dags/{dag_id}/pause")
async def pause_dag(
    dag_id: str,
    request: PauseDAGRequest,
    current_user: CurrentUser = RequireDataScientist,
    db: Session = Depends(get_db),
):
    """
    Pause or resume a DAG.
    """
    stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
    dag = db.execute(stmt).scalar_one_or_none()
    
    if not dag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DAG not found: {dag_id}",
        )
    
    dag.status = "paused" if request.paused else "active"
    db.commit()
    
    # TODO: Actually pause via Airflow API
    # from app.integrations.airflow_client import pause_dag
    # await pause_dag(dag_id, request.paused)
    
    return {"message": f"DAG {'paused' if request.paused else 'resumed'}"}


@router.post("/dags/{dag_id}/stopRun")
async def stop_dag_run(
    dag_id: str,
    request: StopRunRequest,
    current_user: CurrentUser = RequireDataScientist,
    db: Session = Depends(get_db),
):
    """
    Stop an active DAG run.
    """
    stmt = select(PipelineRun).where(
        PipelineRun.run_id == request.run_id,
        PipelineRun.dag_id == dag_id,
    )
    run = db.execute(stmt).scalar_one_or_none()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {request.run_id}",
        )
    
    if run.state not in ["running", "queued"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run is not active (state: {run.state})",
        )
    
    run.state = "cancelled"
    run.end_time = datetime.now()
    db.commit()
    
    # TODO: Actually stop via Airflow API
    
    return {"message": "Run stopped"}


# ============================================================================
# Run History Endpoints
# ============================================================================


@router.get("/dags/{dag_id}/runs", response_model=DAGRunsListResponse)
async def list_dag_runs(
    dag_id: str,
    state: str | None = Query(None),
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
    search_run_id: str | None = Query(None, alias="searchRunId"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
    current_user: CurrentUser = RequireDataScientist,
    db: Session = Depends(get_db),
):
    """
    List runs for a DAG with filters.
    """
    stmt = select(PipelineRun).where(PipelineRun.dag_id == dag_id)
    
    if state:
        stmt = stmt.where(PipelineRun.state == state.lower())
    
    if search_run_id:
        stmt = stmt.where(PipelineRun.run_id.ilike(f"%{search_run_id}%"))
    
    # Count total
    from sqlalchemy import func
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = db.execute(count_stmt).scalar() or 0
    
    # Pagination
    offset = (page - 1) * page_size
    stmt = stmt.order_by(PipelineRun.start_time.desc()).offset(offset).limit(page_size)
    
    runs = db.execute(stmt).scalars().all()
    
    data = [
        DAGRunResponse(
            runId=run.run_id,
            dagId=run.dag_id,
            conf=run.conf,
            state=run.state,
            start=run.start_time,
            end=run.end_time,
            durationSeconds=run.duration_seconds,
            triggeredBy=run.triggered_by_label,
        )
        for run in runs
    ]
    
    return DAGRunsListResponse(
        data=data,
        meta=RunsListMeta(page=page, pageSize=page_size, total=total),
    )


@router.get("/runs/{run_id}", response_model=DAGRunResponse)
async def get_run(
    run_id: str,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Get run details.
    """
    stmt = select(PipelineRun).where(PipelineRun.run_id == run_id)
    run = db.execute(stmt).scalar_one_or_none()
    
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )
    
    return DAGRunResponse(
        runId=run.run_id,
        dagId=run.dag_id,
        conf=run.conf,
        state=run.state,
        start=run.start_time,
        end=run.end_time,
        durationSeconds=run.duration_seconds,
        triggeredBy=run.triggered_by_label,
    )


@router.get("/runs/{run_id}/graph", response_model=GraphResponse)
async def get_run_graph(
    run_id: str,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Get task graph for a run.
    """
    stmt = select(PipelineRunTask).where(PipelineRunTask.run_id == run_id)
    tasks = db.execute(stmt).scalars().all()
    
    nodes = [
        GraphNode(id=t.task_id, label=t.label, state=t.state)
        for t in tasks
    ]
    
    # Simple linear edges for now
    edges = []
    task_ids = [t.task_id for t in tasks]
    for i in range(len(task_ids) - 1):
        edges.append(GraphEdge(**{"from": task_ids[i], "to": task_ids[i + 1]}))
    
    return GraphResponse(nodes=nodes, edges=edges)


@router.get("/runs/{run_id}/gantt", response_model=GanttResponse)
async def get_run_gantt(
    run_id: str,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Get Gantt chart data for a run.
    """
    stmt = select(PipelineRunTask).where(PipelineRunTask.run_id == run_id)
    tasks = db.execute(stmt).scalars().all()
    
    gantt_tasks = [
        GanttTask(
            taskId=t.task_id,
            label=t.label,
            start=t.start_time,
            end=t.end_time,
            state=t.state,
        )
        for t in tasks
    ]
    
    return GanttResponse(tasks=gantt_tasks)


@router.get("/runs/{run_id}/logs", response_model=LogsResponse)
async def get_run_logs(
    run_id: str,
    cursor: str | None = Query(None),
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Get logs for a run.
    """
    stmt = (
        select(PipelineRunLog)
        .where(PipelineRunLog.run_id == run_id)
        .order_by(PipelineRunLog.ts.desc())
        .limit(100)
    )
    logs = db.execute(stmt).scalars().all()
    
    entries = [
        LogEntry(timestamp=log.ts, level=log.level, message=log.message)
        for log in reversed(logs)
    ]
    
    return LogsResponse(entries=entries, nextCursor=None)


# ============================================================================
# DAG Settings Endpoints
# ============================================================================


@router.patch("/dags/{dag_id}/settings", response_model=DAGDetailResponse)
async def update_dag_settings(
    dag_id: str,
    request: DAGSettingsUpdate,
    current_user: CurrentUser = RequireDataScientist,
    db: Session = Depends(get_db),
):
    """
    Update DAG settings.
    """
    stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
    dag = db.execute(stmt).scalar_one_or_none()
    
    if not dag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DAG not found: {dag_id}",
        )
    
    # Update fields
    if request.schedule_cron is not None:
        dag.schedule_cron = request.schedule_cron
    if request.timezone is not None:
        dag.timezone = request.timezone
    if request.catchup is not None:
        dag.catchup = request.catchup
    if request.max_active_runs is not None:
        dag.max_active_runs = request.max_active_runs
    
    if request.default_args:
        if request.default_args.retries is not None:
            dag.default_retries = request.default_args.retries
        if request.default_args.retry_delay_minutes is not None:
            dag.default_retry_delay_minutes = request.default_args.retry_delay_minutes
        if request.default_args.owner is not None:
            dag.default_owner = request.default_args.owner
        if request.default_args.tags is not None:
            dag.default_tags = request.default_args.tags
    
    db.commit()
    db.refresh(dag)
    
    return DAGDetailResponse(
        dagId=dag.dag_id,
        name=dag.name,
        description=dag.description,
        status=dag.status,
        owner=dag.default_owner,
        tags=dag.default_tags or [],
        timezone=dag.timezone,
        scheduleCron=dag.schedule_cron,
        scheduleLabel=dag.schedule_label,
        catchup=dag.catchup,
        maxActiveRuns=dag.max_active_runs,
    )

