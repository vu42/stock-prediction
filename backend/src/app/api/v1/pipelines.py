"""
Pipeline (Airflow) API endpoints.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ExternalServiceError
from app.core.logging import get_logger
from app.core.security import CurrentUser, DataScientistUser
from app.db.models import PipelineDAG, PipelineRun, PipelineRunLog, PipelineRunTask
from app.db.session import get_db
from app.integrations.airflow_client import airflow_client
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

logger = get_logger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipelines"])


def _extract_schedule_string(schedule_value: Any) -> str:
    """
    Extract schedule string from Airflow schedule_interval.
    
    Airflow 2.x returns schedule_interval as an object like:
    {'__type': 'CronExpression', 'value': '0 17 * * *'}
    
    This helper extracts the actual cron string.
    """
    if schedule_value is None:
        return "manual"
    if isinstance(schedule_value, str):
        return schedule_value
    if isinstance(schedule_value, dict):
        # Handle CronExpression object from Airflow 2.x
        if "value" in schedule_value:
            return schedule_value["value"]
        if "__type" in schedule_value:
            return schedule_value.get("value", "manual")
    return str(schedule_value) if schedule_value else "manual"


# ============================================================================
# DAG Catalog Endpoints
# ============================================================================


@router.post("/dags/sync")
async def sync_dags_from_airflow(
    current_user: DataScientistUser,
    db: Session = Depends(get_db),
):
    """
    Sync DAGs from Airflow to local database.
    Fetches all DAGs from Airflow and updates/creates local records.
    """
    try:
        airflow_dags = airflow_client.list_dags()
        synced_count = 0
        
        for airflow_dag in airflow_dags:
            dag_id = airflow_dag.get("dag_id")
            if not dag_id:
                continue
                
            # Check if DAG exists in local DB
            stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
            existing_dag = db.execute(stmt).scalar_one_or_none()
            
            if existing_dag:
                # Update existing DAG
                existing_dag.status = "paused" if airflow_dag.get("is_paused") else "active"
                existing_dag.description = airflow_dag.get("description")
                existing_dag.schedule_cron = _extract_schedule_string(airflow_dag.get("schedule_interval"))
                existing_dag.default_owner = ",".join(airflow_dag.get("owners", []))
                existing_dag.default_tags = [t.get("name") for t in airflow_dag.get("tags", [])]
            else:
                # Create new DAG record
                new_dag = PipelineDAG(
                    dag_id=dag_id,
                    name=dag_id.replace("_", " ").title(),
                    description=airflow_dag.get("description"),
                    status="paused" if airflow_dag.get("is_paused") else "active",
                    schedule_cron=_extract_schedule_string(airflow_dag.get("schedule_interval")),
                    schedule_label=None,
                    default_owner=",".join(airflow_dag.get("owners", [])),
                    default_tags=[t.get("name") for t in airflow_dag.get("tags", [])],
                )
                db.add(new_dag)
            
            synced_count += 1
        
        db.commit()
        logger.info(f"Synced {synced_count} DAGs from Airflow by user {current_user.username}")
        return {"message": f"Synced {synced_count} DAGs from Airflow", "count": synced_count}
        
    except ExternalServiceError:
        raise
    except Exception as e:
        logger.error(f"Failed to sync DAGs from Airflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to sync DAGs: {str(e)}",
        )


@router.get("/dags", response_model=list[DAGResponse])
async def list_dags(
    current_user: CurrentUser,
    source: str = Query("db", regex="^(db|airflow)$"),
    db: Session = Depends(get_db),
):
    """
    Get list of all DAGs.
    
    - **source**: "db" to fetch from local database (default), "airflow" to fetch directly from Airflow
    """
    if source == "airflow":
        # Fetch directly from Airflow
        try:
            airflow_dags = airflow_client.list_dags()
            result = []
            
            for airflow_dag in airflow_dags:
                dag_id = airflow_dag.get("dag_id")
                if not dag_id:
                    continue
                
                # Get last run from Airflow (prioritize running/queued runs)
                last_run = None
                try:
                    # First check for running runs
                    running_runs, _ = airflow_client.list_dag_runs(dag_id, limit=1, state=['running'])
                    if running_runs:
                        last_run = running_runs[0]
                    else:
                        # Check for queued runs
                        queued_runs, _ = airflow_client.list_dag_runs(dag_id, limit=1, state=['queued'])
                        if queued_runs:
                            last_run = queued_runs[0]
                        else:
                            # Get most recent completed run
                            runs, _ = airflow_client.list_dag_runs(dag_id, limit=1)
                            last_run = runs[0] if runs else None
                except Exception:
                    pass
                
                result.append(
                    DAGResponse(
                        dagId=dag_id,
                        name=dag_id.replace("_", " ").title(),
                        description=airflow_dag.get("description"),
                        status="paused" if airflow_dag.get("is_paused") else "active",
                        scheduleCron=_extract_schedule_string(airflow_dag.get("schedule_interval")),
                        scheduleLabel=None,
                        nextRunAt=airflow_dag.get("next_dagrun"),
                        lastRunAt=last_run.get("start_date") if last_run else None,
                        lastRunState=last_run.get("state") if last_run else None,
                    )
                )
            
            return result
            
        except ExternalServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch DAGs from Airflow: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch DAGs from Airflow: {str(e)}",
            )
    
    # Fetch from local database (default)
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
    source: str = Query("airflow", regex="^(db|airflow)$"),
    db: Session = Depends(get_db),
):
    """
    Get DAG details.
    
    - **source**: "airflow" to fetch from Airflow (default), "db" to fetch from local database
    """
    if source == "airflow":
        # Fetch directly from Airflow
        try:
            airflow_dag = airflow_client.get_dag(dag_id)
            
            # Get default_args from DB if available
            stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
            db_dag = db.execute(stmt).scalar_one_or_none()
            
            # Prefer DB values over Airflow values (DB has user edits)
            return DAGDetailResponse(
                dagId=airflow_dag.get("dag_id"),
                name=db_dag.name if db_dag else airflow_dag.get("dag_id", "").replace("_", " ").title(),
                description=airflow_dag.get("description"),
                status="paused" if airflow_dag.get("is_paused") else "active",
                owner=db_dag.default_owner if db_dag else ",".join(airflow_dag.get("owners", [])),
                tags=db_dag.default_tags if db_dag and db_dag.default_tags else [t.get("name") for t in airflow_dag.get("tags", [])],
                timezone=db_dag.timezone if db_dag else "Asia/Ho_Chi_Minh",
                scheduleCron=db_dag.schedule_cron if db_dag else (_extract_schedule_string(airflow_dag.get("schedule_interval")) or airflow_dag.get("timetable_summary") or "manual"),
                scheduleLabel=airflow_dag.get("timetable_description"),
                catchup=db_dag.catchup if db_dag else False,
                maxActiveRuns=db_dag.max_active_runs if db_dag else airflow_dag.get("max_active_runs", 1),
                defaultRetries=db_dag.default_retries if db_dag else 0,
                defaultRetryDelayMinutes=db_dag.default_retry_delay_minutes if db_dag else 5,
            )
            
        except ExternalServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch DAG from Airflow: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch DAG from Airflow: {str(e)}",
            )
    
    # Fetch from local database
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
        defaultRetries=dag.default_retries,
        defaultRetryDelayMinutes=dag.default_retry_delay_minutes,
    )


@router.post("/dags/{dag_id}/trigger", response_model=TriggerRunResponse)
async def trigger_dag_run(
    dag_id: str,
    current_user: DataScientistUser,
    request: TriggerRunRequest | None = None,
    db: Session = Depends(get_db),
):
    """
    Trigger a new DAG run via Airflow API.
    Auto-unpauses the DAG if it's currently paused.
    """
    try:
        # Check if DAG is paused and unpause if needed
        dag_info = airflow_client.get_dag(dag_id)
        if dag_info.get("is_paused"):
            logger.info(f"DAG {dag_id} is paused, unpausing before trigger")
            airflow_client.pause_dag(dag_id, paused=False)
            
            # Update local DB status
            stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
            local_dag = db.execute(stmt).scalar_one_or_none()
            if local_dag:
                local_dag.status = "active"
        
        # Trigger DAG run via Airflow API
        conf = request.conf if request else None
        airflow_response = airflow_client.trigger_dag_run(dag_id, conf=conf)
        
        # Extract run_id from Airflow response
        run_id = airflow_response.get("dag_run_id", f"manual__{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        
        # Ensure DAG exists in local DB before saving run (for FK constraint)
        stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
        existing_dag = db.execute(stmt).scalar_one_or_none()
        
        if not existing_dag:
            # Fetch DAG info from Airflow and create local record
            try:
                airflow_dag = airflow_client.get_dag(dag_id)
                new_dag = PipelineDAG(
                    dag_id=dag_id,
                    name=dag_id.replace("_", " ").title(),
                    description=airflow_dag.get("description"),
                    status="paused" if airflow_dag.get("is_paused") else "active",
                    schedule_cron=_extract_schedule_string(airflow_dag.get("schedule_interval")) or airflow_dag.get("timetable_summary") or "manual",
                    schedule_label=airflow_dag.get("timetable_description"),
                    default_owner=",".join(airflow_dag.get("owners", [])),
                    default_tags=[t.get("name") for t in airflow_dag.get("tags", [])],
                )
                db.add(new_dag)
                db.flush()  # Flush to make DAG available for FK
            except Exception as e:
                logger.warning(f"Could not sync DAG {dag_id} to local DB: {e}")
        
        # Save run record to our database
        run = PipelineRun(
            run_id=run_id,
            dag_id=dag_id,
            conf=conf,
            state=airflow_response.get("state", "queued"),
            start_time=datetime.now(timezone.utc),
            triggered_by_label=f"Manual trigger by {current_user.username}",
            triggered_by_user_id=current_user.id,
        )
        db.add(run)
        db.commit()
        
        logger.info(f"Triggered DAG {dag_id} run {run_id} by user {current_user.username}")
        return TriggerRunResponse(runId=run_id)
        
    except ExternalServiceError:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger DAG {dag_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to trigger DAG: {str(e)}",
        )


@router.post("/dags/{dag_id}/pause")
async def pause_dag(
    dag_id: str,
    request: PauseDAGRequest,
    current_user: DataScientistUser,
    db: Session = Depends(get_db),
):
    """
    Pause or resume a DAG via Airflow API.
    """
    try:
        # Pause/unpause via Airflow API
        airflow_client.pause_dag(dag_id, paused=request.paused)
        
        # Update local database record if exists
        stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
        dag = db.execute(stmt).scalar_one_or_none()
        
        if dag:
            dag.status = "paused" if request.paused else "active"
            db.commit()
        
        action = "paused" if request.paused else "resumed"
        logger.info(f"DAG {dag_id} {action} by user {current_user.username}")
        return {"message": f"DAG {action}"}
        
    except ExternalServiceError:
        raise
    except Exception as e:
        logger.error(f"Failed to pause/resume DAG {dag_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to pause/resume DAG: {str(e)}",
        )


@router.post("/dags/{dag_id}/stopRun")
async def stop_dag_run(
    dag_id: str,
    request: StopRunRequest,
    current_user: DataScientistUser,
    db: Session = Depends(get_db),
):
    """
    Stop an active DAG run via Airflow API.
    """
    try:
        # Stop run via Airflow API (marks as failed)
        airflow_client.stop_dag_run(dag_id, request.run_id)
        
        # Update local database record if exists
        stmt = select(PipelineRun).where(
            PipelineRun.run_id == request.run_id,
            PipelineRun.dag_id == dag_id,
        )
        run = db.execute(stmt).scalar_one_or_none()
        
        if run:
            run.state = "cancelled"
            run.end_time = datetime.now(timezone.utc)
            db.commit()
        
        logger.info(f"Stopped DAG run {request.run_id} by user {current_user.username}")
        return {"message": "Run stopped"}
        
    except ExternalServiceError:
        raise
    except Exception as e:
        logger.error(f"Failed to stop DAG run {request.run_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to stop DAG run: {str(e)}",
        )


# ============================================================================
# Run History Endpoints
# ============================================================================


@router.get("/dags/{dag_id}/runs", response_model=DAGRunsListResponse)
async def list_dag_runs(
    dag_id: str,
    current_user: DataScientistUser,
    source: str = Query("airflow", regex="^(db|airflow)$"),
    state: str | None = Query(None, description="Comma-separated states: running,success,failed,queued"),
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
    search_run_id: str | None = Query(None, alias="searchRunId"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
    db: Session = Depends(get_db),
):
    """
    List runs for a DAG with filters.
    
    - **source**: "airflow" to fetch from Airflow (default), "db" to fetch from local database
    - **state**: Comma-separated list of states (e.g., "running,queued" or "success,failed")
    """
    if source == "airflow":
        try:
            # Parse comma-separated states
            state_list = None
            if state:
                state_list = [s.strip().lower() for s in state.split(",") if s.strip()]
            
            # Convert date strings to ISO format for Airflow API
            start_date_gte = None
            start_date_lte = None
            if from_date:
                # Add time to make it start of day
                start_date_gte = f"{from_date}T00:00:00+00:00"
            if to_date:
                # Add time to make it end of day
                start_date_lte = f"{to_date}T23:59:59+00:00"
            
            # Fetch from Airflow
            offset = (page - 1) * page_size
            airflow_runs, total = airflow_client.list_dag_runs(
                dag_id,
                limit=page_size,
                offset=offset,
                state=state_list,
                start_date_gte=start_date_gte,
                start_date_lte=start_date_lte,
                dag_run_id_pattern=search_run_id,
            )
            
            # Calculate duration for each run
            data = []
            for run in airflow_runs:
                start_date = run.get("start_date")
                end_date = run.get("end_date")
                duration = None
                if start_date and end_date:
                    try:
                        from datetime import datetime as dt
                        start = dt.fromisoformat(start_date.replace("Z", "+00:00"))
                        end = dt.fromisoformat(end_date.replace("Z", "+00:00"))
                        duration = int((end - start).total_seconds())
                    except Exception:
                        pass
                
                data.append(
                    DAGRunResponse(
                        runId=run.get("dag_run_id"),
                        dagId=dag_id,
                        conf=run.get("conf"),
                        state=run.get("state"),
                        start=run.get("start_date"),
                        end=run.get("end_date"),
                        durationSeconds=duration,
                        triggeredBy=run.get("external_trigger", "scheduler") and "Manual" or "Scheduler",
                    )
                )
            
            return DAGRunsListResponse(
                data=data,
                meta=RunsListMeta(page=page, pageSize=page_size, total=total),
            )
            
        except ExternalServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch runs from Airflow: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch runs from Airflow: {str(e)}",
            )
    
    # Fetch from local database
    stmt = select(PipelineRun).where(PipelineRun.dag_id == dag_id)
    
    if state:
        # Support comma-separated states
        state_list = [s.strip().lower() for s in state.split(",") if s.strip()]
        if len(state_list) == 1:
            stmt = stmt.where(PipelineRun.state == state_list[0])
        elif len(state_list) > 1:
            stmt = stmt.where(PipelineRun.state.in_(state_list))
    
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
    dag_id: str = Query(..., alias="dagId"),
    source: str = Query("airflow", regex="^(db|airflow)$"),
    db: Session = Depends(get_db),
):
    """
    Get task graph for a run.
    
    - **dagId**: DAG identifier (required)
    - **source**: "airflow" to fetch from Airflow (default), "db" to fetch from local database
    """
    if source == "airflow":
        try:
            task_instances = airflow_client.list_task_instances(dag_id, run_id)
            
            nodes = [
                GraphNode(
                    id=t.get("task_id"),
                    label=t.get("task_id", "").replace("_", " ").title(),
                    state=t.get("state"),
                )
                for t in task_instances
            ]
            
            # Simple linear edges based on task order
            edges = []
            task_ids = [t.get("task_id") for t in task_instances]
            for i in range(len(task_ids) - 1):
                if task_ids[i] and task_ids[i + 1]:
                    edges.append(GraphEdge(**{"from": task_ids[i], "to": task_ids[i + 1]}))
            
            return GraphResponse(nodes=nodes, edges=edges)
            
        except ExternalServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch task graph from Airflow: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch task graph: {str(e)}",
            )
    
    # Fetch from local database
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
    dag_id: str = Query(..., alias="dagId"),
    source: str = Query("airflow", regex="^(db|airflow)$"),
    db: Session = Depends(get_db),
):
    """
    Get Gantt chart data for a run.
    
    - **dagId**: DAG identifier (required)
    - **source**: "airflow" to fetch from Airflow (default), "db" to fetch from local database
    """
    if source == "airflow":
        try:
            task_instances = airflow_client.list_task_instances(dag_id, run_id)
            
            gantt_tasks = [
                GanttTask(
                    taskId=t.get("task_id"),
                    label=t.get("task_id", "").replace("_", " ").title(),
                    start=t.get("start_date"),
                    end=t.get("end_date"),
                    state=t.get("state"),
                )
                for t in task_instances
            ]
            
            return GanttResponse(tasks=gantt_tasks)
            
        except ExternalServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Gantt data from Airflow: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch Gantt data: {str(e)}",
            )
    
    # Fetch from local database
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
    current_user: CurrentUser,
    dag_id: str = Query(..., alias="dagId"),
    task_id: str | None = Query(None, alias="taskId"),
    source: str = Query("airflow", regex="^(db|airflow)$"),
    cursor: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Get logs for a run.
    
    - **dagId**: DAG identifier (required)
    - **taskId**: Optional task ID to get logs for a specific task
    - **source**: "airflow" to fetch from Airflow (default), "db" to fetch from local database
    """
    if source == "airflow":
        try:
            entries = []
            
            if task_id:
                # Get logs for specific task
                log_content = airflow_client.get_task_logs(dag_id, run_id, task_id)
                for line in log_content.split("\n"):
                    if line.strip():
                        # Parse log line - simple parsing
                        level = "INFO"
                        if "ERROR" in line.upper():
                            level = "ERROR"
                        elif "WARNING" in line.upper() or "WARN" in line.upper():
                            level = "WARNING"
                        
                        entries.append(
                            LogEntry(
                                timestamp=datetime.now(timezone.utc),
                                level=level,
                                message=line,
                            )
                        )
            else:
                # Get logs for all tasks
                task_instances = airflow_client.list_task_instances(dag_id, run_id)
                
                for task in task_instances:
                    tid = task.get("task_id")
                    if not tid:
                        continue
                    
                    try:
                        log_content = airflow_client.get_task_logs(dag_id, run_id, tid)
                        for line in log_content.split("\n")[-50:]:  # Last 50 lines per task
                            if line.strip():
                                level = "INFO"
                                if "ERROR" in line.upper():
                                    level = "ERROR"
                                elif "WARNING" in line.upper() or "WARN" in line.upper():
                                    level = "WARNING"
                                
                                entries.append(
                                    LogEntry(
                                        timestamp=datetime.now(timezone.utc),
                                        level=level,
                                        message=f"[{tid}] {line}",
                                    )
                                )
                    except Exception as e:
                        logger.warning(f"Failed to get logs for task {tid}: {e}")
            
            return LogsResponse(entries=entries, nextCursor=None)
            
        except ExternalServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch logs from Airflow: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch logs: {str(e)}",
            )
    
    # Fetch from local database
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
    current_user: DataScientistUser,
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
        defaultRetries=dag.default_retries,
        defaultRetryDelayMinutes=dag.default_retry_delay_minutes,
    )

