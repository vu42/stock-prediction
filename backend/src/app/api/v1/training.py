"""
Training configuration and experiment API endpoints.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import CurrentUser, DataScientistUser
from app.db.models import ExperimentLog, ExperimentRun, ExperimentState, TrainingConfig
from app.db.session import get_db
from app.schemas import (
    ArtifactResponse,
    ConfigSavedResponse,
    ExperimentRunCreate,
    ExperimentRunCreatedResponse,
    ExperimentRunResponse,
    LogEntry,
    LogTailResponse,
    RunsListResponse,
    TrainingConfigCreate,
    TrainingConfigResponse,
    TrainingConfigSchema,
    ValidateConfigResponse,
    ValidationBlocker,
)

router = APIRouter(tags=["Training"])


# ============================================================================
# Configuration Endpoints
# ============================================================================


@router.get("/features/config", response_model=TrainingConfigResponse)
async def get_training_config(
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Load latest saved training configuration for the current user.
    """
    stmt = (
        select(TrainingConfig)
        .where(
            TrainingConfig.owner_user_id == current_user.id,
            TrainingConfig.is_active == True,  # noqa: E712
        )
        .order_by(TrainingConfig.updated_at.desc())
        .limit(1)
    )
    config = db.execute(stmt).scalar_one_or_none()

    if not config:
        # Return default config
        return TrainingConfigResponse(
            id="default",
            name="Default Configuration",
            description="Default training configuration",
            config=TrainingConfigSchema(),
            version=1,
            isActive=True,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        )

    return TrainingConfigResponse(
        id=str(config.id),
        name=config.name,
        description=config.description,
        config=TrainingConfigSchema(**config.config),
        version=config.version,
        isActive=config.is_active,
        createdAt=config.created_at,
        updatedAt=config.updated_at,
    )


@router.post("/features/config", response_model=ConfigSavedResponse)
async def save_training_config(
    request: TrainingConfigCreate,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Create or update a saved configuration.
    """
    # Check if config with same name exists for this user
    stmt = select(TrainingConfig).where(
        TrainingConfig.owner_user_id == current_user.id,
        TrainingConfig.name == request.name,
        TrainingConfig.is_active == True,  # noqa: E712
    )
    existing = db.execute(stmt).scalar_one_or_none()

    if existing:
        # Update existing (create new version)
        existing.is_active = False
        new_version = existing.version + 1
    else:
        new_version = 1

    # Create new config
    config = TrainingConfig(
        owner_user_id=current_user.id,
        name=request.name,
        description=request.description,
        config=request.config.model_dump(),
        version=new_version,
        is_active=True,
    )
    db.add(config)
    db.commit()
    db.refresh(config)

    return ConfigSavedResponse(
        configId=str(config.id),
        savedAt=config.created_at,
    )


@router.post("/features/validate", response_model=ValidateConfigResponse)
async def validate_training_config(
    request: TrainingConfigSchema,
    current_user: CurrentUser,
):
    """
    Validate config and compute a rough run preview / cost estimate.
    """
    blockers: list[ValidationBlocker] = []
    warnings: list[ValidationBlocker] = []

    # Validate stock universe
    if not request.universe.use_all_vn30 and not request.universe.tickers:
        blockers.append(
            ValidationBlocker(
                fieldPath="universe.tickers",
                message="At least one ticker must be selected or enable 'Use all VN30'",
            )
        )

    # Validate data window
    if request.data_window.mode == "last_n_days":
        if not request.data_window.last_n_days or request.data_window.last_n_days < 60:
            blockers.append(
                ValidationBlocker(
                    fieldPath="dataWindow.lastNDays",
                    message="Last N days must be at least 60 for sufficient training data",
                )
            )

    # Validate horizons
    if not request.targets.horizons:
        blockers.append(
            ValidationBlocker(
                fieldPath="targets.horizons",
                message="At least one prediction horizon must be selected",
            )
        )

    # Validate train/test split
    if request.targets.train_pct + request.targets.test_pct != 100:
        blockers.append(
            ValidationBlocker(
                fieldPath="targets.trainPct",
                message="Train % + Test % must equal 100",
            )
        )

    # Warnings
    if request.scaling.method == "none" and request.models.svr.enabled:
        warnings.append(
            ValidationBlocker(
                fieldPath="scaling.method",
                message="SVR works best with scaling enabled",
            )
        )

    # Estimate runtime
    num_tickers = 30 if request.universe.use_all_vn30 else len(request.universe.tickers)
    est_minutes = num_tickers * 2  # ~2 minutes per ticker

    return ValidateConfigResponse(
        isValid=len(blockers) == 0,
        blockers=blockers,
        warnings=warnings,
        runPreview={
            "estRuntimeMinutes": est_minutes,
            "estCost": None,
        },
    )


# ============================================================================
# Experiment Run Endpoints
# ============================================================================


@router.post("/experiments/run", response_model=ExperimentRunCreatedResponse)
async def start_experiment_run(
    request: ExperimentRunCreate,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Start a new training run using a validated configuration.
    """
    # Get config
    stmt = select(TrainingConfig).where(TrainingConfig.id == request.config_id)
    config = db.execute(stmt).scalar_one_or_none()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration not found: {request.config_id}",
        )

    # Create experiment run
    run = ExperimentRun(
        config_id=config.id,
        owner_user_id=current_user.id,
        scope=request.scope,
        state=ExperimentState.PENDING.value,
        notes=request.notes,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Queue the training job via RQ
    from app.integrations.queue import enqueue_training_job

    enqueue_training_job(str(run.id))

    return ExperimentRunCreatedResponse(runId=str(run.id))


@router.get("/experiments/{run_id}", response_model=ExperimentRunResponse)
async def get_experiment_run(
    run_id: UUID,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Get experiment run details.
    """
    stmt = select(ExperimentRun).where(ExperimentRun.id == run_id)
    run = db.execute(stmt).scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment run not found: {run_id}",
        )

    return ExperimentRunResponse(
        runId=str(run.id),
        state=run.state,
        progressPct=float(run.progress_pct) if run.progress_pct else None,
        eta=run.eta,
        startedAt=run.started_at,
        finishedAt=run.finished_at,
        scope=run.scope,
        notes=run.notes,
    )


@router.get("/experiments/{run_id}/logs/tail", response_model=LogTailResponse)
async def get_experiment_logs(
    run_id: UUID,
    current_user: DataScientistUser,
    cursor: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Get log tail for an experiment run.
    """
    stmt = (
        select(ExperimentLog)
        .where(ExperimentLog.run_id == run_id)
        .order_by(ExperimentLog.ts.desc())
        .limit(100)
    )
    logs = db.execute(stmt).scalars().all()

    entries = [
        LogEntry(
            timestamp=log.ts,
            level=log.level,
            message=log.message,
        )
        for log in reversed(logs)
    ]

    return LogTailResponse(
        entries=entries,
        nextCursor=None,
    )


@router.get("/experiments/runs", response_model=RunsListResponse)
async def list_experiment_runs(
    current_user: DataScientistUser,
    limit: int = Query(10, ge=1, le=100),
    page: int = Query(1, ge=1),
    state: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    List past experiment runs.
    """
    stmt = select(ExperimentRun).where(ExperimentRun.owner_user_id == current_user.id)

    if state:
        stmt = stmt.where(ExperimentRun.state == state)

    stmt = stmt.order_by(ExperimentRun.created_at.desc())

    # Pagination
    offset = (page - 1) * limit
    stmt = stmt.offset(offset).limit(limit)

    runs = db.execute(stmt).scalars().all()

    data = [
        ExperimentRunResponse(
            runId=str(run.id),
            state=run.state,
            progressPct=float(run.progress_pct) if run.progress_pct else None,
            eta=run.eta,
            startedAt=run.started_at,
            finishedAt=run.finished_at,
            scope=run.scope,
            notes=run.notes,
        )
        for run in runs
    ]

    return RunsListResponse(
        data=data,
        meta={"page": page, "limit": limit},
    )


@router.get("/experiments/{run_id}/artifacts", response_model=ArtifactResponse)
async def get_experiment_artifacts(
    run_id: UUID,
    current_user: DataScientistUser,
    ticker: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    List artifacts per ticker for an experiment run.
    """
    from app.db.models import ExperimentTickerArtifact, Stock

    stmt = select(ExperimentTickerArtifact).where(
        ExperimentTickerArtifact.run_id == run_id
    )

    if ticker:
        stmt = stmt.join(Stock).where(Stock.ticker == ticker)

    artifacts = db.execute(stmt).scalars().all()

    ticker_artifacts = []
    for a in artifacts:
        # Get ticker symbol
        stock_stmt = select(Stock).where(Stock.id == a.stock_id)
        stock = db.execute(stock_stmt).scalar_one_or_none()

        files = []
        if a.evaluation_png_url:
            files.append({"type": "evaluation_png", "url": a.evaluation_png_url})
        if a.future_png_url:
            files.append({"type": "future_png", "url": a.future_png_url})
        if a.model_pkl_url:
            files.append({"type": "model_pkl", "url": a.model_pkl_url})
        if a.scaler_pkl_url:
            files.append({"type": "scaler_pkl", "url": a.scaler_pkl_url})
        if a.future_predictions_csv:
            files.append(
                {"type": "future_predictions_csv", "url": a.future_predictions_csv}
            )

        ticker_artifacts.append(
            {
                "ticker": stock.ticker if stock else "unknown",
                "metrics": a.metrics,
                "files": files,
            }
        )

    return ArtifactResponse(tickerArtifacts=ticker_artifacts)
