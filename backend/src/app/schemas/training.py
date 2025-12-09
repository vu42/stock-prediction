"""
Training configuration and experiment schemas.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Training Configuration Schemas
# ============================================================================


class StockUniverseConfig(BaseModel):
    """Stock universe configuration schema."""

    tickers: list[str] = Field(default_factory=list)
    use_all_vn30: bool = Field(default=False, alias="useAllVn30")

    class Config:
        populate_by_name = True


class DataWindowConfig(BaseModel):
    """Data window configuration schema."""

    mode: str = "last_n_days"  # "last_n_days" or "date_range"
    last_n_days: int | None = Field(240, alias="lastNDays")
    start_date: str | None = Field(None, alias="startDate")
    end_date: str | None = Field(None, alias="endDate")
    skip_refetch: bool = Field(default=False, alias="skipRefetch")

    class Config:
        populate_by_name = True


class IndicatorsConfig(BaseModel):
    """Technical indicators configuration schema."""

    sma_windows: list[int] = Field(default=[5, 20, 60], alias="smaWindows")
    ema_fast: int = Field(default=12, alias="emaFast")
    ema_slow: int = Field(default=26, alias="emaSlow")
    use_roc: bool = Field(default=True, alias="useRoc")
    rsi_window: int = Field(default=14, alias="rsiWindow")
    macd_fast: int = Field(default=12, alias="macdFast")
    macd_slow: int = Field(default=26, alias="macdSlow")
    macd_signal: int = Field(default=9, alias="macdSignal")
    bb_window: int = Field(default=20, alias="bbWindow")
    bb_std: int = Field(default=2, alias="bbStd")
    atr_window: int = Field(default=14, alias="atrWindow")
    volume_ma_window: int = Field(default=20, alias="volumeMaWindow")
    leakage_guard: bool = Field(default=True, alias="leakageGuard")

    class Config:
        populate_by_name = True


class TargetSplitsConfig(BaseModel):
    """Target and splits configuration schema."""

    horizons: list[int] = Field(default=[3, 7, 15, 30])
    lookback_window: int = Field(default=60, alias="lookbackWindow")
    train_pct: int = Field(default=80, alias="trainPct")
    test_pct: int = Field(default=20, alias="testPct")

    class Config:
        populate_by_name = True


class ModelParams(BaseModel):
    """Individual model parameters schema."""

    enabled: bool = True
    n_estimators: int | None = Field(None, alias="nEstimators")
    max_depth: int | None = Field(None, alias="maxDepth")
    learning_rate: float | None = Field(None, alias="learningRate")
    c: float | None = None
    epsilon: float | None = None
    gamma: str | None = None
    alpha: float | None = None

    class Config:
        populate_by_name = True


class ModelsConfig(BaseModel):
    """Models configuration schema."""

    random_forest: ModelParams = Field(default_factory=ModelParams, alias="randomForest")
    gradient_boosting: ModelParams = Field(default_factory=ModelParams, alias="gradientBoosting")
    svr: ModelParams = Field(default_factory=ModelParams)
    ridge: ModelParams = Field(default_factory=ModelParams)

    class Config:
        populate_by_name = True


class ScalingConfig(BaseModel):
    """Scaling configuration schema."""

    method: str = "standard"  # "standard" or "none"


class EnsembleConfig(BaseModel):
    """Ensemble configuration schema."""

    method: str = "mean"  # "mean", "median", "weighted"
    learn_weights: bool = Field(default=False, alias="learnWeights")

    class Config:
        populate_by_name = True


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration schema."""

    random_seed: int = Field(default=42, alias="randomSeed")

    class Config:
        populate_by_name = True


class TrainingConfigSchema(BaseModel):
    """Full training configuration schema."""

    universe: StockUniverseConfig = Field(default_factory=StockUniverseConfig)
    data_window: DataWindowConfig = Field(default_factory=DataWindowConfig, alias="dataWindow")
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    targets: TargetSplitsConfig = Field(default_factory=TargetSplitsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)

    class Config:
        populate_by_name = True


class TrainingConfigCreate(BaseModel):
    """Create training configuration schema."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    config: TrainingConfigSchema


class TrainingConfigResponse(BaseModel):
    """Training configuration response schema."""

    id: str
    name: str
    description: str | None = None
    config: TrainingConfigSchema
    version: int
    is_active: bool = Field(..., alias="isActive")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True


class ConfigSavedResponse(BaseModel):
    """Config saved response schema."""

    config_id: str = Field(..., alias="configId")
    saved_at: datetime = Field(..., alias="savedAt")

    class Config:
        populate_by_name = True


# ============================================================================
# Validation Schemas
# ============================================================================


class ValidateConfigRequest(BaseModel):
    """Validate config request schema - accepts either config object or configId."""

    config: TrainingConfigSchema | None = None
    config_id: str | None = Field(None, alias="configId")

    class Config:
        populate_by_name = True


class ValidationBlocker(BaseModel):
    """Validation blocker schema."""

    field_path: str = Field(..., alias="fieldPath")
    message: str

    class Config:
        populate_by_name = True


class RunPreview(BaseModel):
    """Run preview schema."""

    est_runtime_minutes: int = Field(..., alias="estRuntimeMinutes")
    est_cost: float | None = Field(None, alias="estCost")

    class Config:
        populate_by_name = True


class ValidateConfigResponse(BaseModel):
    """Validate config response schema."""

    is_valid: bool = Field(..., alias="isValid")
    blockers: list[ValidationBlocker] = Field(default_factory=list)
    warnings: list[ValidationBlocker] = Field(default_factory=list)
    run_preview: RunPreview | None = Field(None, alias="runPreview")

    class Config:
        populate_by_name = True


# ============================================================================
# Experiment Run Schemas
# ============================================================================


class ExperimentRunCreate(BaseModel):
    """Create experiment run schema."""

    config_id: str = Field(..., alias="configId")
    scope: str = "selected"  # "all_vn30" or "selected"
    seeds: dict[str, int] | None = None
    notes: str | None = None

    class Config:
        populate_by_name = True


class ExperimentRunResponse(BaseModel):
    """Experiment run response schema."""

    run_id: str = Field(..., alias="runId")
    created_at: datetime = Field(..., alias="createdAt")
    state: str
    config_summary: str | None = Field(None, alias="configSummary")
    progress_pct: float | None = Field(None, alias="progressPct")
    eta: datetime | None = None
    started_at: datetime | None = Field(None, alias="startedAt")
    finished_at: datetime | None = Field(None, alias="finishedAt")
    scope: str | None = None
    notes: str | None = None

    class Config:
        populate_by_name = True
        from_attributes = True


class ExperimentRunCreatedResponse(BaseModel):
    """Run created response schema."""

    run_id: str = Field(..., alias="runId")

    class Config:
        populate_by_name = True


class LogEntry(BaseModel):
    """Log entry schema."""

    timestamp: datetime
    level: str
    message: str


class LogTailResponse(BaseModel):
    """Log tail response schema."""

    entries: list[LogEntry]
    next_cursor: str | None = Field(None, alias="nextCursor")

    class Config:
        populate_by_name = True


class RunsListResponse(BaseModel):
    """Runs list response schema."""

    data: list[ExperimentRunResponse]
    meta: dict[str, Any]


# ============================================================================
# Artifact Schemas
# ============================================================================


class ArtifactFile(BaseModel):
    """Artifact file schema."""

    type: str  # evaluation_png, future_png, model_pkl, scaler_pkl, future_predictions_csv
    url: str


class TickerArtifact(BaseModel):
    """Per-ticker artifact schema."""

    ticker: str
    metrics: dict[str, float] | None = None
    files: list[ArtifactFile] = Field(default_factory=list)


class ArtifactResponse(BaseModel):
    """Artifacts response schema."""

    ticker_artifacts: list[TickerArtifact] = Field(
        default_factory=list, alias="tickerArtifacts"
    )

    class Config:
        populate_by_name = True

