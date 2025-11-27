"""
Prediction schemas for request/response validation.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class HorizonPrediction(BaseModel):
    """Single horizon prediction schema."""

    predicted_change_pct: float | None = Field(None, alias="predictedChangePct")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Stock predictions response schema."""

    ticker: str
    horizons: dict[str, HorizonPrediction]


class ChartPoint(BaseModel):
    """Chart data point schema."""

    date: str
    actual_price: float | None = Field(None, alias="actualPrice")
    predicted_price: float | None = Field(None, alias="predictedPrice")

    class Config:
        populate_by_name = True


class ChartDataResponse(BaseModel):
    """Chart data response schema."""

    points: list[ChartPoint]
    range: str


class HorizonMetric(BaseModel):
    """Horizon metric schema."""

    mape_pct: float = Field(..., alias="mapePct")

    class Config:
        populate_by_name = True


class ModelStatusResponse(BaseModel):
    """Model status response schema."""

    state: str  # fresh, stable, stale
    last_updated_at: str | None = Field(None, alias="lastUpdatedAt")
    metrics: dict[str, HorizonMetric]

    class Config:
        populate_by_name = True


class PredictionSummaryCreate(BaseModel):
    """Create prediction summary schema."""

    stock_id: int = Field(..., alias="stockId")
    as_of_date: str = Field(..., alias="asOfDate")
    horizon_days: int = Field(..., alias="horizonDays")
    predicted_change_pct: float = Field(..., alias="predictedChangePct")
    experiment_run_id: str | None = Field(None, alias="experimentRunId")

    class Config:
        populate_by_name = True

