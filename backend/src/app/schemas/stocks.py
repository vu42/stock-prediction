"""
Stock schemas for request/response validation.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class StockResponse(BaseModel):
    """Stock detail response schema."""

    ticker: str
    name: str
    logo_url: str | None = Field(None, alias="logoUrl")
    description: str | None = None
    sector: str | None = None
    exchange: str | None = None
    market_cap: float | None = Field(None, alias="marketCap")
    trading_volume: int | None = Field(None, alias="tradingVolume")
    links: "StockLinksResponse | None" = None

    class Config:
        populate_by_name = True
        from_attributes = True


class StockLinksResponse(BaseModel):
    """Stock links response schema."""

    financial_report_url: str | None = Field(None, alias="financialReportUrl")
    company_website_url: str | None = Field(None, alias="companyWebsiteUrl")

    class Config:
        populate_by_name = True


class StockPriceResponse(BaseModel):
    """Stock price response schema."""

    date: date
    open: float | None
    high: float | None
    low: float | None
    close: float
    volume: int | None

    class Config:
        from_attributes = True


class PctChangeResponse(BaseModel):
    """Percent change response schema."""

    actual_pct: float | None = Field(None, alias="actualPct")
    # predicted_pct: float | None = Field(None, alias="predictedPct")
    actual_price: float | None = Field(None, alias="actualPrice")

    class Config:
        populate_by_name = True


class PredictedPctChangeResponse(BaseModel):
    """Predicted percent change response schema."""

    predicted_pct: float | None = Field(None, alias="predictedPct")
    predicted_price: float | None = Field(None, alias="predictedPrice")

    class Config:
        populate_by_name = True


class MarketTableItemResponse(BaseModel):
    """Market table row response schema."""

    symbol: str
    name: str
    sector: str | None = None
    current_price: float | None = Field(None, alias="currentPrice")
    pct_change: dict[str, PctChangeResponse] = Field(default_factory=dict, alias="pctChange")
    predicted_pct_change: dict[str, PredictedPctChangeResponse] = Field(
        default_factory=dict, alias="predictedPctChange"
    )
    sparkline_14d: list["SparklinePoint"] = Field(default_factory=list, alias="sparkline14d")

    class Config:
        populate_by_name = True


class SparklinePoint(BaseModel):
    """Sparkline data point schema."""

    date: str
    price: float
    is_predicted: bool = Field(False, alias="isPredicted")

    class Config:
        populate_by_name = True


class MarketTableMetaResponse(BaseModel):
    """Market table metadata response schema."""

    total: int
    page: int
    page_size: int = Field(..., alias="pageSize")
    sectors: list[str]

    class Config:
        populate_by_name = True


class MarketTableResponse(BaseModel):
    """Market table response schema."""

    data: list[MarketTableItemResponse]
    meta: MarketTableMetaResponse


class TopPickResponse(BaseModel):
    """Top pick response schema."""

    ticker: str
    name: str
    sector: str | None = None
    horizon_days: int = Field(..., alias="horizonDays")
    predicted_change_pct: float = Field(..., alias="predictedChangePct")
    current_price: float | None = Field(None, alias="currentPrice")

    class Config:
        populate_by_name = True


class MyListResponse(BaseModel):
    """My List (user saved stocks) response schema."""

    ticker: str
    name: str
    sector: str | None = None
    horizon_days: int = Field(..., alias="horizonDays")
    predicted_change_pct: float = Field(..., alias="predictedChangePct")
    current_price: float | None = Field(None, alias="currentPrice")
    added_at: str = Field(..., alias="addedAt")  # ISO timestamp

    class Config:
        populate_by_name = True


class StockCreate(BaseModel):
    """Stock creation schema."""

    ticker: str = Field(..., min_length=1, max_length=10)
    name: str = Field(..., min_length=1, max_length=255)
    sector: str | None = None
    exchange: str | None = None
    description: str | None = None
    logo_url: str | None = Field(None, alias="logoUrl")
    financial_report_url: str | None = Field(None, alias="financialReportUrl")
    company_website_url: str | None = Field(None, alias="companyWebsiteUrl")

    class Config:
        populate_by_name = True


class StockUpdate(BaseModel):
    """Stock update schema."""

    name: str | None = None
    sector: str | None = None
    exchange: str | None = None
    description: str | None = None
    logo_url: str | None = Field(None, alias="logoUrl")
    financial_report_url: str | None = Field(None, alias="financialReportUrl")
    company_website_url: str | None = Field(None, alias="companyWebsiteUrl")
    is_active: bool | None = Field(None, alias="isActive")

    class Config:
        populate_by_name = True


# Update forward references
StockResponse.model_rebuild()
MarketTableItemResponse.model_rebuild()

