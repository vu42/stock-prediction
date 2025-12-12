"""
Stock and prediction models.
Maps to SPECS.md Section 6.2: stocks, stock_prices, stock_prediction_summaries,
stock_prediction_points, model_statuses, model_horizon_metrics
"""

import uuid
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
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
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin


class Stock(Base, TimestampMixin):
    """
    Master data for VN30 tickers.
    """

    __tablename__ = "stocks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(
        String(10),
        unique=True,
        nullable=False,
        index=True,
    )
    name = Column(String(255), nullable=False)
    sector = Column(String(100), nullable=True)
    exchange = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    logo_url = Column(String(500), nullable=True)
    financial_report_url = Column(String(500), nullable=True)
    company_website_url = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    prices = relationship(
        "StockPrice",
        back_populates="stock",
        cascade="all, delete-orphan",
    )
    prediction_summaries = relationship(
        "StockPredictionSummary",
        back_populates="stock",
        cascade="all, delete-orphan",
    )
    prediction_points = relationship(
        "StockPredictionPoint",
        back_populates="stock",
        cascade="all, delete-orphan",
    )
    model_statuses = relationship(
        "ModelStatus",
        back_populates="stock",
        cascade="all, delete-orphan",
    )
    saved_by_users = relationship(
        "UserSavedStock",
        back_populates="stock",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Stock(id={self.id}, ticker={self.ticker}, name={self.name})>"


class StockPrice(Base):
    """
    Daily OHLCV data per stock.
    """

    __tablename__ = "stock_prices"
    __table_args__ = (
        UniqueConstraint("stock_id", "price_date", name="uq_stock_price_date"),
        Index("idx_stock_prices_stock_date", "stock_id", "price_date"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(
        BigInteger,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
    )
    price_date = Column(Date, nullable=False)
    open_price = Column(Numeric(18, 4), nullable=True)
    high_price = Column(Numeric(18, 4), nullable=True)
    low_price = Column(Numeric(18, 4), nullable=True)
    close_price = Column(Numeric(18, 4), nullable=False)
    volume = Column(BigInteger, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    stock = relationship("Stock", back_populates="prices")

    def __repr__(self) -> str:
        return f"<StockPrice(stock_id={self.stock_id}, date={self.price_date}, close={self.close_price})>"


class StockPredictionSummary(Base):
    """
    Per (stock, as_of_date, horizon_days) predicted % change.
    Used for Top Picks / Should Buy / Should Sell and Home table.
    """

    __tablename__ = "stock_prediction_summaries"
    __table_args__ = (
        UniqueConstraint(
            "stock_id", "as_of_date", "horizon_days", name="uq_pred_summary"
        ),
        Index("idx_pred_summaries_horizon_date", "horizon_days", "as_of_date"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(
        BigInteger,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
    )
    as_of_date = Column(Date, nullable=False)
    horizon_days = Column(Integer, nullable=False)
    predicted_change_pct = Column(Numeric(7, 4), nullable=False)
    experiment_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiment_runs.id"),
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    stock = relationship("Stock", back_populates="prediction_summaries")

    def __repr__(self) -> str:
        return f"<StockPredictionSummary(stock_id={self.stock_id}, horizon={self.horizon_days}d, pct={self.predicted_change_pct})>"


class StockPredictionPoint(Base):
    """
    Per-day forecast prices for the Price & Forecast chart overlay.
    """

    __tablename__ = "stock_prediction_points"
    __table_args__ = (
        UniqueConstraint(
            "stock_id",
            "experiment_run_id",
            "horizon_days",
            "prediction_date",
            name="uq_pred_point",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(
        BigInteger,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
    )
    experiment_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiment_runs.id"),
        nullable=True,
    )
    horizon_days = Column(Integer, nullable=False)
    prediction_date = Column(Date, nullable=False)
    predicted_price = Column(Numeric(18, 4), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    stock = relationship("Stock", back_populates="prediction_points")

    def __repr__(self) -> str:
        return f"<StockPredictionPoint(stock_id={self.stock_id}, date={self.prediction_date}, price={self.predicted_price})>"


class FreshnessState(str, Enum):
    """Model freshness state enumeration."""

    FRESH = "fresh"
    STABLE = "stable"
    STALE = "stale"


class ModelStatus(Base):
    """
    Latest model state per stock (fresh/stable/stale).
    """

    __tablename__ = "model_statuses"
    __table_args__ = (Index("ux_model_status_latest", "stock_id", "last_updated_at"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(
        BigInteger,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
    )
    experiment_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiment_runs.id"),
        nullable=True,
    )
    freshness_state = Column(
        String(20),
        nullable=False,
        default=FreshnessState.FRESH.value,
    )
    last_updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    stock = relationship("Stock", back_populates="model_statuses")
    horizon_metrics = relationship(
        "ModelHorizonMetric",
        back_populates="model_status",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ModelStatus(stock_id={self.stock_id}, state={self.freshness_state})>"


class ModelHorizonMetric(Base):
    """
    Per-model, per-horizon metrics (e.g., MAPE).
    """

    __tablename__ = "model_horizon_metrics"
    __table_args__ = (
        UniqueConstraint("model_status_id", "horizon_days", name="uq_metric_horizon"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_status_id = Column(
        BigInteger,
        ForeignKey("model_statuses.id", ondelete="CASCADE"),
        nullable=False,
    )
    horizon_days = Column(Integer, nullable=False)
    mape_pct = Column(Numeric(6, 3), nullable=False)

    # Relationships
    model_status = relationship("ModelStatus", back_populates="horizon_metrics")

    def __repr__(self) -> str:
        return f"<ModelHorizonMetric(model_status_id={self.model_status_id}, horizon={self.horizon_days}d, mape={self.mape_pct})>"


class CrawlMetadata(Base):
    """
    Tracks last crawl dates for incremental updates.
    Legacy table for backward compatibility.
    """

    __tablename__ = "crawl_metadata"

    stock_symbol = Column(String(10), primary_key=True)
    last_crawl_date = Column(Date, nullable=True)
    last_data_date = Column(Date, nullable=True)
    total_records = Column(Integer, default=0)
    last_updated = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<CrawlMetadata(symbol={self.stock_symbol}, last_data={self.last_data_date})>"


class UserSavedStock(Base, TimestampMixin):
    """
    User's saved stocks (watchlist/my list).
    Maps to SPECS.md requirement for "My List" feature.
    """

    __tablename__ = "user_saved_stocks"
    __table_args__ = (
        UniqueConstraint("user_id", "stock_id", name="uq_user_saved_stock"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stock_id = Column(
        BigInteger,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Relationships
    user = relationship("User", back_populates="saved_stocks")
    stock = relationship("Stock", back_populates="saved_by_users")

    def __repr__(self) -> str:
        return f"<UserSavedStock(user_id={self.user_id}, stock_id={self.stock_id})>"
