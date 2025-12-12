"""
User saved stocks service.
Provides methods for managing user's saved stocks (My List).
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import Stock, StockPredictionSummary, UserSavedStock
from app.services.predictions import get_current_price
from app.services.stock_service import get_stock_by_ticker

logger = get_logger(__name__)


def get_user_saved_stocks(
    db: Session,
    user_id: str,
    limit: int = 5,
    horizon_days: int = 7,
) -> list[dict[str, Any]]:
    """
    Get user's saved stocks with predictions.
    
    Args:
        db: Database session
        user_id: User UUID (string)
        limit: Number of stocks to return (default 5)
        horizon_days: Prediction horizon in days (default 7)
        
    Returns:
        List of saved stocks with ticker, name, sector, predicted change, current price, added_at
    """
    from sqlalchemy import func, and_
    
    # Get latest date for predictions
    latest_date_stmt = (
        select(func.max(StockPredictionSummary.as_of_date))
        .where(StockPredictionSummary.horizon_days == horizon_days)
    )
    latest_date = db.execute(latest_date_stmt).scalar()
    
    # Query user's saved stocks with predictions
    join_conditions = [
        Stock.id == StockPredictionSummary.stock_id,
        StockPredictionSummary.horizon_days == horizon_days,
    ]
    if latest_date:
        join_conditions.append(StockPredictionSummary.as_of_date == latest_date)
    
    stmt = (
        select(
            Stock.ticker,
            Stock.name,
            Stock.sector,
            StockPredictionSummary.predicted_change_pct,
            UserSavedStock.created_at.label("added_at"),
        )
        .join(UserSavedStock, Stock.id == UserSavedStock.stock_id)
        .outerjoin(
            StockPredictionSummary,
            and_(*join_conditions) if join_conditions else False,
        )
        .where(
            UserSavedStock.user_id == user_id,
            Stock.is_active == True,  # noqa: E712
        )
        .order_by(UserSavedStock.created_at.desc())
        .limit(limit)
    )
    
    results = db.execute(stmt).all()
    
    # Get current prices and format response
    saved_stocks = []
    for row in results:
        current_price = get_current_price(db, row.ticker)
        saved_stocks.append({
            "ticker": row.ticker,
            "name": row.name,
            "sector": row.sector,
            "horizonDays": horizon_days,
            "predictedChangePct": float(row.predicted_change_pct) if row.predicted_change_pct else 0.0,
            "currentPrice": current_price,
            "addedAt": row.added_at.isoformat() if row.added_at else datetime.now().isoformat(),
        })
    
    return saved_stocks


def add_stock_to_list(
    db: Session,
    user_id: str,
    stock_id: int,
) -> UserSavedStock:
    """
    Add a stock to user's saved list.
    
    Args:
        db: Database session
        user_id: User UUID
        stock_id: Stock ID
        
    Returns:
        UserSavedStock instance
        
    Raises:
        ValueError: If stock is already in user's list
    """
    # Check if already exists
    existing = db.execute(
        select(UserSavedStock).where(
            UserSavedStock.user_id == user_id,
            UserSavedStock.stock_id == stock_id,
        )
    ).scalar_one_or_none()
    
    if existing:
        raise ValueError("Stock is already in user's saved list")
    
    saved_stock = UserSavedStock(
        user_id=user_id,
        stock_id=stock_id,
    )
    db.add(saved_stock)
    db.commit()
    db.refresh(saved_stock)
    
    return saved_stock


def remove_stock_from_list(
    db: Session,
    user_id: str,
    stock_id: int,
) -> bool:
    """
    Remove a stock from user's saved list.
    
    Args:
        db: Database session
        user_id: User UUID
        stock_id: Stock ID
        
    Returns:
        True if removed, False if not found
    """
    saved_stock = db.execute(
        select(UserSavedStock).where(
            UserSavedStock.user_id == user_id,
            UserSavedStock.stock_id == stock_id,
        )
    ).scalar_one_or_none()
    
    if not saved_stock:
        return False
    
    db.delete(saved_stock)
    db.commit()
    
    return True

