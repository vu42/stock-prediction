"""
Prediction query service.
Provides methods for top-picks, market-table, chart data, and model status.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import (
    ModelHorizonMetric,
    ModelStatus,
    Stock,
    StockPredictionPoint,
    StockPredictionSummary,
    StockPrice,
)

logger = get_logger(__name__)


def get_top_picks(
    db: Session,
    bucket: str = "should_buy",
    limit: int = 5,
    horizon_days: int = 7,
) -> list[dict[str, Any]]:
    """
    Get top stock picks for "Should Buy" or "Should Sell" tabs.
    
    Args:
        db: Database session
        bucket: "should_buy" (highest positive predicted %) or "should_sell" (highest negative predicted %)
        limit: Number of stocks to return
        horizon_days: Prediction horizon in days
        
    Returns:
        List of top picks with ticker, name, sector, predicted change, current price
    """
    # Get latest predictions for the horizon
    latest_date_subq = (
        select(func.max(StockPredictionSummary.as_of_date))
        .where(StockPredictionSummary.horizon_days == horizon_days)
        .scalar_subquery()
    )
    
    # Order by predicted_change_pct (desc for buy, asc for sell)
    order_clause = (
        StockPredictionSummary.predicted_change_pct.desc()
        if bucket == "should_buy"
        else StockPredictionSummary.predicted_change_pct.asc()
    )
    
    # Additional filter: positive for buy, negative for sell
    change_filter = (
        StockPredictionSummary.predicted_change_pct > 0
        if bucket == "should_buy"
        else StockPredictionSummary.predicted_change_pct < 0
    )
    
    stmt = (
        select(
            Stock.ticker,
            Stock.name,
            Stock.sector,
            StockPredictionSummary.horizon_days,
            StockPredictionSummary.predicted_change_pct,
        )
        .join(StockPredictionSummary, Stock.id == StockPredictionSummary.stock_id)
        .where(
            and_(
                StockPredictionSummary.as_of_date == latest_date_subq,
                StockPredictionSummary.horizon_days == horizon_days,
                change_filter,
                Stock.is_active == True,  # noqa: E712
            )
        )
        .order_by(order_clause)
        .limit(limit)
    )
    
    results = db.execute(stmt).all()
    
    # Get current prices for these stocks
    picks = []
    for row in results:
        current_price = get_current_price(db, row.ticker)
        picks.append({
            "ticker": row.ticker,
            "name": row.name,
            "sector": row.sector,
            "horizonDays": row.horizon_days,
            "predictedChangePct": float(row.predicted_change_pct),
            "currentPrice": current_price,
        })
    
    return picks


def get_current_price(db: Session, ticker: str) -> float | None:
    """
    Get the most recent closing price for a stock.
    
    Args:
        db: Database session
        ticker: Stock ticker symbol
        
    Returns:
        Current (most recent) closing price or None
    """
    stmt = (
        select(StockPrice.close_price)
        .join(Stock, StockPrice.stock_id == Stock.id)
        .where(Stock.ticker == ticker)
        .order_by(StockPrice.price_date.desc())
        .limit(1)
    )
    result = db.execute(stmt).scalar_one_or_none()
    return float(result) if result else None


def get_market_table_data(
    db: Session,
    search: str | None = None,
    sector: str | None = None,
    sort_by: str = "change_7d",
    sort_dir: str = "desc",
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """
    Get market table data with search, filter, sort, and pagination.
    
    Args:
        db: Database session
        search: Search string for ticker/name
        sector: Filter by sector
        sort_by: Sort column (change_1d, change_3d, change_7d, price)
        sort_dir: Sort direction (asc, desc)
        page: Page number
        page_size: Items per page
        
    Returns:
        Dict with data array and meta (total, page, pageSize, sectors)
    """
    # Base query for stocks
    base_query = select(Stock).where(Stock.is_active == True)  # noqa: E712
    
    if search:
        search_filter = Stock.ticker.ilike(f"%{search}%") | Stock.name.ilike(f"%{search}%")
        base_query = base_query.where(search_filter)
    
    if sector:
        base_query = base_query.where(Stock.sector == sector)
    
    # Get total count
    count_stmt = select(func.count()).select_from(base_query.subquery())
    total = db.execute(count_stmt).scalar() or 0
    
    # Get distinct sectors for filter options
    sectors_stmt = select(Stock.sector).where(Stock.sector.isnot(None)).distinct()
    sectors = [r[0] for r in db.execute(sectors_stmt).all()]
    
    # Pagination
    offset = (page - 1) * page_size
    stocks_stmt = base_query.offset(offset).limit(page_size)
    stocks = db.execute(stocks_stmt).scalars().all()
    
    # Build data with predictions
    data = []
    for stock in stocks:
        current_price = get_current_price(db, stock.ticker)
        pct_change = get_pct_changes(db, stock.id)
        sparkline = get_sparkline_data(db, stock.id, days=7)
        
        data.append({
            "symbol": stock.ticker,
            "name": stock.name,
            "sector": stock.sector,
            "currentPrice": current_price,
            "pctChange": pct_change,
            "sparkline7d": sparkline,
        })
    
    # Sort (in-memory for now, could optimize with DB sorting)
    sort_key_map = {
        "change_1d": lambda x: x["pctChange"]["1d"]["predictedPct"] or 0,
        "change_3d": lambda x: x["pctChange"]["3d"]["predictedPct"] or 0,
        "change_7d": lambda x: x["pctChange"]["7d"]["predictedPct"] or 0,
        "price": lambda x: x["currentPrice"] or 0,
    }
    if sort_by in sort_key_map:
        data.sort(key=sort_key_map[sort_by], reverse=(sort_dir == "desc"))
    
    return {
        "data": data,
        "meta": {
            "total": total,
            "page": page,
            "pageSize": page_size,
            "sectors": sectors,
        },
    }


def get_pct_changes(db: Session, stock_id: int) -> dict[str, dict[str, float | None]]:
    """
    Get actual and predicted % changes for 1d, 3d, 7d horizons.
    
    Args:
        db: Database session
        stock_id: Stock ID
        
    Returns:
        Dict with horizon keys and actualPct/predictedPct values
    """
    result = {
        "1d": {"actualPct": None, "predictedPct": None},
        "3d": {"actualPct": None, "predictedPct": None},
        "7d": {"actualPct": None, "predictedPct": None},
    }
    
    # Get recent prices for actual % calculation
    prices_stmt = (
        select(StockPrice.price_date, StockPrice.close_price)
        .where(StockPrice.stock_id == stock_id)
        .order_by(StockPrice.price_date.desc())
        .limit(8)  # Need up to 7 days back + today
    )
    prices = db.execute(prices_stmt).all()
    
    if len(prices) >= 2:
        current = float(prices[0].close_price)
        
        # Calculate actual % changes
        for horizon, key in [(1, "1d"), (3, "3d"), (7, "7d")]:
            if len(prices) > horizon:
                past = float(prices[horizon].close_price)
                result[key]["actualPct"] = round((current - past) / past * 100, 2)
    
    # Get predicted % from summaries
    for horizon_days, key in [(1, "1d"), (3, "3d"), (7, "7d")]:
        pred_stmt = (
            select(StockPredictionSummary.predicted_change_pct)
            .where(
                and_(
                    StockPredictionSummary.stock_id == stock_id,
                    StockPredictionSummary.horizon_days == horizon_days,
                )
            )
            .order_by(StockPredictionSummary.as_of_date.desc())
            .limit(1)
        )
        pred = db.execute(pred_stmt).scalar_one_or_none()
        if pred:
            result[key]["predictedPct"] = float(pred)
    
    return result


def get_sparkline_data(
    db: Session,
    stock_id: int,
    days: int = 7,
) -> list[dict[str, Any]]:
    """
    Get sparkline price data for the last N days.
    
    Args:
        db: Database session
        stock_id: Stock ID
        days: Number of days
        
    Returns:
        List of {date, price} dicts
    """
    stmt = (
        select(StockPrice.price_date, StockPrice.close_price)
        .where(StockPrice.stock_id == stock_id)
        .order_by(StockPrice.price_date.desc())
        .limit(days)
    )
    results = db.execute(stmt).all()
    
    # Reverse to chronological order
    return [
        {"date": r.price_date.isoformat(), "price": float(r.close_price)}
        for r in reversed(results)
    ]


def get_stock_predictions(
    db: Session,
    ticker: str,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    """
    Get predicted % change for specified horizons.
    
    Args:
        db: Database session
        ticker: Stock ticker
        horizons: List of horizon days (default: [3, 7, 15, 30])
        
    Returns:
        Dict with ticker and horizons dict
    """
    if horizons is None:
        horizons = [3, 7, 15, 30]
    
    # Get stock
    stock_stmt = select(Stock).where(Stock.ticker == ticker)
    stock = db.execute(stock_stmt).scalar_one_or_none()
    
    if not stock:
        return {"ticker": ticker, "horizons": {}}
    
    result = {"ticker": ticker, "horizons": {}}
    
    for horizon in horizons:
        pred_stmt = (
            select(StockPredictionSummary.predicted_change_pct)
            .where(
                and_(
                    StockPredictionSummary.stock_id == stock.id,
                    StockPredictionSummary.horizon_days == horizon,
                )
            )
            .order_by(StockPredictionSummary.as_of_date.desc())
            .limit(1)
        )
        pred = db.execute(pred_stmt).scalar_one_or_none()
        result["horizons"][str(horizon)] = {
            "predictedChangePct": float(pred) if pred else None
        }
    
    return result


def get_chart_data(
    db: Session,
    ticker: str,
    range_param: str = "7d",
) -> dict[str, Any]:
    """
    Get historical and predicted price data for chart.
    
    Args:
        db: Database session
        ticker: Stock ticker
        range_param: Range (3d, 7d, 15d, 30d)
        
    Returns:
        Dict with points array and range
    """
    # Parse range
    days_map = {"3d": 3, "7d": 7, "15d": 15, "30d": 30}
    days = days_map.get(range_param, 7)
    
    # Get stock
    stock_stmt = select(Stock).where(Stock.ticker == ticker)
    stock = db.execute(stock_stmt).scalar_one_or_none()
    
    if not stock:
        return {"points": [], "range": range_param}
    
    # Get historical prices
    price_stmt = (
        select(StockPrice.price_date, StockPrice.close_price)
        .where(StockPrice.stock_id == stock.id)
        .order_by(StockPrice.price_date.desc())
        .limit(days * 2)  # Get more for context
    )
    prices = db.execute(price_stmt).all()
    
    # Get prediction points
    pred_stmt = (
        select(StockPredictionPoint.prediction_date, StockPredictionPoint.predicted_price)
        .where(
            and_(
                StockPredictionPoint.stock_id == stock.id,
                StockPredictionPoint.horizon_days == days,
            )
        )
        .order_by(StockPredictionPoint.prediction_date.desc())
        .limit(days)
    )
    predictions = db.execute(pred_stmt).all()
    
    # Build points array
    points = []
    pred_dict = {p.prediction_date: float(p.predicted_price) for p in predictions}
    
    for p in reversed(prices):
        points.append({
            "date": p.price_date.isoformat(),
            "actualPrice": float(p.close_price),
            "predictedPrice": pred_dict.get(p.price_date),
        })
    
    return {"points": points[-days:], "range": range_param}


def get_model_status(db: Session, ticker: str) -> dict[str, Any]:
    """
    Get model status for a stock (freshness state, last updated, MAPE per horizon).
    
    Args:
        db: Database session
        ticker: Stock ticker
        
    Returns:
        Dict with state, lastUpdatedAt, metrics
    """
    # Get stock
    stock_stmt = select(Stock).where(Stock.ticker == ticker)
    stock = db.execute(stock_stmt).scalar_one_or_none()
    
    if not stock:
        return {
            "state": "stale",
            "lastUpdatedAt": None,
            "metrics": {},
        }
    
    # Get latest model status
    status_stmt = (
        select(ModelStatus)
        .where(ModelStatus.stock_id == stock.id)
        .order_by(ModelStatus.last_updated_at.desc())
        .limit(1)
    )
    status = db.execute(status_stmt).scalar_one_or_none()
    
    if not status:
        return {
            "state": "stale",
            "lastUpdatedAt": None,
            "metrics": {},
        }
    
    # Get horizon metrics
    metrics_stmt = (
        select(ModelHorizonMetric)
        .where(ModelHorizonMetric.model_status_id == status.id)
    )
    metrics = db.execute(metrics_stmt).scalars().all()
    
    metrics_dict = {}
    for m in metrics:
        metrics_dict[f"{m.horizon_days}d"] = {"mapePct": float(m.mape_pct)}
    
    return {
        "state": status.freshness_state,
        "lastUpdatedAt": status.last_updated_at.isoformat() if status.last_updated_at else None,
        "metrics": metrics_dict,
    }


def get_models_overview(db: Session) -> list[dict[str, Any]]:
    """
    Get models overview for all active stocks.
    
    Returns data for the Models page table including:
    - ticker
    - lastTrained (from ModelStatus.last_updated_at)
    - mape for 7d, 15d, 30d horizons
    - predictions for 7d, 15d, 30d horizons
    - plotUrl from latest experiment artifact
    
    Args:
        db: Database session
        
    Returns:
        List of model overview dicts
    """
    from app.db.models import ExperimentTickerArtifact
    
    # Get all active stocks
    stocks_stmt = select(Stock).where(Stock.is_active == True)  # noqa: E712
    stocks = db.execute(stocks_stmt).scalars().all()
    
    results = []
    horizons = [7, 15, 30]
    
    for stock in stocks:
        # Get latest model status for lastTrained
        status_stmt = (
            select(ModelStatus)
            .where(ModelStatus.stock_id == stock.id)
            .order_by(ModelStatus.last_updated_at.desc())
            .limit(1)
        )
        status = db.execute(status_stmt).scalar_one_or_none()
        
        last_trained = None
        mape_dict: dict[str, float | None] = {"7d": None, "15d": None, "30d": None}
        
        if status:
            last_trained = status.last_updated_at
            
            # Get horizon metrics for MAPE
            metrics_stmt = (
                select(ModelHorizonMetric)
                .where(ModelHorizonMetric.model_status_id == status.id)
            )
            metrics = db.execute(metrics_stmt).scalars().all()
            
            for m in metrics:
                key = f"{m.horizon_days}d"
                if key in mape_dict:
                    mape_dict[key] = float(m.mape_pct)
        
        # Get predictions for each horizon
        predictions_dict: dict[str, float | None] = {"7d": None, "15d": None, "30d": None}
        for horizon in horizons:
            pred_stmt = (
                select(StockPredictionSummary.predicted_change_pct)
                .where(
                    and_(
                        StockPredictionSummary.stock_id == stock.id,
                        StockPredictionSummary.horizon_days == horizon,
                    )
                )
                .order_by(StockPredictionSummary.as_of_date.desc())
                .limit(1)
            )
            pred = db.execute(pred_stmt).scalar_one_or_none()
            if pred:
                predictions_dict[f"{horizon}d"] = float(pred)
        
        # Get latest artifact for plotUrl
        plot_url = None
        artifact_stmt = (
            select(ExperimentTickerArtifact.evaluation_png_url)
            .where(ExperimentTickerArtifact.stock_id == stock.id)
            .order_by(ExperimentTickerArtifact.created_at.desc())
            .limit(1)
        )
        artifact_url = db.execute(artifact_stmt).scalar_one_or_none()
        if artifact_url:
            plot_url = artifact_url
        
        results.append({
            "ticker": stock.ticker,
            "lastTrained": last_trained.isoformat() if last_trained else None,
            "mape": mape_dict,
            "predictions": predictions_dict,
            "plotUrl": plot_url,
        })
    
    return results

