"""
Prediction API endpoints.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import ChartDataResponse, ModelStatusResponse, PredictionResponse
from app.services import get_chart_data, get_model_status, get_stock_predictions

router = APIRouter(tags=["Predictions"])


@router.get("/stocks/{ticker}/predictions", response_model=PredictionResponse)
async def get_predictions_for_stock(
    ticker: str,
    horizons: str = Query("3,7,15,30"),
    db: Session = Depends(get_db),
):
    """
    Get predicted % change for specified horizons.
    
    - **ticker**: Stock ticker symbol
    - **horizons**: Comma-separated list of horizon days (e.g., "3,7,15,30")
    """
    horizon_list = [int(h.strip()) for h in horizons.split(",")]
    result = get_stock_predictions(db, ticker, horizons=horizon_list)
    return PredictionResponse(**result)


@router.get("/stocks/{ticker}/chart", response_model=ChartDataResponse)
async def get_chart_for_stock(
    ticker: str,
    range_param: str = Query("7d", alias="range", regex="^(3d|7d|15d|30d)$"),
    db: Session = Depends(get_db),
):
    """
    Get historical and predicted price data for chart.
    
    - **ticker**: Stock ticker symbol
    - **range**: Chart range (3d, 7d, 15d, 30d)
    """
    result = get_chart_data(db, ticker, range_param=range_param)
    return ChartDataResponse(**result)


@router.get("/models/{ticker}/status", response_model=ModelStatusResponse)
async def get_model_status_for_ticker(
    ticker: str,
    db: Session = Depends(get_db),
):
    """
    Get model status for a stock (freshness state, last updated, MAPE per horizon).
    
    - **ticker**: Stock ticker symbol
    """
    result = get_model_status(db, ticker)
    return ModelStatusResponse(**result)

