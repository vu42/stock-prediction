"""
Stock API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.security import CurrentUser
from app.db.session import get_db
from app.services.stock_service import get_stock_by_ticker
from app.schemas import (
    MarketTableResponse,
    MyListResponse,
    StockCreate,
    StockResponse,
    StockUpdate,
    TopPickResponse,
)
from app.services import (
    add_stock_to_list,
    create_stock,
    get_all_stocks,
    get_market_table_data,
    get_stock_detail,
    get_top_picks,
    get_user_saved_stocks,
    remove_stock_from_list,
    update_stock,
)

router = APIRouter(prefix="/stocks", tags=["Stocks"])


@router.get("/top-picks", response_model=list[TopPickResponse])
async def get_top_stock_picks(
    bucket: str = Query("should_buy", regex="^(should_buy|should_sell)$"),
    limit: int = Query(5, ge=1, le=20),
    horizon_days: int = Query(7, alias="horizonDays"),
    db: Session = Depends(get_db),
):
    """
    Get top stock picks for "Should Buy" or "Should Sell" tabs.
    
    - **bucket**: "should_buy" (highest positive predicted %) or "should_sell" (highest negative predicted %)
    - **limit**: Number of stocks to return (default 5, max 20)
    - **horizonDays**: Prediction horizon in days (default 7)
    """
    picks = get_top_picks(db, bucket=bucket, limit=limit, horizon_days=horizon_days)
    return [TopPickResponse(**pick) for pick in picks]


@router.get("/my-list", response_model=list[MyListResponse])
async def get_my_list(
    current_user: CurrentUser,
    limit: int = Query(5, ge=1, le=20),
    horizon_days: int = Query(7, alias="horizonDays"),
    db: Session = Depends(get_db),
):
    """
    Get current user's saved stocks (My List).
    
    - **limit**: Number of stocks to return (default 5, max 20)
    - **horizonDays**: Prediction horizon in days (default 7)
    - Requires authentication
    """
    saved_stocks = get_user_saved_stocks(
        db,
        user_id=str(current_user.id),
        limit=limit,
        horizon_days=horizon_days,
    )
    return [MyListResponse(**stock) for stock in saved_stocks]


@router.post("/my-list/{ticker}", status_code=status.HTTP_201_CREATED)
async def add_stock_to_my_list(
    ticker: str,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Add a stock to current user's saved list (My List).
    
    - **ticker**: Stock ticker symbol (e.g., "FPT", "VCB")
    - Requires authentication
    
    Returns:
        Success message with stock details
    """
    try:
        # Get stock by ticker
        stock = get_stock_by_ticker(db, ticker)
        
        # Add to user's list
        add_stock_to_list(
            db,
            user_id=str(current_user.id),
            stock_id=stock.id,
        )
        
        return {
            "message": f"Stock {ticker} added to your list",
            "ticker": stock.ticker,
            "name": stock.name,
        }
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e.message),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.delete("/my-list/{ticker}", status_code=status.HTTP_200_OK)
async def remove_stock_from_my_list(
    ticker: str,
    current_user: CurrentUser,
    db: Session = Depends(get_db),
):
    """
    Remove a stock from current user's saved list (My List).
    
    - **ticker**: Stock ticker symbol (e.g., "FPT", "VCB")
    - Requires authentication
    
    Returns:
        Success message
    """
    try:
        # Get stock by ticker
        stock = get_stock_by_ticker(db, ticker)
        
        # Remove from user's list
        removed = remove_stock_from_list(
            db,
            user_id=str(current_user.id),
            stock_id=stock.id,
        )
        
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock {ticker} is not in your list",
            )
        
        return {
            "message": f"Stock {ticker} removed from your list",
        }
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e.message),
        )


@router.get("/market-table", response_model=MarketTableResponse)
async def get_market_table(
    search: str | None = Query(None),
    sector: str | None = Query(None),
    sort_by: str = Query("change_7d", alias="sortBy"),
    sort_dir: str = Query("desc", regex="^(asc|desc)$", alias="sortDir"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
    db: Session = Depends(get_db),
):
    """
    Get market table data with search, filter, sort, and pagination.
    
    - **search**: Search string for ticker/name
    - **sector**: Filter by sector
    - **sortBy**: Sort column (change_7d, change_15d, change_30d, price)
    - **sortDir**: Sort direction (asc, desc)
    - **page**: Page number
    - **pageSize**: Items per page
    """
    result = get_market_table_data(
        db,
        search=search,
        sector=sector,
        sort_by=sort_by,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )
    return MarketTableResponse(**result)


@router.get("/{ticker}", response_model=StockResponse)
async def get_stock(
    ticker: str,
    db: Session = Depends(get_db),
):
    """
    Get detailed stock information by ticker symbol.
    """
    detail = get_stock_detail(db, ticker)
    return StockResponse(
        ticker=detail["ticker"],
        name=detail["name"],
        logoUrl=detail["logoUrl"],
        description=detail["description"],
        sector=detail["sector"],
        exchange=detail["exchange"],
        marketCap=detail["marketCap"],
        tradingVolume=detail["tradingVolume"],
        links=detail["links"],
    )


@router.get("", response_model=list[StockResponse])
async def list_stocks(
    active_only: bool = Query(True, alias="activeOnly"),
    db: Session = Depends(get_db),
):
    """
    List all stocks.
    """
    stocks = get_all_stocks(db, active_only=active_only)
    return [
        StockResponse(
            ticker=s.ticker,
            name=s.name,
            sector=s.sector,
            exchange=s.exchange,
            description=s.description,
            logoUrl=s.logo_url,
        )
        for s in stocks
    ]


@router.post("", response_model=StockResponse, status_code=201)
async def create_new_stock(
    request: StockCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new stock record.
    """
    stock = create_stock(
        db,
        ticker=request.ticker,
        name=request.name,
        sector=request.sector,
        exchange=request.exchange,
        description=request.description,
        logo_url=request.logo_url,
        financial_report_url=request.financial_report_url,
        company_website_url=request.company_website_url,
    )
    return StockResponse(
        ticker=stock.ticker,
        name=stock.name,
        sector=stock.sector,
        exchange=stock.exchange,
        description=stock.description,
        logoUrl=stock.logo_url,
    )


@router.patch("/{ticker}", response_model=StockResponse)
async def update_existing_stock(
    ticker: str,
    request: StockUpdate,
    db: Session = Depends(get_db),
):
    """
    Update an existing stock record.
    """
    stock = update_stock(
        db,
        ticker=ticker,
        name=request.name,
        sector=request.sector,
        exchange=request.exchange,
        description=request.description,
        logo_url=request.logo_url,
        financial_report_url=request.financial_report_url,
        company_website_url=request.company_website_url,
        is_active=request.is_active,
    )
    return StockResponse(
        ticker=stock.ticker,
        name=stock.name,
        sector=stock.sector,
        exchange=stock.exchange,
        description=stock.description,
        logoUrl=stock.logo_url,
    )

