"""
Stock service - CRUD operations and business logic for stocks.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.models import Stock

logger = get_logger(__name__)


def get_stock_by_ticker(db: Session, ticker: str) -> Stock:
    """
    Get a stock by ticker symbol.
    
    Args:
        db: Database session
        ticker: Stock ticker symbol
        
    Returns:
        Stock model instance
        
    Raises:
        NotFoundError: If stock not found
    """
    stmt = select(Stock).where(Stock.ticker == ticker.upper())
    stock = db.execute(stmt).scalar_one_or_none()
    
    if not stock:
        raise NotFoundError("Stock", ticker)
    
    return stock


def get_stock_by_id(db: Session, stock_id: int) -> Stock:
    """
    Get a stock by ID.
    
    Args:
        db: Database session
        stock_id: Stock ID
        
    Returns:
        Stock model instance
        
    Raises:
        NotFoundError: If stock not found
    """
    stmt = select(Stock).where(Stock.id == stock_id)
    stock = db.execute(stmt).scalar_one_or_none()
    
    if not stock:
        raise NotFoundError("Stock", stock_id)
    
    return stock


def get_all_stocks(
    db: Session,
    active_only: bool = True,
) -> list[Stock]:
    """
    Get all stocks.
    
    Args:
        db: Database session
        active_only: If True, return only active stocks
        
    Returns:
        List of Stock model instances
    """
    stmt = select(Stock)
    if active_only:
        stmt = stmt.where(Stock.is_active == True)  # noqa: E712
    stmt = stmt.order_by(Stock.ticker)
    
    return list(db.execute(stmt).scalars().all())


def create_stock(
    db: Session,
    ticker: str,
    name: str,
    sector: str | None = None,
    exchange: str | None = None,
    description: str | None = None,
    logo_url: str | None = None,
    financial_report_url: str | None = None,
    company_website_url: str | None = None,
) -> Stock:
    """
    Create a new stock.
    
    Args:
        db: Database session
        ticker: Stock ticker symbol
        name: Company name
        sector: Business sector
        exchange: Stock exchange
        description: Company description
        logo_url: URL to company logo
        financial_report_url: URL to financial reports
        company_website_url: Company website URL
        
    Returns:
        Created Stock model instance
    """
    stock = Stock(
        ticker=ticker.upper(),
        name=name,
        sector=sector,
        exchange=exchange,
        description=description,
        logo_url=logo_url,
        financial_report_url=financial_report_url,
        company_website_url=company_website_url,
        is_active=True,
    )
    db.add(stock)
    db.commit()
    db.refresh(stock)
    
    logger.info(f"Created stock: {ticker}")
    return stock


def update_stock(
    db: Session,
    ticker: str,
    **kwargs: Any,
) -> Stock:
    """
    Update a stock.
    
    Args:
        db: Database session
        ticker: Stock ticker symbol
        **kwargs: Fields to update
        
    Returns:
        Updated Stock model instance
        
    Raises:
        NotFoundError: If stock not found
    """
    stock = get_stock_by_ticker(db, ticker)
    
    allowed_fields = {
        "name", "sector", "exchange", "description",
        "logo_url", "financial_report_url", "company_website_url", "is_active",
    }
    
    for key, value in kwargs.items():
        if key in allowed_fields and value is not None:
            setattr(stock, key, value)
    
    db.commit()
    db.refresh(stock)
    
    logger.info(f"Updated stock: {ticker}")
    return stock


def delete_stock(db: Session, ticker: str) -> None:
    """
    Delete a stock (soft delete by setting is_active=False).
    
    Args:
        db: Database session
        ticker: Stock ticker symbol
        
    Raises:
        NotFoundError: If stock not found
    """
    stock = get_stock_by_ticker(db, ticker)
    stock.is_active = False
    db.commit()
    
    logger.info(f"Soft deleted stock: {ticker}")


def get_stock_detail(db: Session, ticker: str) -> dict[str, Any]:
    """
    Get detailed stock information for the Stock Detail page.
    
    Args:
        db: Database session
        ticker: Stock ticker symbol
        
    Returns:
        Dict with stock details
        
    Raises:
        NotFoundError: If stock not found
    """
    stock = get_stock_by_ticker(db, ticker)
    
    return {
        "ticker": stock.ticker,
        "name": stock.name,
        "logoUrl": stock.logo_url,
        "description": stock.description,
        "sector": stock.sector,
        "exchange": stock.exchange,
        "marketCap": None,  # Would need external data source
        "tradingVolume": None,  # Would calculate from recent prices
        "links": {
            "financialReportUrl": stock.financial_report_url,
            "companyWebsiteUrl": stock.company_website_url,
        },
    }


def seed_vn30_stocks(db: Session) -> int:
    """
    Seed the database with VN30 stock records.
    
    Args:
        db: Database session
        
    Returns:
        Number of stocks created
    """
    from app.core.config import settings
    
    created_count = 0
    for ticker in settings.vn30_stocks:
        # Check if already exists
        stmt = select(Stock).where(Stock.ticker == ticker)
        existing = db.execute(stmt).scalar_one_or_none()
        
        if not existing:
            stock = Stock(
                ticker=ticker,
                name=ticker,  # Placeholder, should be updated with real names
                is_active=True,
            )
            db.add(stock)
            created_count += 1
    
    db.commit()
    logger.info(f"Seeded {created_count} VN30 stocks")
    return created_count

