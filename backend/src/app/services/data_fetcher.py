"""
Data fetching service - Handles API crawling with incremental updates.
Migrated from modules/data_fetcher.py to use SQLAlchemy.
"""

import json
import ssl
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import CrawlMetadata, Stock, StockPrice

logger = get_logger(__name__)

# SSL configuration
ssl._create_default_https_context = ssl._create_unverified_context


def get_last_data_date(db: Session, stock_symbol: str) -> str | None:
    """
    Get the last date available for a stock in database.
    Used for incremental crawling.
    
    Args:
        db: Database session
        stock_symbol: Stock ticker symbol
        
    Returns:
        Last date in 'YYYY-MM-DD' format, or None if no data exists
    """
    # First try to get from stock_prices via Stock
    stmt = (
        select(Stock)
        .where(Stock.ticker == stock_symbol)
    )
    stock = db.execute(stmt).scalar_one_or_none()
    
    if stock:
        price_stmt = (
            select(StockPrice.price_date)
            .where(StockPrice.stock_id == stock.id)
            .order_by(StockPrice.price_date.desc())
            .limit(1)
        )
        result = db.execute(price_stmt).scalar_one_or_none()
        if result:
            return result.strftime("%Y-%m-%d")
    
    # Fallback to crawl_metadata for backward compatibility
    meta_stmt = select(CrawlMetadata).where(CrawlMetadata.stock_symbol == stock_symbol)
    meta = db.execute(meta_stmt).scalar_one_or_none()
    if meta and meta.last_data_date:
        return meta.last_data_date.strftime("%Y-%m-%d")
    
    return None


def get_or_create_stock(db: Session, stock_symbol: str) -> Stock:
    """
    Get or create a stock record.
    
    Args:
        db: Database session
        stock_symbol: Stock ticker symbol
        
    Returns:
        Stock model instance
    """
    stmt = select(Stock).where(Stock.ticker == stock_symbol)
    stock = db.execute(stmt).scalar_one_or_none()
    
    if not stock:
        stock = Stock(
            ticker=stock_symbol,
            name=stock_symbol,  # Will be updated with proper name later
            is_active=True,
        )
        db.add(stock)
        db.flush()
        logger.info(f"[{stock_symbol}] Created new stock record")
    
    return stock


def insert_stock_data(
    db: Session,
    df: pd.DataFrame,
    stock_symbol: str,
) -> int:
    """
    Insert stock data into database with UPSERT (update on conflict).
    Also updates crawl_metadata table.
    
    Args:
        db: Database session
        df: DataFrame with stock data (columns: date, open, high, low, close, volume)
        stock_symbol: Stock symbol
        
    Returns:
        Number of records inserted/updated
    """
    if df.empty:
        return 0
    
    # Get or create stock record
    stock = get_or_create_stock(db, stock_symbol)
    
    # Prepare records for batch insert
    records = []
    for _, row in df.iterrows():
        date_val = pd.to_datetime(row.get("date") or row.get("Date")).date()
        records.append({
            "stock_id": stock.id,
            "price_date": date_val,
            "open_price": float(row.get("open") or row.get("Open", 0)),
            "high_price": float(row.get("high") or row.get("High", 0)),
            "low_price": float(row.get("low") or row.get("Low", 0)),
            "close_price": float(row.get("close") or row.get("Close", 0)),
            "volume": int(row.get("volume") or row.get("Volume", 0)),
        })
    
    # PostgreSQL UPSERT using ON CONFLICT
    stmt = insert(StockPrice).values(records)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_stock_price_date",
        set_={
            "open_price": stmt.excluded.open_price,
            "high_price": stmt.excluded.high_price,
            "low_price": stmt.excluded.low_price,
            "close_price": stmt.excluded.close_price,
            "volume": stmt.excluded.volume,
        },
    )
    db.execute(stmt)
    
    # Update crawl metadata
    last_date = pd.to_datetime(
        df["date" if "date" in df.columns else "Date"].max()
    ).date()
    
    # Count total records
    count_stmt = select(StockPrice).where(StockPrice.stock_id == stock.id)
    total_records = len(db.execute(count_stmt).all())
    
    meta_stmt = insert(CrawlMetadata).values(
        stock_symbol=stock_symbol,
        last_crawl_date=datetime.now().date(),
        last_data_date=last_date,
        total_records=total_records,
    )
    meta_stmt = meta_stmt.on_conflict_do_update(
        index_elements=["stock_symbol"],
        set_={
            "last_crawl_date": datetime.now().date(),
            "last_data_date": last_date,
            "total_records": total_records,
            "last_updated": datetime.now(),
        },
    )
    db.execute(meta_stmt)
    
    db.commit()
    return len(records)


def fetch_stock_data(
    db: Session,
    stock_symbol: str,
    to_date: str | None = None,
) -> bool:
    """
    Fetch stock data from VNDirect API with INCREMENTAL CRAWLING.
    
    Logic:
    - Check last date in database
    - If exists: Only fetch from (last_date + 1) to today
    - If not exists: Fetch all historical data from DATA_START_DATE
    - Insert into PostgreSQL with UPSERT
    
    Args:
        db: Database session
        stock_symbol: Stock symbol to fetch (e.g. "VCB", "FPT")
        to_date: End date for fetching (defaults to today)
        
    Returns:
        True if successful, False otherwise
    """
    end_date = to_date or datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Check last date in database for incremental crawling
        last_date = get_last_data_date(db, stock_symbol)
        
        if last_date:
            # Incremental: Only fetch from day after last date
            start_date = (
                datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            logger.info(f"[{stock_symbol}] Incremental crawl from {start_date} to {end_date}")
            
            # Already up-to-date
            if start_date > end_date:
                logger.info(f"[{stock_symbol}] Already up-to-date (last: {last_date})")
                return True
        else:
            # First time: fetch all historical data
            start_date = settings.data_start_date
            logger.info(f"[{stock_symbol}] Initial crawl from {start_date} to {end_date}")
        
        # Build API request URL
        query_params = f"sort=date&q=code:{stock_symbol}~date:gte:{start_date}~date:lte:{end_date}&size=9990&page=1"
        api_url = f"{settings.vndirect_api_url}?{query_params}"
        
        # Fetch data from API
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        request = Request(api_url, headers=headers)
        response_data = urlopen(request, timeout=30).read()
        parsed_data = json.loads(response_data)["data"]
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_data)
        
        if len(df) == 0:
            logger.info(f"[{stock_symbol}] No new data (market closed or weekend)")
            return True  # Not an error
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Map VNDirect API column names to our standard names
        # VNDirect uses 'nmvolume' (normal matched volume) instead of 'volume'
        if "nmvolume" in df.columns and "volume" not in df.columns:
            df["volume"] = df["nmvolume"]
        
        # Insert into database (with UPSERT on conflict)
        inserted_count = insert_stock_data(db, df, stock_symbol)
        logger.info(f"[{stock_symbol}] Inserted/Updated {inserted_count} records in database")
        
        return True
        
    except Exception as e:
        logger.error(f"[{stock_symbol}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def fetch_all_stocks(
    db: Session,
    stock_symbols: list[str],
    to_date: str | None = None,
) -> dict[str, list[str]]:
    """
    Fetch data for multiple stocks.
    
    Args:
        db: Database session
        stock_symbols: List of stock symbols
        to_date: End date for fetching
        
    Returns:
        Dict with success/failed lists
    """
    results: dict[str, list[str]] = {"success": [], "failed": []}
    
    for stock in stock_symbols:
        if fetch_stock_data(db, stock, to_date=to_date):
            results["success"].append(stock)
        else:
            results["failed"].append(stock)
    
    return results


def load_stock_data_from_db(db: Session, stock_symbol: str) -> pd.DataFrame:
    """
    Load all stock data for a symbol from PostgreSQL.
    
    Args:
        db: Database session
        stock_symbol: Stock symbol to load
        
    Returns:
        DataFrame with stock price data ordered by date
    """
    # Get stock
    stmt = select(Stock).where(Stock.ticker == stock_symbol)
    stock = db.execute(stmt).scalar_one_or_none()
    
    if not stock:
        return pd.DataFrame()
    
    # Get prices
    price_stmt = (
        select(StockPrice)
        .where(StockPrice.stock_id == stock.id)
        .order_by(StockPrice.price_date.asc())
    )
    prices = db.execute(price_stmt).scalars().all()
    
    if not prices:
        return pd.DataFrame()
    
    # Convert to DataFrame
    data = []
    for p in prices:
        data.append({
            "date": p.price_date,
            "open": float(p.open_price) if p.open_price else 0,
            "high": float(p.high_price) if p.high_price else 0,
            "low": float(p.low_price) if p.low_price else 0,
            "close": float(p.close_price),
            "volume": p.volume or 0,
        })
    
    return pd.DataFrame(data)


def get_database_stats(db: Session) -> list[dict]:
    """
    Get statistics about data in database.
    
    Args:
        db: Database session
        
    Returns:
        List of stats per stock
    """
    from sqlalchemy import func
    
    stmt = (
        select(
            Stock.ticker,
            func.count(StockPrice.id).label("record_count"),
            func.min(StockPrice.price_date).label("first_date"),
            func.max(StockPrice.price_date).label("last_date"),
        )
        .join(StockPrice, Stock.id == StockPrice.stock_id)
        .group_by(Stock.ticker)
        .order_by(Stock.ticker)
    )
    
    results = db.execute(stmt).all()
    
    stats = []
    for row in results:
        stats.append({
            "symbol": row.ticker,
            "records": row.record_count,
            "first_date": row.first_date.strftime("%Y-%m-%d") if row.first_date else None,
            "last_date": row.last_date.strftime("%Y-%m-%d") if row.last_date else None,
        })
    
    return stats

