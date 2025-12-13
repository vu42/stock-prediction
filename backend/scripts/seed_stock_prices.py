"""
Script to seed stock price data from CSV files in stock_data folder.
Run with: docker exec stock-prediction-api python -m scripts.seed_stock_prices

This script reads historical stock data from CSV files (VNDirect format)
and inserts them into the stock_prices table.

CSV files are expected in: /app/stock_data/{TICKER}/{TICKER}_historical_data.csv
"""

import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Stock, StockPrice
from app.db.session import SessionLocal

# Path to stock data folder
# In Docker: /app/stock_data/
# Locally: ../../stock_data/ from backend/scripts/
def get_stock_data_dir():
    """Get the stock data directory path."""
    # Check environment variable first
    if os.environ.get("STOCK_DATA_DIR"):
        return os.environ["STOCK_DATA_DIR"]
    
    # In Docker container, use /app/stock_data
    docker_path = "/app/stock_data"
    if os.path.exists(docker_path):
        return docker_path
    
    # Locally, go up from backend/scripts to project root
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "stock_data"
    )
    return local_path


STOCK_DATA_DIR = get_stock_data_dir()


def get_or_create_stock(db, ticker: str) -> Stock:
    """
    Get existing stock or create a new one.
    
    Args:
        db: Database session
        ticker: Stock ticker symbol
        
    Returns:
        Stock model instance
    """
    stmt = select(Stock).where(Stock.ticker == ticker)
    stock = db.execute(stmt).scalar_one_or_none()
    
    if not stock:
        stock = Stock(
            ticker=ticker,
            name=ticker,  # Will be updated with proper name later
            is_active=True,
        )
        db.add(stock)
        db.flush()
        print(f"  Created new stock record: {ticker}")
    
    return stock


def seed_stock_prices_from_csv():
    """
    Seed stock prices from CSV files in stock_data folder.
    
    CSV format (VNDirect):
    - code: Stock ticker
    - date: Trading date (YYYY-MM-DD)
    - open, high, low, close: Price values
    - nmVolume: Trading volume
    """
    db = SessionLocal()
    
    try:
        stock_data_path = Path(STOCK_DATA_DIR)
        
        if not stock_data_path.exists():
            print(f"❌ Stock data directory not found: {stock_data_path}")
            print("   Please ensure stock_data folder is mounted in Docker container.")
            return
        
        # Find all ticker folders
        ticker_folders = [d for d in stock_data_path.iterdir() if d.is_dir()]
        
        if not ticker_folders:
            print(f"⚠️  No ticker folders found in {stock_data_path}")
            return
        
        print(f"📊 Found {len(ticker_folders)} ticker folders in {stock_data_path}\n")
        
        total_inserted = 0
        total_updated = 0
        
        for ticker_folder in sorted(ticker_folders):
            ticker = ticker_folder.name.upper()
            csv_file = ticker_folder / f"{ticker}_historical_data.csv"
            
            if not csv_file.exists():
                print(f"⚠️  {ticker}: CSV file not found, skipping")
                continue
            
            print(f"Processing {ticker}...")
            
            # Get or create stock record
            stock = get_or_create_stock(db, ticker)
            
            # Read CSV file
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"  ❌ Error reading CSV: {e}")
                continue
            
            if df.empty:
                print(f"  ⚠️  Empty CSV file, skipping")
                continue
            
            # Prepare records for bulk upsert
            records = []
            for _, row in df.iterrows():
                try:
                    price_date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
                    
                    # Note: VNDirect CSV prices are in thousands VND (e.g., 93.7 = 93,700 VND)
                    # Convert to VND by multiplying by 1000 for consistency with the system
                    price_multiplier = 1000  # Convert from thousands to VND
                    
                    records.append({
                        "stock_id": stock.id,
                        "price_date": price_date,
                        "open_price": Decimal(str(float(row["open"]) * price_multiplier)) if pd.notna(row["open"]) else None,
                        "high_price": Decimal(str(float(row["high"]) * price_multiplier)) if pd.notna(row["high"]) else None,
                        "low_price": Decimal(str(float(row["low"]) * price_multiplier)) if pd.notna(row["low"]) else None,
                        "close_price": Decimal(str(float(row["close"]) * price_multiplier)) if pd.notna(row["close"]) else Decimal("0"),
                        "volume": int(row["nmVolume"]) if pd.notna(row["nmVolume"]) else 0,
                    })
                except Exception as e:
                    print(f"  ⚠️  Error parsing row: {e}")
                    continue
            
            if not records:
                print(f"  ⚠️  No valid records to insert")
                continue
            
            # Bulk upsert using PostgreSQL ON CONFLICT
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
            
            result = db.execute(stmt)
            db.commit()
            
            # Get date range for logging
            dates = [r["price_date"] for r in records]
            min_date = min(dates)
            max_date = max(dates)
            
            print(f"  ✓ {ticker}: {len(records)} records ({min_date} to {max_date})")
            total_inserted += len(records)
        
        print(f"\n✅ Seeding complete!")
        print(f"   Total records processed: {total_inserted}")
        print(f"   Tickers processed: {len(ticker_folders)}")
        
    except Exception as e:
        print(f"❌ Error seeding stock prices: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_stock_prices_from_csv()

