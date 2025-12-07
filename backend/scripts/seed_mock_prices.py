"""
Script to seed mock price data for all stocks in the database.
Run with: docker exec stock-prediction-api python -m scripts.seed_mock_prices

Note: Mocks prices for all symbols from scripts.constant VN30_STOCKS list.
"""

from datetime import date, timedelta
from decimal import Decimal
import random

from scripts.constant import VN30_STOCKS

from sqlalchemy import select

from app.db.models import Stock, StockPrice
from app.db.session import SessionLocal

# Use all symbols from scripts.constant VN30_STOCKS
MOCK_SYMBOLS = VN30_STOCKS

# Base prices for the mocked symbols (in VND, thousands)
# If a stock is not in this list, a default price will be used
BASE_PRICES = {
    "FPT": 125000.0,   # ~125k VND
    "VCB": 95000.0,    # ~95k VND
    "VNM": 85000.0,    # ~85k VND
    "HPG": 28000.0,    # ~28k VND
    "VIC": 75000.0,    # ~75k VND
    "VHM": 65000.0,    # ~65k VND
    "MSN": 78000.0,    # ~78k VND
    "SAB": 180000.0,   # ~180k VND
}


def generate_mock_price(base_price: float, date_offset: int, volatility: float = 0.02) -> dict:
    """
    Generate mock OHLCV data for a stock.
    
    Args:
        base_price: Base price for the stock
        date_offset: Days offset from today (0 = today, negative = past)
        volatility: Price volatility factor (default 2%)
        
    Returns:
        Dict with open, high, low, close, volume
    """
    # Random walk with slight upward trend
    trend = 1 + (random.random() - 0.45) * volatility  # Slight positive bias
    price_change = base_price * trend - base_price
    
    close_price = base_price + price_change
    open_price = base_price + (random.random() - 0.5) * base_price * volatility * 0.5
    
    high_price = max(open_price, close_price) + abs(random.random() * base_price * volatility * 0.3)
    low_price = min(open_price, close_price) - abs(random.random() * base_price * volatility * 0.3)
    
    # Volume: higher on days with bigger price movements
    price_range = high_price - low_price
    base_volume = 1000000  # Base volume
    volume = int(base_volume * (1 + price_range / base_price * 10))
    
    return {
        "open": round(Decimal(str(open_price)), 4),
        "high": round(Decimal(str(high_price)), 4),
        "low": round(Decimal(str(low_price)), 4),
        "close": round(Decimal(str(close_price)), 4),
        "volume": volume,
    }


def seed_mock_prices(days_back: int = 60):
    """
    Seed mock price data for stocks in the database.
    Only processes stocks that match the mocked symbols from scripts.constant.
    
    Args:
        days_back: Number of days of historical data to generate (default 60)
    """
    db = SessionLocal()
    try:
        # Get only stocks that match our mocked symbols
        stmt = select(Stock).where(
            Stock.is_active == True,  # noqa: E712
            Stock.ticker.in_([s.upper() for s in MOCK_SYMBOLS])
        ).order_by(Stock.ticker)
        all_stocks = db.execute(stmt).scalars().all()
        
        if not all_stocks:
            print("⚠️  No matching stocks found. Please run seed_stocks.py first.")
            print(f"   Expected symbols: {', '.join(MOCK_SYMBOLS)}")
            return
        
        print(f"📊 Found {len(all_stocks)} stocks matching mocked symbols. Generating price data...\n")
        print(f"   Mocked symbols: {', '.join(MOCK_SYMBOLS)}\n")
        
        today = date.today()
        print(f"Today: {today}")
        total_inserted = 0
        
        for stock in all_stocks:
            ticker = stock.ticker
            
            # Check existing prices
            existing_stmt = select(StockPrice).where(StockPrice.stock_id == stock.id)
            existing_prices = db.execute(existing_stmt).scalars().all()
            existing_dates = {p.price_date for p in existing_prices}
            
            # Get base price for this stock, or use default
            current_price = BASE_PRICES.get(ticker.upper(), 50000.0)
            inserted_count = 0
            
            for day_offset in range(days_back - 1, -1, -1):  # From oldest to newest
                price_date = today - timedelta(days=day_offset)
                
                # Skip weekends (Saturday=5, Sunday=6)
                if price_date.weekday() >= 5:
                    continue
                
                # Skip if already exists
                if price_date in existing_dates:
                    continue
                
                # Generate mock price data
                price_data = generate_mock_price(current_price, day_offset)
                
                # Update current_price for next iteration (use close price)
                current_price = float(price_data["close"])
                
                # Create StockPrice record
                stock_price = StockPrice(
                    stock_id=stock.id,
                    price_date=price_date,
                    open_price=price_data["open"],
                    high_price=price_data["high"],
                    low_price=price_data["low"],
                    close_price=price_data["close"],
                    volume=price_data["volume"],
                )
                db.add(stock_price)
                inserted_count += 1
            
            if inserted_count > 0:
                db.commit()
                print(f"✓ {ticker}: Inserted {inserted_count} days of price data")
                total_inserted += inserted_count
            else:
                print(f"○ {ticker}: No new data (already exists or no trading days)")
        
        print(f"\n✅ Seeding complete!")
        print(f"   Total price records inserted: {total_inserted}")
        print(f"   Stocks processed: {len(all_stocks)}")
        print(f"   Date range: {(today - timedelta(days=days_back)).isoformat()} to {today.isoformat()}")
        
    except Exception as e:
        print(f"❌ Error seeding mock prices: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    seed_mock_prices(days_back=60)

