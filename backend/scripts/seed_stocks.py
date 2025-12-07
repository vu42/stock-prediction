"""
Script to seed the database with VN30 stocks.
Run with: docker exec stock-prediction-api python -m scripts.seed_stocks
Or rebuild the container first if scripts directory was just added.

Note: Mocks all symbols from scripts.constant VN30_STOCKS list.
"""

from scripts.constant import VN30_STOCKS

from sqlalchemy import select

from app.db.models import Stock
from app.db.session import SessionLocal
from app.services.stock_service import create_stock, update_stock

# Stock metadata mapping
STOCK_METADATA = {
    "FPT": {"name": "FPT Corporation", "sector": "Technology"},
    "VCB": {"name": "Vietcombank", "sector": "Financial Services"},
    "VNM": {"name": "Vinamilk", "sector": "Consumer Goods"},
    "HPG": {"name": "Hoa Phat Group", "sector": "Industrial"},
    "VIC": {"name": "Vingroup", "sector": "Real Estate"},
    "VHM": {"name": "Vinhomes", "sector": "Real Estate"},
    "MSN": {"name": "Masan Group", "sector": "Consumer Goods"},
    "SAB": {"name": "Sabeco", "sector": "Consumer Goods"},
}

# Use all symbols from scripts.constant VN30_STOCKS
MOCK_SYMBOLS = VN30_STOCKS


def seed_stocks():
    """Seed the database with all mock symbols from scripts.constant VN30_STOCKS."""
    db = SessionLocal()
    try:
        created_count = 0
        updated_count = 0
        
        for ticker in MOCK_SYMBOLS:
            if ticker not in STOCK_METADATA:
                print(f"⚠️  Skipping {ticker}: No metadata found")
                continue
                
            stock_info = STOCK_METADATA[ticker]
            
            # Check if stock already exists
            stmt = select(Stock).where(Stock.ticker == ticker.upper())
            existing = db.execute(stmt).scalar_one_or_none()
            
            if existing:
                # Update existing stock with full information
                update_stock(
                    db,
                    ticker=ticker,
                    name=stock_info["name"],
                    sector=stock_info["sector"],
                    exchange="HOSE",
                    description=f"{stock_info['name']} is a leading company in the {stock_info['sector']} sector.",
                    is_active=True,
                )
                updated_count += 1
                print(f"✓ Updated: {ticker} - {stock_info['name']}")
            else:
                # Create new stock
                create_stock(
                    db,
                    ticker=ticker,
                    name=stock_info["name"],
                    sector=stock_info["sector"],
                    exchange="HOSE",
                    description=f"{stock_info['name']} is a leading company in the {stock_info['sector']} sector.",
                )
                created_count += 1
                print(f"✓ Created: {ticker} - {stock_info['name']}")
        
        print(f"\n✅ Seeding complete!")
        print(f"   Created: {created_count} stocks")
        print(f"   Updated: {updated_count} stocks")
        print(f"   Total: {len(MOCK_SYMBOLS)} stocks (mocked from scripts.constant VN30_STOCKS)")
        
    except Exception as e:
        print(f"❌ Error seeding stocks: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_stocks()

