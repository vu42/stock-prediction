"""
Script to seed the database with VN30 stocks.
Run with: docker exec stock-prediction-api python -m scripts.seed_stocks
Or rebuild the container first if scripts directory was just added.
"""

from sqlalchemy import select

from app.db.models import Stock
from app.db.session import SessionLocal
from app.services.stock_service import create_stock, update_stock

# VN30 stocks with their full names and sectors
VN30_STOCKS = [
    {"ticker": "ACB", "name": "Asia Commercial Bank", "sector": "Financial Services"},
    {"ticker": "BID", "name": "Bank for Investment and Development of Vietnam", "sector": "Financial Services"},
    {"ticker": "BVH", "name": "Bao Viet Holdings", "sector": "Insurance"},
    {"ticker": "CTG", "name": "VietinBank", "sector": "Financial Services"},
    {"ticker": "FPT", "name": "FPT Corporation", "sector": "Technology"},
    {"ticker": "GAS", "name": "PetroVietnam Gas", "sector": "Energy"},
    {"ticker": "HDB", "name": "HDBank", "sector": "Financial Services"},
    {"ticker": "HPG", "name": "Hoa Phat Group", "sector": "Industrial"},
    {"ticker": "MBB", "name": "Military Bank", "sector": "Financial Services"},
    {"ticker": "MSN", "name": "Masan Group", "sector": "Consumer Goods"},
    {"ticker": "MWG", "name": "Mobile World Investment Corporation", "sector": "Retail"},
    {"ticker": "PLX", "name": "Petrolimex", "sector": "Energy"},
    {"ticker": "POW", "name": "PetroVietnam Power Corporation", "sector": "Energy"},
    {"ticker": "SAB", "name": "Sabeco", "sector": "Consumer Goods"},
    {"ticker": "SSI", "name": "Saigon Securities Inc.", "sector": "Financial Services"},
    {"ticker": "STB", "name": "Sacombank", "sector": "Financial Services"},
    {"ticker": "TCB", "name": "Techcombank", "sector": "Financial Services"},
    {"ticker": "TPB", "name": "TPBank", "sector": "Financial Services"},
    {"ticker": "VCB", "name": "Vietcombank", "sector": "Financial Services"},
    {"ticker": "VHM", "name": "Vinhomes", "sector": "Real Estate"},
    {"ticker": "VIC", "name": "Vingroup", "sector": "Real Estate"},
    {"ticker": "VJC", "name": "VietJet Air", "sector": "Airlines"},
    {"ticker": "VNM", "name": "Vinamilk", "sector": "Consumer Goods"},
    {"ticker": "VPB", "name": "VPBank", "sector": "Financial Services"},
    {"ticker": "VRE", "name": "Vincom Retail", "sector": "Real Estate"},
    {"ticker": "VSH", "name": "Vietnam Dairy Products", "sector": "Consumer Goods"},
    {"ticker": "VTI", "name": "Viettel Global Investment", "sector": "Technology"},
    {"ticker": "VTO", "name": "Viettel Post", "sector": "Logistics"},
]


def seed_stocks():
    """Seed the database with VN30 stocks."""
    db = SessionLocal()
    try:
        created_count = 0
        updated_count = 0
        
        for stock_data in VN30_STOCKS:
            ticker = stock_data["ticker"]
            
            # Check if stock already exists
            stmt = select(Stock).where(Stock.ticker == ticker.upper())
            existing = db.execute(stmt).scalar_one_or_none()
            
            if existing:
                # Update existing stock with full information
                update_stock(
                    db,
                    ticker=ticker,
                    name=stock_data["name"],
                    sector=stock_data["sector"],
                    exchange="HOSE",
                    description=f"{stock_data['name']} is a leading company in the {stock_data['sector']} sector.",
                    is_active=True,
                )
                updated_count += 1
                print(f"✓ Updated: {ticker} - {stock_data['name']}")
            else:
                # Create new stock
                create_stock(
                    db,
                    ticker=ticker,
                    name=stock_data["name"],
                    sector=stock_data["sector"],
                    exchange="HOSE",
                    description=f"{stock_data['name']} is a leading company in the {stock_data['sector']} sector.",
                )
                created_count += 1
                print(f"✓ Created: {ticker} - {stock_data['name']}")
        
        print(f"\n✅ Seeding complete!")
        print(f"   Created: {created_count} stocks")
        print(f"   Updated: {updated_count} stocks")
        print(f"   Total: {len(VN30_STOCKS)} stocks")
        
    except Exception as e:
        print(f"❌ Error seeding stocks: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_stocks()

