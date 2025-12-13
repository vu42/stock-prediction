"""
Script to seed the database with VN30+ stock metadata.
Run with: docker exec stock-prediction-api python -m scripts.seed_stocks

This script updates all stocks in the database with complete metadata:
- Name (company full name)
- Sector (industry classification)  
- Exchange (HOSE/HNX)
- Description (auto-generated from name and sector)
"""

from sqlalchemy import select

from app.db.models import Stock
from app.db.session import SessionLocal
from app.services.stock_service import create_stock, update_stock

# Stock metadata mapping - Complete VN30 + additional stocks
STOCK_METADATA = {
    # Banking & Financial Services
    "ACB": {"name": "Asia Commercial Bank", "sector": "Banking"},
    "BID": {"name": "BIDV", "sector": "Banking"},
    "CTG": {"name": "VietinBank", "sector": "Banking"},
    "HDB": {"name": "HDBank", "sector": "Banking"},
    "MBB": {"name": "Military Bank", "sector": "Banking"},
    "STB": {"name": "Sacombank", "sector": "Banking"},
    "TCB": {"name": "Techcombank", "sector": "Banking"},
    "TPB": {"name": "TPBank", "sector": "Banking"},
    "VCB": {"name": "Vietcombank", "sector": "Banking"},
    "VIB": {"name": "Vietnam International Bank", "sector": "Banking"},
    "VPB": {"name": "VPBank", "sector": "Banking"},
    
    # Securities & Financial Services
    "SSI": {"name": "SSI Securities", "sector": "Financial Services"},
    
    # Insurance
    "BVH": {"name": "Bao Viet Holdings", "sector": "Insurance"},
    
    # Real Estate
    "BCM": {"name": "Becamex IDC", "sector": "Real Estate"},
    "KDH": {"name": "Khang Dien House", "sector": "Real Estate"},
    "NVL": {"name": "Novaland", "sector": "Real Estate"},
    "PDR": {"name": "Phat Dat Real Estate", "sector": "Real Estate"},
    "VHM": {"name": "Vinhomes", "sector": "Real Estate"},
    "VIC": {"name": "Vingroup", "sector": "Real Estate"},
    "VRE": {"name": "Vincom Retail", "sector": "Real Estate"},
    
    # Technology
    "FPT": {"name": "FPT Corporation", "sector": "Technology"},
    
    # Industrial & Materials
    "GAS": {"name": "PV Gas", "sector": "Energy"},
    "GVR": {"name": "Vietnam Rubber Group", "sector": "Materials"},
    "HPG": {"name": "Hoa Phat Group", "sector": "Industrial"},
    "PLX": {"name": "Petrolimex", "sector": "Energy"},
    "POW": {"name": "PetroVietnam Power", "sector": "Energy"},
    
    # Consumer Goods & Retail
    "MSN": {"name": "Masan Group", "sector": "Consumer Goods"},
    "MWG": {"name": "Mobile World Group", "sector": "Retail"},
    "SAB": {"name": "Sabeco", "sector": "Consumer Goods"},
    "VNM": {"name": "Vinamilk", "sector": "Consumer Goods"},
    
    # Aviation & Logistics
    "VJC": {"name": "Vietjet Air", "sector": "Aviation"},
    "VTP": {"name": "Viettel Post", "sector": "Logistics"},
    
    # Diversified
    "VPI": {"name": "Van Phu Invest", "sector": "Real Estate"},
}

def seed_stocks():
    """
    Seed the database with stock metadata.
    
    This will:
    1. Update existing stocks with complete metadata (name, sector, exchange, description)
    2. Create new stocks from STOCK_METADATA if they don't exist
    """
    db = SessionLocal()
    try:
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        # First, get all existing stocks from DB
        stmt = select(Stock).order_by(Stock.ticker)
        existing_stocks = db.execute(stmt).scalars().all()
        existing_tickers = {s.ticker for s in existing_stocks}
        
        # Update existing stocks with metadata
        for stock in existing_stocks:
            ticker = stock.ticker
            if ticker in STOCK_METADATA:
                stock_info = STOCK_METADATA[ticker]
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
                print(f"✓ Updated: {ticker} - {stock_info['name']} ({stock_info['sector']})")
            else:
                skipped_count += 1
                print(f"⚠️  No metadata for: {ticker} (keeping existing data)")
        
        # Create any stocks from STOCK_METADATA that don't exist yet
        for ticker, stock_info in STOCK_METADATA.items():
            if ticker not in existing_tickers:
                create_stock(
                    db,
                    ticker=ticker,
                    name=stock_info["name"],
                    sector=stock_info["sector"],
                    exchange="HOSE",
                    description=f"{stock_info['name']} is a leading company in the {stock_info['sector']} sector.",
                )
                created_count += 1
                print(f"✓ Created: {ticker} - {stock_info['name']} ({stock_info['sector']})")
        
        print(f"\n✅ Seeding complete!")
        print(f"   Created: {created_count} stocks")
        print(f"   Updated: {updated_count} stocks")
        print(f"   Skipped: {skipped_count} stocks (no metadata)")
        print(f"   Total metadata available: {len(STOCK_METADATA)} stocks")
        
    except Exception as e:
        print(f"❌ Error seeding stocks: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_stocks()

