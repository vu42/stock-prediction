#!/usr/bin/env python3
"""
Initialize database tables for stock prediction system
Run this once before training models
"""
from modules.database import init_database

if __name__ == "__main__":
    print("Initializing database tables...")
    print("=" * 60)

    try:
        init_database()
        print("\n✅ Database initialized successfully!")
        print("\nTables created:")
        print("  - stock_prices (stores OHLCV data)")
        print("  - crawl_metadata (tracks last crawl dates)")
        print("\nYou can now run: python train_local.py VCB")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Failed to initialize database: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check PostgreSQL is running: docker ps | grep stock-postgres")
        print("2. Verify config.py has correct database credentials")
        print("3. Test connection: psql -h localhost -U postgres -d stock_prediction")
        import traceback

        traceback.print_exc()
