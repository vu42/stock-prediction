"""
Script to seed mock model status and horizon metrics for all stocks.
Run with: docker exec stock-prediction-api python -m scripts.seed_mock_model_metrics

This creates ModelStatus and ModelHorizonMetric records which are used by:
- The Models page to show MAPE values for each ticker
- The generate_evaluation_results script to generate report Section 6.7

Note: Mocks model metrics for all symbols from scripts.constant VN30_STOCKS list.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import random

from scripts.constant import VN30_STOCKS

from sqlalchemy import select

from app.db.models import Stock, ModelStatus, ModelHorizonMetric
from app.db.session import SessionLocal

# Use all symbols from scripts.constant VN30_STOCKS
MOCK_SYMBOLS = VN30_STOCKS

# Horizons to generate metrics for
HORIZONS = [7, 15, 30]

# MAPE ranges for more realistic mock data
# Shorter horizons tend to have lower MAPE
MAPE_RANGES = {
    7: (2.0, 6.0),    # 7D: 2-6% MAPE
    15: (3.5, 8.0),   # 15D: 3.5-8% MAPE
    30: (5.0, 12.0),  # 30D: 5-12% MAPE
}


def generate_mock_mape(horizon_days: int) -> float:
    """
    Generate mock MAPE value for a given horizon.
    
    Args:
        horizon_days: Prediction horizon (7, 15, or 30 days)
        
    Returns:
        MAPE percentage value
    """
    min_mape, max_mape = MAPE_RANGES.get(horizon_days, (3.0, 10.0))
    return round(random.uniform(min_mape, max_mape), 3)


def seed_mock_model_metrics():
    """
    Seed mock model status and horizon metrics for stocks.
    Only processes stocks that match the mocked symbols from scripts.constant.
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
            return
        
        print(f"📊 Seeding mock model metrics for {len(all_stocks)} stocks...")
        print(f"   Symbols: {', '.join([s.ticker for s in all_stocks])}")
        print()
        
        total_statuses = 0
        total_metrics = 0
        
        for stock in all_stocks:
            ticker = stock.ticker
            
            # Check if model status already exists for this stock
            existing_stmt = select(ModelStatus).where(
                ModelStatus.stock_id == stock.id
            )
            existing = db.execute(existing_stmt).scalar_one_or_none()
            
            if existing:
                # Update existing model status
                status = existing
                status.freshness_state = "fresh"
                status.last_updated_at = datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 48))
                print(f"↻ {ticker}: Updating existing model status")
            else:
                # Create new model status
                status = ModelStatus(
                    stock_id=stock.id,
                    freshness_state="fresh",
                    last_updated_at=datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 48)),
                )
                db.add(status)
                db.flush()  # Get the ID
                total_statuses += 1
                print(f"✓ {ticker}: Created new model status")
            
            # Now add/update horizon metrics
            for horizon in HORIZONS:
                mape_value = generate_mock_mape(horizon)
                
                # Check if metric already exists
                metric_stmt = select(ModelHorizonMetric).where(
                    ModelHorizonMetric.model_status_id == status.id,
                    ModelHorizonMetric.horizon_days == horizon
                )
                existing_metric = db.execute(metric_stmt).scalar_one_or_none()
                
                if existing_metric:
                    existing_metric.mape_pct = Decimal(str(mape_value))
                else:
                    metric = ModelHorizonMetric(
                        model_status_id=status.id,
                        horizon_days=horizon,
                        mape_pct=Decimal(str(mape_value)),
                    )
                    db.add(metric)
                    total_metrics += 1
                
                print(f"   {horizon}D MAPE: {mape_value:.2f}%")
        
        db.commit()
        
        print(f"\n✅ Seeding complete!")
        print(f"   New model statuses created: {total_statuses}")
        print(f"   New horizon metrics created: {total_metrics}")
        print(f"   Stocks processed: {len(all_stocks)}")
        print(f"   Horizons: {', '.join([f'{h}d' for h in HORIZONS])}")
        print(f"\n💡 Note: MAPE values are randomly generated for testing purposes.")
        print(f"   They represent expected prediction error percentages.")
        print(f"   Used by the Models page and evaluation results script.")
        
    except Exception as e:
        print(f"❌ Error seeding mock model metrics: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    seed_mock_model_metrics()
