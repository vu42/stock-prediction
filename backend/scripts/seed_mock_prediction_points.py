"""
Script to seed mock prediction points (per-day forecast prices) for all stocks.
Run with: docker exec stock-prediction-api python -m scripts.seed_mock_prediction_points

This creates StockPredictionPoint records which are used by the chart API
to show predicted prices for future dates.

Note: Mocks prediction points for all symbols from scripts.constant VN30_STOCKS list.
"""

from datetime import date, timedelta
from decimal import Decimal
import random

from scripts.constant import VN30_STOCKS

from sqlalchemy import select, and_

from app.db.models import Stock, StockPredictionPoint, StockPrice
from app.db.session import SessionLocal

# Use all symbols from scripts.constant VN30_STOCKS
MOCK_SYMBOLS = VN30_STOCKS

# Horizons to generate predictions for
HORIZONS = [7, 15, 30]


def generate_future_price(
    base_price: float,
    days_ahead: int,
    horizon_days: int,
    volatility: float = 0.02,
) -> float:
    """
    Generate mock predicted price for a future date.
    
    Args:
        base_price: Current stock price
        days_ahead: How many days into the future (1, 2, 3, ...)
        horizon_days: Prediction horizon (7, 15, or 30 days)
        volatility: Daily price volatility factor (default 2%)
        
    Returns:
        Predicted price for that future date
    """
    # Calculate trend based on horizon (longer horizons have more trend)
    horizon_multiplier = {7: 1.0, 15: 1.5, 30: 2.0}.get(horizon_days, 1.0)
    
    # Random walk with slight upward bias (60% chance positive)
    direction = 1 if random.random() < 0.6 else -1
    
    # Daily change percentage
    daily_change_pct = (random.uniform(0.1, 0.5) * horizon_multiplier * volatility * direction)
    
    # Apply compound growth over days_ahead
    predicted_price = base_price * ((1 + daily_change_pct) ** days_ahead)
    
    # Add some random noise
    noise_factor = random.uniform(0.98, 1.02)
    predicted_price *= noise_factor
    
    return round(predicted_price, 2)


def seed_mock_prediction_points():
    """
    Seed mock prediction points (per-day forecast prices) for stocks.
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
            print(f"   Expected symbols: {', '.join(MOCK_SYMBOLS)}")
            return
        
        print(f"📊 Found {len(all_stocks)} stocks matching mocked symbols. Generating prediction points...\n")
        print(f"   Mocked symbols: {', '.join(MOCK_SYMBOLS)}\n")
        
        # Use today as the reference date
        today = date.today()
        
        total_inserted = 0
        total_updated = 0
        
        for stock in all_stocks:
            ticker = stock.ticker
            
            # Get latest price date and price
            latest_price_stmt = (
                select(StockPrice.price_date, StockPrice.close_price)
                .where(StockPrice.stock_id == stock.id)
                .order_by(StockPrice.price_date.desc())
                .limit(1)
            )
            latest_price_result = db.execute(latest_price_stmt).first()
            
            if not latest_price_result:
                print(f"⚠️  {ticker}: No price data found. Skipping.")
                continue
            
            latest_price_date = latest_price_result.price_date
            base_price = float(latest_price_result.close_price)
            
            # Start predictions from the day after latest price date
            prediction_start_date = latest_price_date + timedelta(days=1)
            
            for horizon_days in HORIZONS:
                # Generate predictions for each day in the horizon
                prediction_end_date = prediction_start_date + timedelta(days=horizon_days - 1)
                
                points_created = 0
                for day_offset in range(horizon_days):
                    prediction_date = prediction_start_date + timedelta(days=day_offset)
                    
                    # Check if prediction point already exists
                    existing_stmt = select(StockPredictionPoint).where(
                        and_(
                            StockPredictionPoint.stock_id == stock.id,
                            StockPredictionPoint.horizon_days == horizon_days,
                            StockPredictionPoint.prediction_date == prediction_date,
                            StockPredictionPoint.experiment_run_id.is_(None),  # Mock data
                        )
                    )
                    existing = db.execute(existing_stmt).scalar_one_or_none()
                    
                    # Generate predicted price
                    predicted_price = generate_future_price(
                        base_price, day_offset + 1, horizon_days
                    )
                    
                    if existing:
                        # Update existing prediction
                        existing.predicted_price = Decimal(str(predicted_price))
                        total_updated += 1
                    else:
                        # Create new prediction point
                        prediction_point = StockPredictionPoint(
                            stock_id=stock.id,
                            horizon_days=horizon_days,
                            prediction_date=prediction_date,
                            predicted_price=Decimal(str(predicted_price)),
                            experiment_run_id=None,  # Mock data, no experiment run
                        )
                        db.add(prediction_point)
                        total_inserted += 1
                        points_created += 1
                
                # Commit after each horizon to avoid large transactions
                db.commit()
            
            # Show progress
            print(f"✓ {ticker}: Generated {points_created} prediction points per horizon")
        
        print(f"\n✅ Seeding complete!")
        print(f"   Total prediction points inserted: {total_inserted}")
        print(f"   Total prediction points updated: {total_updated}")
        print(f"   Stocks processed: {len(all_stocks)}")
        print(f"   Horizons: {', '.join([f'{h}d' for h in HORIZONS])}")
        print(f"   Prediction start date: {prediction_start_date.isoformat()}")
        print(f"\n💡 Note: Prediction points are randomly generated for testing purposes.")
        print(f"   They represent per-day forecast prices for future dates.")
        print(f"   Used by the chart API to overlay predicted prices on historical data.")
        
    except Exception as e:
        print(f"❌ Error seeding mock prediction points: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    seed_mock_prediction_points()

