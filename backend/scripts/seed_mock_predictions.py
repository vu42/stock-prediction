"""
Script to seed mock prediction summaries for all stocks.
Run with: docker exec stock-prediction-api python -m scripts.seed_mock_predictions

Note: Mocks predictions for all symbols from scripts.constant VN30_STOCKS list.
"""

from datetime import date, timedelta
from decimal import Decimal
import random

from scripts.constant import VN30_STOCKS

from sqlalchemy import select, and_

from app.db.models import Stock, StockPredictionSummary, StockPrice
from app.db.session import SessionLocal

# Use all symbols from scripts.constant VN30_STOCKS
MOCK_SYMBOLS = VN30_STOCKS

# Horizons to generate predictions for
HORIZONS = [7, 15, 30]


def generate_mock_prediction(
    base_price: float,
    horizon_days: int,
    volatility: float = 0.03,
) -> float:
    """
    Generate mock predicted % change for a stock.
    
    Args:
        base_price: Current stock price
        horizon_days: Prediction horizon (7, 15, or 30 days)
        volatility: Price volatility factor (default 3%)
        
    Returns:
        Predicted % change (can be positive or negative)
    """
    # Longer horizons tend to have larger absolute changes
    horizon_multiplier = {7: 1.0, 15: 1.5, 30: 2.0}.get(horizon_days, 1.0)
    
    # Random walk: 60% chance positive, 40% chance negative
    # This ensures we have both "should buy" and "should sell" candidates
    direction = 1 if random.random() < 0.6 else -1
    
    # Generate change percentage with some randomness
    base_change = random.uniform(0.5, 3.0) * horizon_multiplier * volatility * 100
    predicted_pct = direction * base_change
    
    # Add some noise
    noise = random.uniform(-0.5, 0.5)
    predicted_pct += noise
    
    return round(predicted_pct, 2)


def seed_mock_predictions():
    """
    Seed mock prediction summaries for stocks.
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
        
        print(f"📊 Found {len(all_stocks)} stocks matching mocked symbols. Generating prediction summaries...\n")
        print(f"   Mocked symbols: {', '.join(MOCK_SYMBOLS)}\n")
        
        # Use today as the as_of_date for all predictions
        as_of_date = date.today()
        
        total_inserted = 0
        total_updated = 0
        
        for stock in all_stocks:
            ticker = stock.ticker
            
            # Get current price (latest close price from stock_prices)
            # We'll use a default base price if no price data exists
            base_price = 50000.0  # Default fallback
            
            # Try to get actual current price from stock_prices if available
            latest_price_stmt = (
                select(StockPrice.close_price)
                .where(StockPrice.stock_id == stock.id)
                .order_by(StockPrice.price_date.desc())
                .limit(1)
            )
            latest_price = db.execute(latest_price_stmt).scalar_one_or_none()
            if latest_price:
                base_price = float(latest_price)
            
            predictions_data = []
            for horizon_days in HORIZONS:
                # Check if prediction already exists for this stock/date/horizon
                existing_stmt = select(StockPredictionSummary).where(
                    and_(
                        StockPredictionSummary.stock_id == stock.id,
                        StockPredictionSummary.as_of_date == as_of_date,
                        StockPredictionSummary.horizon_days == horizon_days,
                    )
                )
                existing = db.execute(existing_stmt).scalar_one_or_none()
                
                # Generate mock prediction
                predicted_pct = generate_mock_prediction(base_price, horizon_days)
                
                if existing:
                    # Update existing prediction
                    existing.predicted_change_pct = Decimal(str(predicted_pct))
                    total_updated += 1
                else:
                    # Create new prediction
                    prediction = StockPredictionSummary(
                        stock_id=stock.id,
                        as_of_date=as_of_date,
                        horizon_days=horizon_days,
                        predicted_change_pct=Decimal(str(predicted_pct)),
                        experiment_run_id=None,  # Mock data, no experiment run
                    )
                    db.add(prediction)
                    total_inserted += 1
                
                # Store for progress display
                predictions_data.append((horizon_days, predicted_pct))
            
            # Commit after each stock to avoid large transactions
            db.commit()
            
            # Show progress
            predictions_str = ", ".join([
                f"{h}d: {pct:+.2f}%"
                for h, pct in predictions_data
            ])
            print(f"✓ {ticker}: {predictions_str}")
        
        print(f"\n✅ Seeding complete!")
        print(f"   Total prediction summaries inserted: {total_inserted}")
        print(f"   Total prediction summaries updated: {total_updated}")
        print(f"   Stocks processed: {len(all_stocks)}")
        print(f"   Horizons: {', '.join([f'{h}d' for h in HORIZONS])}")
        print(f"   As of date: {as_of_date.isoformat()}")
        print(f"\n💡 Note: Predictions are randomly generated for testing purposes.")
        print(f"   Should Buy tab will show stocks with positive predicted changes.")
        print(f"   Should Sell tab will show stocks with negative predicted changes.")
        
    except Exception as e:
        print(f"❌ Error seeding mock predictions: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    seed_mock_predictions()

