#!/usr/bin/env python3
"""
Local training script - Train models on your laptop without Airflow

Usage:
    python train_local.py              # Train default stocks (VCB, FPT)
    python train_local.py VCB          # Train single stock
    python train_local.py VCB FPT VNM  # Train multiple stocks
    python train_local.py --all        # Train all VN30 stocks
"""
import sys
import argparse
from datetime import datetime
from modules.model_trainer_sklearn import (
    train_prediction_model_sklearn,
    evaluate_model_sklearn,
    predict_future_prices_sklearn,
)
from modules.data_fetcher import fetch_stock_data
from modules.database import init_database
from config import VN30_STOCKS


def train_single_stock(stock_symbol, fetch_data=True, continue_training=False):
    """
    Train model for a single stock

    Args:
        stock_symbol: Stock symbol to train
        fetch_data: Whether to fetch latest data first
        continue_training: Whether to continue from existing model

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Training {stock_symbol}")
    print(f"{'='*60}\n")

    try:
        # Step 1: Fetch latest data (optional)
        if fetch_data:
            print("Step 1: Fetching latest data...")
            context = {"to_date": datetime.now().strftime("%Y-%m-%d")}
            fetch_result = fetch_stock_data(stock_symbol, **context)

            if not fetch_result:
                print(f"❌ Failed to fetch data for {stock_symbol}")
                return False
            print("✅ Data fetched successfully")
        else:
            print("Step 1: Skipping data fetch (using existing database)")

        # Step 2: Train model
        print("\nStep 2: Training model...")
        print(
            f"   Mode: {'Incremental (updating existing model)' if continue_training else 'Fresh training'}"
        )
        train_result = train_prediction_model_sklearn(
            stock_symbol, continue_training=continue_training
        )

        if not train_result:
            print(f"❌ Training failed for {stock_symbol}")
            return False
        print("✅ Model trained successfully")

        # Step 3: Evaluate
        print("\nStep 3: Evaluating model...")
        eval_result = evaluate_model_sklearn(stock_symbol)
        if eval_result:
            print("✅ Evaluation completed")
        else:
            print("⚠️  Evaluation completed (check logs for details)")

        # Step 4: Predict future
        print("\nStep 4: Generating 30-day predictions...")
        predict_result = predict_future_prices_sklearn(stock_symbol, days_ahead=30)
        if predict_result:
            print("✅ Predictions generated")
        else:
            print("⚠️  Prediction completed (check logs for details)")

        print(f"\n{'='*60}")
        print(f"✅ {stock_symbol} completed successfully!")
        print(f"📁 Results saved to: output/{stock_symbol}/")
        print(f"   - {stock_symbol}_sklearn_model.pkl")
        print(f"   - {stock_symbol}_sklearn_scaler.pkl")
        print(f"   - {stock_symbol}_sklearn_evaluation.png")
        print(f"   - {stock_symbol}_sklearn_future_predictions.csv")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\n❌ Error training {stock_symbol}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train stock prediction models locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_local.py                  # Train VCB and FPT
  python train_local.py VCB              # Train only VCB
  python train_local.py VCB FPT VNM      # Train multiple stocks
  python train_local.py --all            # Train all VN30 stocks
  python train_local.py VCB --no-fetch   # Skip data fetching
  python train_local.py VCB --continue   # Continue training existing model
        """,
    )

    parser.add_argument(
        "stocks",
        nargs="*",
        default=["VCB", "FPT"],
        help="Stock symbols to train (default: VCB FPT)",
    )
    parser.add_argument("--all", action="store_true", help="Train all VN30 stocks")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching data (use existing database)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Continue training from existing model",
    )

    args = parser.parse_args()

    # Determine which stocks to train
    if args.all:
        stocks = VN30_STOCKS
        print(f"Training all {len(VN30_STOCKS)} VN30 stocks")
    else:
        stocks = args.stocks

    # Validate stocks
    stocks = [s.upper() for s in stocks]

    print("\n" + "=" * 60)
    print("STOCK PREDICTION MODEL TRAINING")
    print("=" * 60)
    print(f"Stocks to train: {', '.join(stocks)}")
    print(f"Fetch data: {'No' if args.no_fetch else 'Yes'}")
    print(f"Training mode: {'Incremental' if args.continue_training else 'Fresh'}")
    print("=" * 60)

    # Initialize database (if not already initialized)
    try:
        print("\nInitializing database (if needed)...")
        init_database()
        print("✅ Database ready")
    except Exception as e:
        print(f"⚠️  Database initialization: {str(e)}")
        print("Continuing anyway (tables may already exist)...")

    # Train each stock
    success_count = 0
    failed_stocks = []

    for i, stock in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] Processing {stock}...")

        if train_single_stock(
            stock,
            fetch_data=not args.no_fetch,
            continue_training=args.continue_training,
        ):
            success_count += 1
        else:
            failed_stocks.append(stock)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total stocks: {len(stocks)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {len(failed_stocks)}")
    if failed_stocks:
        print(f"Failed stocks: {', '.join(failed_stocks)}")
    print("=" * 60)

    return 0 if success_count == len(stocks) else 1


if __name__ == "__main__":
    sys.exit(main())
