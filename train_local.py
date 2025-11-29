#!/usr/bin/env python3
"""
Local training script - Train models on your laptop without Airflow

Usage:
    python train_local.py              # Train default stocks (VCB, FPT) with recursive model
    python train_local.py VCB          # Train single stock
    python train_local.py VCB FPT VNM  # Train multiple stocks
    python train_local.py --all        # Train all VN30 stocks
    python train_local.py --multi-horizon  # Use direct multi-horizon model (1d, 3d, 7d, 15d, 30d)
"""
import sys
import time
import argparse
from datetime import datetime
from modules.model_trainer import (
    train_prediction_model,
    evaluate_model,
    predict_future_prices,
    # Multi-horizon functions
    train_multi_horizon_model,
    evaluate_multi_horizon_model,
    predict_multi_horizon,
)
from modules.data_fetcher import fetch_stock_data
from modules.database import init_database
from config import VN30_STOCKS, PREDICTION_HORIZONS


def train_single_stock(
    stock_symbol, fetch_data=True, continue_training=False, multi_horizon=False
):
    """
    Train model for a single stock

    Args:
        stock_symbol: Stock symbol to train
        fetch_data: Whether to fetch latest data first
        continue_training: Whether to continue from existing model
        multi_horizon: If True, use direct multi-horizon model (1d, 3d, 7d, 15d, 30d)

    Returns:
        tuple: (success: bool, timing_info: dict)
    """
    model_type = "Multi-Horizon Direct" if multi_horizon else "Recursive (1-step)"

    # Timing tracking
    timing = {
        "fetch": 0.0,
        "train": 0.0,
        "evaluate": 0.0,
        "predict": 0.0,
        "total": 0.0,
    }
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"Training {stock_symbol} - {model_type} Model")
    print(f"{'='*60}\n")

    try:
        # Step 1: Fetch latest data (optional)
        if fetch_data:
            print("Step 1: Fetching latest data...")
            step_start = time.time()
            context = {"to_date": datetime.now().strftime("%Y-%m-%d")}
            fetch_result = fetch_stock_data(stock_symbol, **context)
            timing["fetch"] = time.time() - step_start

            if not fetch_result:
                print(f"❌ Failed to fetch data for {stock_symbol}")
                return False, timing
            print(f"✅ Data fetched successfully ({timing['fetch']:.2f}s)")
        else:
            print("Step 1: Skipping data fetch (using existing database)")

        # Step 2: Train model
        print("\nStep 2: Training model...")
        print(
            f"   Mode: {'Incremental (updating existing model)' if continue_training else 'Fresh training'}"
        )
        step_start = time.time()

        if multi_horizon:
            print(f"   Horizons: {PREDICTION_HORIZONS} days")
            train_result = train_multi_horizon_model(
                stock_symbol, continue_training=continue_training
            )
        else:
            train_result = train_prediction_model(
                stock_symbol, continue_training=continue_training
            )
        timing["train"] = time.time() - step_start

        if not train_result:
            print(f"❌ Training failed for {stock_symbol}")
            return False, timing
        print(f"✅ Model trained successfully ({timing['train']:.2f}s)")

        # Step 3: Evaluate
        print("\nStep 3: Evaluating model...")
        step_start = time.time()
        if multi_horizon:
            eval_result = evaluate_multi_horizon_model(stock_symbol)
        else:
            eval_result = evaluate_model(stock_symbol)
        timing["evaluate"] = time.time() - step_start

        if eval_result:
            print(f"✅ Evaluation completed ({timing['evaluate']:.2f}s)")
        else:
            print(
                f"⚠️  Evaluation completed ({timing['evaluate']:.2f}s) (check logs for details)"
            )

        # Step 4: Predict future
        if multi_horizon:
            print(
                f"\nStep 4: Generating predictions for horizons {PREDICTION_HORIZONS}..."
            )
        else:
            print("\nStep 4: Generating 30-day predictions...")
        step_start = time.time()

        if multi_horizon:
            predict_result = predict_multi_horizon(stock_symbol)
        else:
            predict_result = predict_future_prices(stock_symbol, days_ahead=30)
        timing["predict"] = time.time() - step_start

        if predict_result:
            print(f"✅ Predictions generated ({timing['predict']:.2f}s)")
        else:
            print(
                f"⚠️  Prediction completed ({timing['predict']:.2f}s) (check logs for details)"
            )

        timing["total"] = time.time() - total_start

        print(f"\n{'='*60}")
        print(f"✅ {stock_symbol} completed successfully!")
        print(f"📁 Results saved to: output/{stock_symbol}/")

        if multi_horizon:
            print(f"   - {stock_symbol}_multi_horizon_model.pkl")
            print(f"   - {stock_symbol}_scaler.pkl")
            print(f"   - {stock_symbol}_multi_horizon_evaluation.png")
            print(f"   - {stock_symbol}_multi_horizon_predictions.csv")
            print(f"   - {stock_symbol}_multi_horizon_future.png")
        else:
            print(f"   - {stock_symbol}_model.pkl")
            print(f"   - {stock_symbol}_scaler.pkl")
            print(f"   - {stock_symbol}_evaluation.png")
            print(f"   - {stock_symbol}_future_predictions.csv")

        print(f"\n⏱️  Timing breakdown for {stock_symbol}:")
        if fetch_data:
            print(f"   - Fetch data:  {timing['fetch']:>7.2f}s")
        print(f"   - Training:    {timing['train']:>7.2f}s")
        print(f"   - Evaluation:  {timing['evaluate']:>7.2f}s")
        print(f"   - Prediction:  {timing['predict']:>7.2f}s")
        print(f"   - TOTAL:       {timing['total']:>7.2f}s")
        print(f"{'='*60}")
        return True, timing

    except Exception as e:
        timing["total"] = time.time() - total_start
        print(f"\n❌ Error training {stock_symbol}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, timing


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train stock prediction models locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_local.py                  # Train VCB and FPT (recursive model)
  python train_local.py VCB              # Train only VCB
  python train_local.py VCB FPT VNM      # Train multiple stocks
  python train_local.py --all            # Train all VN30 stocks
  python train_local.py VCB --no-fetch   # Skip data fetching
  python train_local.py VCB --continue   # Continue training existing model
  
Multi-Horizon Mode (direct prediction for 1d, 3d, 7d, 15d, 30d):
  python train_local.py --multi-horizon          # Train with multi-horizon model
  python train_local.py VCB --multi-horizon      # Single stock, multi-horizon
  python train_local.py --all --multi-horizon    # All stocks, multi-horizon
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
    parser.add_argument(
        "--multi-horizon",
        action="store_true",
        help=f"Use direct multi-horizon model (horizons: {PREDICTION_HORIZONS} days)",
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

    # Model type description
    if args.multi_horizon:
        model_desc = f"Multi-Horizon Direct (horizons: {PREDICTION_HORIZONS})"
    else:
        model_desc = "Recursive (1-step, chained for multi-day)"

    print("\n" + "=" * 60)
    print("STOCK PREDICTION MODEL TRAINING")
    print("=" * 60)
    print(f"Stocks to train: {', '.join(stocks)}")
    print(f"Model type: {model_desc}")
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
    all_timings = {}
    total_start = time.time()

    for i, stock in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] Processing {stock}...")

        success, timing = train_single_stock(
            stock,
            fetch_data=not args.no_fetch,
            continue_training=args.continue_training,
            multi_horizon=args.multi_horizon,
        )
        all_timings[stock] = timing

        if success:
            success_count += 1
        else:
            failed_stocks.append(stock)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model type: {model_desc}")
    print(f"Total stocks: {len(stocks)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {len(failed_stocks)}")
    if failed_stocks:
        print(f"Failed stocks: {', '.join(failed_stocks)}")

    # Timing summary
    print(f"\n⏱️  TIMING SUMMARY")
    print("-" * 40)

    # Per-stock timing
    total_fetch = sum(t["fetch"] for t in all_timings.values())
    total_train = sum(t["train"] for t in all_timings.values())
    total_eval = sum(t["evaluate"] for t in all_timings.values())
    total_pred = sum(t["predict"] for t in all_timings.values())

    print(
        f"{'Stock':<10} {'Fetch':>8} {'Train':>8} {'Eval':>8} {'Predict':>8} {'Total':>8}"
    )
    print("-" * 60)
    for stock, timing in all_timings.items():
        status = "✅" if stock not in failed_stocks else "❌"
        print(
            f"{status} {stock:<8} "
            f"{timing['fetch']:>7.1f}s "
            f"{timing['train']:>7.1f}s "
            f"{timing['evaluate']:>7.1f}s "
            f"{timing['predict']:>7.1f}s "
            f"{timing['total']:>7.1f}s"
        )
    print("-" * 60)
    print(
        f"{'TOTAL':<10} "
        f"{total_fetch:>7.1f}s "
        f"{total_train:>7.1f}s "
        f"{total_eval:>7.1f}s "
        f"{total_pred:>7.1f}s "
        f"{total_time:>7.1f}s"
    )

    # Average per stock
    if success_count > 0:
        avg_time = (
            sum(all_timings[s]["total"] for s in all_timings if s not in failed_stocks)
            / success_count
        )
        print(f"\nAverage time per successful stock: {avg_time:.1f}s")

    print("=" * 60)

    return 0 if success_count == len(stocks) else 1


if __name__ == "__main__":
    sys.exit(main())
