"""
Orchestrator module - Coordinates the entire pipeline
"""
from modules.data_fetcher import fetch_stock_data
from modules.model_trainer import train_prediction_model, evaluate_model, predict_future_prices


def process_all_stocks(stock_symbols, **context):
    """
    Process all stocks: fetch → train → evaluate → predict.
    
    Args:
        stock_symbols: List of stock symbols to process
        context: Airflow context
        
    Returns:
        dict: Results with success/failed stocks
    """
    results = {
        'success': [],
        'failed': []
    }
    
    total = len(stock_symbols)
    print(f"\n{'='*80}")
    print(f"Processing {total} VN30 stocks")
    print(f"{'='*80}\n")
    
    for idx, stock_symbol in enumerate(stock_symbols, 1):
        print(f"\n--- [{idx}/{total}] {stock_symbol} ---")
        
        try:
            # Step 1: Fetch data (incremental)
            if not fetch_stock_data(stock_symbol, **context):
                print(f"[{stock_symbol}] WARNING: Skipping - fetch failed")
                results['failed'].append(stock_symbol)
                continue
            
            # Step 2: Train model
            if not train_prediction_model(stock_symbol):
                print(f"[{stock_symbol}] WARNING: Skipping - training failed")
                results['failed'].append(stock_symbol)
                continue
            
            # Step 3: Evaluate model
            if not evaluate_model(stock_symbol):
                print(f"[{stock_symbol}] WARNING: Evaluation failed (non-critical)")
            
            # Step 4: Predict future
            if not predict_future_prices(stock_symbol):
                print(f"[{stock_symbol}] WARNING: Prediction failed (non-critical)")
            
            results['success'].append(stock_symbol)
            print(f"[{stock_symbol}] Completed successfully")
            
        except Exception as e:
            print(f"[{stock_symbol}] ERROR: {str(e)}")
            results['failed'].append(stock_symbol)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"   Success: {len(results['success'])}/{total}")
    print(f"   Failed: {len(results['failed'])}/{total}")
    if results['failed']:
        print(f"   Failed stocks: {', '.join(results['failed'])}")
    print(f"{'='*80}\n")
    
    return results

