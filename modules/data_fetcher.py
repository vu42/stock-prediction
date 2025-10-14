"""
Data fetching module - Handles API crawling with incremental updates
"""
import os
import ssl
import json
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
import pandas as pd

from config import API_BASE_URL, DATA_START_DATE, BASE_OUTPUT_DIR
from modules.database import get_last_data_date, insert_stock_data, load_stock_data_from_db

# SSL configuration
ssl._create_default_https_context = ssl._create_unverified_context


def fetch_stock_data(stock_symbol, **context):
    """
    Fetch stock data from VNDirect API with INCREMENTAL CRAWLING.
    
    Logic:
    - Check last date in database
    - If exists: Only fetch from (last_date + 1) to today
    - If not exists: Fetch all historical data from DATA_START_DATE
    - Insert into PostgreSQL with UPSERT
    - Save backup CSV
    
    Args:
        stock_symbol: Stock symbol to fetch (e.g. "VCB", "FPT")
        context: Airflow context (contains execution date)
        
    Returns:
        bool: True if successful, False otherwise
    """
    end_date = context.get("to_date", datetime.now().strftime("%Y-%m-%d"))
    
    try:
        # Check last date in database for incremental crawling
        last_date = get_last_data_date(stock_symbol)
        
        if last_date:
            # Incremental: Only fetch from day after last date
            start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"[{stock_symbol}] Incremental crawl from {start_date} to {end_date}")
            
            # Already up-to-date
            if start_date > end_date:
                print(f"[{stock_symbol}] Already up-to-date (last: {last_date})")
                return True
        else:
            # First time: fetch all historical data
            start_date = DATA_START_DATE
            print(f"[{stock_symbol}] Initial crawl from {start_date} to {end_date}")
        
        # Build API request URL
        query_params = f"sort=date&q=code:{stock_symbol}~date:gte:{start_date}~date:lte:{end_date}&size=9990&page=1"
        api_url = f"{API_BASE_URL}?{query_params}"
        
        # Fetch data from API
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        request = Request(api_url, headers=headers)
        response_data = urlopen(request, timeout=10).read()
        parsed_data = json.loads(response_data)['data']
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_data)
        
        if len(df) == 0:
            print(f"[{stock_symbol}] No new data (market closed or weekend)")
            return True  # Not an error
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Insert into database (with UPSERT on conflict)
        inserted_count = insert_stock_data(df, stock_symbol)
        print(f"[{stock_symbol}] Inserted/Updated {inserted_count} records in database")
        
        # Save backup CSV (optional but recommended)
        _save_backup_csv(stock_symbol)
        
        return True
        
    except Exception as e:
        print(f"[{stock_symbol}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def _save_backup_csv(stock_symbol):
    """
    Save backup CSV from database data.
    
    Args:
        stock_symbol: Stock symbol
    """
    try:
        output_dir = os.path.join(BASE_OUTPUT_DIR, stock_symbol)
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, f"{stock_symbol}_price.csv")
        
        # Load full data from DB and save
        full_df = load_stock_data_from_db(stock_symbol)
        full_df.to_csv(csv_path, index=False)
        print(f"[{stock_symbol}] Backup CSV: {len(full_df)} total records saved to {csv_path}")
        
    except Exception as e:
        print(f"[{stock_symbol}] Warning: Could not save backup CSV: {str(e)}")


def fetch_all_stocks(stock_symbols, **context):
    """
    Fetch data for multiple stocks.
    
    Args:
        stock_symbols: List of stock symbols
        context: Airflow context
        
    Returns:
        dict: Results with success/failed lists
    """
    results = {
        'success': [],
        'failed': []
    }
    
    for stock in stock_symbols:
        if fetch_stock_data(stock, **context):
            results['success'].append(stock)
        else:
            results['failed'].append(stock)
    
    return results

