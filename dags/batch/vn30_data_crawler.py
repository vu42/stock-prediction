"""
VN30 Data Crawler DAG
Incremental data crawling from VNDirect API to PostgreSQL

Schedule: Daily at 5:00 PM Vietnam Time (UTC+7)
Duration: 5-10 minutes
Purpose: Fetch only new data since last crawl (incremental updates)
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = '/Users/aphan/Learning/stock_data/stock-prediction'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pendulum
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from config import STOCK_SYMBOLS
from modules.database import init_database, insert_stock_data
import pandas as pd
import random
from datetime import datetime, timedelta


def init_db_task():
    """Initialize PostgreSQL database tables if not exist."""
    init_database()


def crawl_stock_task(stock_symbol, **context):
    """
    Generate mock data and insert into DB (for demo).
    
    Args:
        stock_symbol: Stock symbol to crawl
        context: Airflow context with execution date
    """
    print(f"[CRAWLER] Generating mock data for {stock_symbol}...")
    
    # Generate 100 days of fake OHLCV data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    base_price = random.uniform(10000, 100000)
    
    data = []
    for date in dates:
        # Random price movement
        open_price = base_price * random.uniform(0.95, 1.05)
        high_price = open_price * random.uniform(1.0, 1.03)
        low_price = open_price * random.uniform(0.97, 1.0)
        close_price = random.uniform(low_price, high_price)
        volume = random.randint(100000, 5000000)
        
        data.append({
            'stock_symbol': stock_symbol,
            'date': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        base_price = close_price  # Next day starts from previous close
    
    # Create DataFrame and insert to DB
    df = pd.DataFrame(data)
    insert_stock_data(df, stock_symbol)
    
    print(f"[CRAWLER] ✓ {stock_symbol}: Generated and inserted {len(data)} records")
    return f"{stock_symbol}: SUCCESS ({len(data)} records inserted)"


# Vietnam timezone
local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

# Default arguments
default_args = {
    'owner': 'data_engineering',
    'email': ['tuph.alex@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
with DAG(
    dag_id='vn30_data_crawler',
    default_args=default_args,
    description=f'Incremental data crawler for {len(STOCK_SYMBOLS)} VN30 stocks from VNDirect API',
    # Production schedule: Daily at 5:00 PM Vietnam Time (UTC+7)
    # schedule_interval='0 17 * * *',  # Uncomment for production
    schedule_interval=None,  # Manual trigger for demo
    start_date=pendulum.datetime(2025, 10, 1, tz=local_tz),
    catchup=False,
    tags=['vn30', 'data-crawler', 'incremental', 'postgres', 'batch'],
    max_active_runs=1,
) as dag:
    
    # Task 1: Initialize database
    init_db = PythonOperator(
        task_id='init_database',
        python_callable=init_db_task,
    )
    
    # Task 2: Crawl each stock in parallel
    crawl_tasks = []
    for stock in STOCK_SYMBOLS:
        task = PythonOperator(
            task_id=f'crawl_{stock}',
            python_callable=crawl_stock_task,
            op_kwargs={'stock_symbol': stock, 'to_date': "{{ ds }}"},
        )
        crawl_tasks.append(task)
    
    # Dependencies: Init DB first, then crawl all stocks in parallel
    init_db >> crawl_tasks

