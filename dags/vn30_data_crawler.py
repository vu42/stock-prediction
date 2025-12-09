"""
VN30 Data Crawler DAG
Incremental data crawling from VNDirect API to PostgreSQL

Schedule: Daily at 5:00 PM Vietnam Time (UTC+7)
Duration: 5-10 minutes
Purpose: Fetch only new data since last crawl (incremental updates)
"""

import sys
import os

# Add backend src to path for imports
backend_src = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend",
    "src",
)
sys.path.insert(0, backend_src)

import pendulum
from datetime import timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# Import from new backend structure
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.data_fetcher import fetch_stock_data


def init_db_task():
    """Initialize PostgreSQL database tables if not exist."""
    # With SQLAlchemy, tables are created via Alembic migrations
    # This task can be used for any startup checks
    from app.db.base import Base
    from app.db.session import engine

    # Create tables if they don't exist (for development)
    # In production, use Alembic migrations
    Base.metadata.create_all(bind=engine)
    print("Database tables verified/created")


def crawl_stock_task(stock_symbol, **context):
    """
    Crawl data for a single stock (incremental).

    Args:
        stock_symbol: Stock symbol to crawl
        context: Airflow context with execution date
    """
    to_date = context.get("ds")  # Airflow execution date

    db = SessionLocal()
    try:
        return fetch_stock_data(db, stock_symbol, to_date=to_date)
    finally:
        db.close()


# Vietnam timezone
local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

# Default arguments
default_args = {
    "owner": "data_engineering",
    "email": [settings.email_recipient],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
with DAG(
    dag_id="vn30_data_crawler",
    default_args=default_args,
    description=f"Incremental data crawler for {len(settings.vn30_stocks)} VN30 stocks from VNDirect API",
    schedule="0 17 * * *",
    start_date=pendulum.datetime(2025, 10, 1, tz=local_tz),
    catchup=False,
    tags=["vn30", "data-crawler", "incremental", "postgres"],
    max_active_runs=1,
) as dag:

    # Task 1: Initialize database
    init_db = PythonOperator(
        task_id="init_database",
        python_callable=init_db_task,
    )

    # Task 2: Crawl each stock in parallel
    crawl_tasks = []
    for stock in settings.vn30_stocks:
        task = PythonOperator(
            task_id=f"crawl_{stock}",
            python_callable=crawl_stock_task,
            op_kwargs={"stock_symbol": stock},
        )
        crawl_tasks.append(task)

    # Dependencies: Init DB first, then crawl all stocks in parallel
    init_db >> crawl_tasks
