"""
VN30 Data Crawler DAG
Incremental data crawling from VNDirect API to PostgreSQL

Schedule: Daily at 5:00 PM Vietnam Time (UTC+7)
Duration: 5-10 minutes
Purpose: Fetch only new data since last crawl (incremental updates)

DAG settings are read from pipeline_dags table in database.
Edit settings via UI (Pipelines > Edit DAG) and they will apply on next DAG refresh.
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
from airflow.operators.python import PythonOperator

# Import from new backend structure
from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.db.models import PipelineDAG
from app.services.data_fetcher import fetch_stock_data
from sqlalchemy import select

logger = get_logger(__name__)

DAG_ID = "vn30_data_crawler"

# Default settings (used if not found in DB)
DEFAULT_SETTINGS = {
    "schedule": "0 17 * * *",
    "catchup": False,
    "max_active_runs": 1,
    "retries": 2,
    "retry_delay_minutes": 5,
    "owner": "data_engineering",
    "tags": ["vn30", "data-crawler", "incremental", "postgres"],
    "timezone": "Asia/Ho_Chi_Minh",
}


def get_dag_settings_from_db(dag_id: str) -> dict:
    """
    Get DAG settings from pipeline_dags table.
    Falls back to defaults if not found.
    
    Args:
        dag_id: The DAG identifier
        
    Returns:
        Dict with DAG settings
    """
    try:
        db = SessionLocal()
        try:
            stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
            dag_record = db.execute(stmt).scalar_one_or_none()
            
            if dag_record:
                logger.info(f"Loaded DAG settings from DB for {dag_id}")
                return {
                    "schedule": dag_record.schedule_cron or DEFAULT_SETTINGS["schedule"],
                    "catchup": dag_record.catchup if dag_record.catchup is not None else DEFAULT_SETTINGS["catchup"],
                    "max_active_runs": dag_record.max_active_runs or DEFAULT_SETTINGS["max_active_runs"],
                    "retries": dag_record.default_retries if dag_record.default_retries is not None else DEFAULT_SETTINGS["retries"],
                    "retry_delay_minutes": dag_record.default_retry_delay_minutes or DEFAULT_SETTINGS["retry_delay_minutes"],
                    "owner": dag_record.default_owner or DEFAULT_SETTINGS["owner"],
                    "tags": dag_record.default_tags or DEFAULT_SETTINGS["tags"],
                    "timezone": dag_record.timezone or DEFAULT_SETTINGS["timezone"],
                }
            else:
                logger.info(f"No DB settings found for {dag_id}, using defaults")
                return DEFAULT_SETTINGS
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Error loading DAG settings from DB: {e}, using defaults")
        return DEFAULT_SETTINGS


def crawl_stock_task(stock_symbol, **context):
    """
    Crawl data for a single stock (incremental).

    Args:
        stock_symbol: Stock symbol to crawl
        context: Airflow context with execution date
    """
    to_date = context.get("ds")

    db = SessionLocal()
    try:
        return fetch_stock_data(db, stock_symbol, to_date=to_date)
    finally:
        db.close()


# Load settings from DB
dag_settings = get_dag_settings_from_db(DAG_ID)

# Vietnam timezone
local_tz = pendulum.timezone(dag_settings["timezone"])

# Default arguments from DB settings
default_args = {
    "owner": dag_settings["owner"],
    "email": [settings.email_recipient],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": dag_settings["retries"],
    "retry_delay": timedelta(minutes=dag_settings["retry_delay_minutes"]),
}

# Define DAG with settings from DB
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description=f"Incremental data crawler for VN30 stocks from VNDirect API (DAG settings from database)",
    schedule=dag_settings["schedule"],
    start_date=pendulum.datetime(2025, 10, 1, tz=local_tz),
    catchup=dag_settings["catchup"],
    tags=dag_settings["tags"],
    max_active_runs=dag_settings["max_active_runs"],
) as dag:

    # Crawl each stock in parallel
    for stock in settings.vn30_stocks:
        PythonOperator(
            task_id=f"crawl_{stock}",
            python_callable=crawl_stock_task,
            op_kwargs={"stock_symbol": stock},
        )
