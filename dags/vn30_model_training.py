"""
VN30 Model Training & Prediction DAG
Train ensemble models and generate predictions

Schedule: Daily at 6:00 PM Vietnam Time (after data crawler)
Duration: 2-3 hours
Purpose: Train models on latest data and predict 30-day future prices

This DAG reads:
- DAG settings (schedule, retries, etc.) from pipeline_dags table
- Training config (models, indicators, etc.) from training_configs table

Edit DAG settings via UI (Pipelines > Edit DAG).
Edit training config via UI (Training page).
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
# ExternalTaskSensor removed - data is fetched on-demand from DB

# Import from new backend structure
from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.db.models import TrainingConfig, PipelineDAG
from app.schemas.training import TrainingConfigSchema
from app.services.model_trainer import (
    train_prediction_model,
    evaluate_model,
    predict_future_prices,
)
from app.services.email_service import send_email_notification
from sqlalchemy import select

logger = get_logger(__name__)

DAG_ID = "vn30_model_training"

# Default DAG settings (used if not found in DB)
DEFAULT_DAG_SETTINGS = {
    "schedule": "0 18 * * *",
    "catchup": False,
    "max_active_runs": 1,
    "retries": 1,
    "retry_delay_minutes": 10,
    "owner": "ml_engineering",
    "tags": ["vn30", "ml", "ensemble", "training", "prediction"],
    "timezone": "Asia/Ho_Chi_Minh",
}


def get_dag_settings_from_db(dag_id: str) -> dict:
    """
    Get DAG settings from pipeline_dags table.
    Falls back to defaults if not found.
    """
    try:
        db = SessionLocal()
        try:
            stmt = select(PipelineDAG).where(PipelineDAG.dag_id == dag_id)
            dag_record = db.execute(stmt).scalar_one_or_none()
            
            if dag_record:
                logger.info(f"Loaded DAG settings from DB for {dag_id}")
                return {
                    "schedule": dag_record.schedule_cron or DEFAULT_DAG_SETTINGS["schedule"],
                    "catchup": dag_record.catchup if dag_record.catchup is not None else DEFAULT_DAG_SETTINGS["catchup"],
                    "max_active_runs": dag_record.max_active_runs or DEFAULT_DAG_SETTINGS["max_active_runs"],
                    "retries": dag_record.default_retries if dag_record.default_retries is not None else DEFAULT_DAG_SETTINGS["retries"],
                    "retry_delay_minutes": dag_record.default_retry_delay_minutes or DEFAULT_DAG_SETTINGS["retry_delay_minutes"],
                    "owner": dag_record.default_owner or DEFAULT_DAG_SETTINGS["owner"],
                    "tags": dag_record.default_tags or DEFAULT_DAG_SETTINGS["tags"],
                    "timezone": dag_record.timezone or DEFAULT_DAG_SETTINGS["timezone"],
                }
            else:
                logger.info(f"No DB settings found for {dag_id}, using defaults")
                return DEFAULT_DAG_SETTINGS
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Error loading DAG settings from DB: {e}, using defaults")
        return DEFAULT_DAG_SETTINGS


def get_latest_training_config():
    """
    Get the latest active TrainingConfig from database.
    
    Returns:
        Tuple of (stock_symbols, models_config) or (default_stocks, None)
    """
    db = SessionLocal()
    try:
        # Get the most recent active config
        stmt = (
            select(TrainingConfig)
            .where(TrainingConfig.is_active == True)
            .order_by(TrainingConfig.updated_at.desc())
            .limit(1)
        )
        config_record = db.execute(stmt).scalar_one_or_none()
        
        if config_record and config_record.config:
            config = config_record.config
            logger.info(f"Using TrainingConfig: {config_record.name} (v{config_record.version})")
            
            # Parse config to get models settings
            try:
                training_config = TrainingConfigSchema(**config)
                models_config = training_config.models
                logger.info(
                    f"Models config: RF={models_config.random_forest.enabled}, "
                    f"GB={models_config.gradient_boosting.enabled}, "
                    f"SVR={models_config.svr.enabled}, "
                    f"Ridge={models_config.ridge.enabled}"
                )
            except Exception as e:
                logger.warning(f"Could not parse models config, using defaults: {e}")
                models_config = None
            
            # Determine stocks to process
            if config.get("universe", {}).get("useAllVn30"):
                stock_symbols = settings.vn30_stocks
            else:
                stock_symbols = config.get("universe", {}).get("tickers", [])
            
            if not stock_symbols:
                logger.warning("No stocks in config, falling back to VN30 list")
                stock_symbols = settings.vn30_stocks
            
            return stock_symbols, models_config
        else:
            logger.info("No TrainingConfig found, using default VN30 stocks")
            return settings.vn30_stocks, None
            
    except Exception as e:
        logger.error(f"Error loading TrainingConfig: {e}")
        return settings.vn30_stocks, None
    finally:
        db.close()


def train_stock_task(stock_symbol, **context):
    """Train model for a single stock using config from DB."""
    ti = context['ti']
    
    # Check if this stock is in the config's stock list
    stock_symbols = ti.xcom_pull(task_ids='load_training_config', key='stock_symbols') or []
    if stock_symbols and stock_symbol not in stock_symbols:
        logger.info(f"Skipping {stock_symbol} - not in training config ({len(stock_symbols)} stocks configured)")
        return {"status": "skipped", "stock": stock_symbol, "reason": "not in config"}
    
    db = SessionLocal()
    try:
        # Get config from XCom (pushed by load_config_task)
        config_data = ti.xcom_pull(task_ids='load_training_config', key='models_config')
        
        # Reconstruct models_config if available
        models_config = None
        if config_data:
            try:
                from app.schemas.training import ModelsConfig
                models_config = ModelsConfig(**config_data)
            except Exception as e:
                logger.warning(f"Could not parse models_config from XCom: {e}")
        
        return train_prediction_model(db, stock_symbol, models_config=models_config)
    finally:
        db.close()


def evaluate_stock_task(stock_symbol, **context):
    """Evaluate model for a single stock."""
    ti = context['ti']
    stock_symbols = ti.xcom_pull(task_ids='load_training_config', key='stock_symbols') or []
    if stock_symbols and stock_symbol not in stock_symbols:
        logger.info(f"Skipping evaluate {stock_symbol} - not in training config")
        return {"status": "skipped", "stock": stock_symbol}
    
    db = SessionLocal()
    try:
        return evaluate_model(db, stock_symbol)
    finally:
        db.close()


def predict_stock_task(stock_symbol, **context):
    """Generate future predictions for a single stock."""
    ti = context['ti']
    stock_symbols = ti.xcom_pull(task_ids='load_training_config', key='stock_symbols') or []
    if stock_symbols and stock_symbol not in stock_symbols:
        logger.info(f"Skipping predict {stock_symbol} - not in training config")
        return {"status": "skipped", "stock": stock_symbol}
    
    db = SessionLocal()
    try:
        return predict_future_prices(db, stock_symbol)
    finally:
        db.close()


def load_config_task(**context):
    """
    Load training config from DB and push to XCom for other tasks.
    Returns list of stock symbols to process.
    """
    stock_symbols, models_config = get_latest_training_config()
    
    # Push models_config to XCom for train tasks
    ti = context['ti']
    if models_config:
        # Convert to dict for XCom serialization
        ti.xcom_push(key='models_config', value=models_config.model_dump())
    else:
        ti.xcom_push(key='models_config', value=None)
    
    ti.xcom_push(key='stock_symbols', value=stock_symbols)
    
    logger.info(f"Loaded config for {len(stock_symbols)} stocks")
    return stock_symbols


def get_stock_symbols(**context):
    """Get stock symbols from XCom."""
    ti = context['ti']
    return ti.xcom_pull(task_ids='load_training_config', key='stock_symbols') or settings.vn30_stocks


def send_email_task(**context):
    """Send summary email report."""
    stock_symbols = get_stock_symbols(**context)
    return send_email_notification(stock_symbols)


# Load DAG settings from DB
dag_settings = get_dag_settings_from_db(DAG_ID)

# Vietnam timezone from DB settings
local_tz = pendulum.timezone(dag_settings["timezone"])

# Default arguments from DB settings
default_args = {
    'owner': dag_settings["owner"],
    'email': [settings.email_recipient],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': dag_settings["retries"],
    'retry_delay': timedelta(minutes=dag_settings["retry_delay_minutes"]),
}

# Define DAG with settings from DB
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Ensemble model training using config from database (DAG settings + TrainingConfig)',
    schedule=dag_settings["schedule"],
    start_date=pendulum.datetime(2025, 10, 1, tz=local_tz),
    catchup=dag_settings["catchup"],
    tags=dag_settings["tags"],
    max_active_runs=dag_settings["max_active_runs"],
) as dag:
    
    # Task 1: Load training config from DB
    # Note: Data is fetched on-demand from DB, no need to wait for crawler
    load_config = PythonOperator(
        task_id='load_training_config',
        python_callable=load_config_task,
    )
    
    # Tasks: Train, Evaluate, Predict for each stock
    # Note: We use settings.vn30_stocks for task generation, but actual processing
    # uses config from DB (via XCom). Unused stock tasks will complete quickly.
    all_predict_tasks = []
    
    for stock in settings.vn30_stocks:
        # Train - now uses models_config from XCom
        train_task = PythonOperator(
            task_id=f'train_{stock}',
            python_callable=train_stock_task,
            op_kwargs={'stock_symbol': stock},
        )
        
        # Evaluate
        evaluate_task = PythonOperator(
            task_id=f'evaluate_{stock}',
            python_callable=evaluate_stock_task,
            op_kwargs={'stock_symbol': stock},
        )
        
        # Predict
        predict_task = PythonOperator(
            task_id=f'predict_{stock}',
            python_callable=predict_stock_task,
            op_kwargs={'stock_symbol': stock},
        )
        
        # Chain: LoadConfig -> Train -> Evaluate -> Predict
        load_config >> train_task >> evaluate_task >> predict_task
        all_predict_tasks.append(predict_task)
    
    # Final task: Send email after all predictions complete
    send_email = PythonOperator(
        task_id='send_email_report',
        python_callable=send_email_task,
    )
    
    # All predictions must complete before email
    all_predict_tasks >> send_email
