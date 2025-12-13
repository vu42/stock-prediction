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
from uuid import uuid4
from airflow import DAG
from airflow.operators.python import PythonOperator
# ExternalTaskSensor removed - data is fetched on-demand from DB

# Import from new backend structure
from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.db.models import TrainingConfig, PipelineDAG, Stock, ExperimentTickerArtifact
from app.schemas.training import TrainingConfigSchema
from app.services.model_trainer import (
    train_prediction_model,
    evaluate_model,
    predict_future_prices,
    get_stock_file_paths,
)
from app.services.email_service import send_email_notification
from app.integrations.storage_s3 import storage_client, upload_model_artifact
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


def ensure_bucket_task(**context):
    """
    Ensure MinIO/S3 bucket exists before uploading artifacts.
    Creates the bucket if it doesn't exist.
    """
    try:
        storage_client.ensure_bucket_exists()
        logger.info(f"Bucket verified/created: {settings.s3_bucket_name}")
        return {"status": "success", "bucket": settings.s3_bucket_name}
    except Exception as e:
        logger.error(f"Failed to ensure bucket exists: {e}")
        # Don't fail the DAG, just log the error - uploads will fail individually
        return {"status": "error", "error": str(e)}


def upload_artifacts_task(stock_symbol, **context):
    """
    Upload model artifacts to MinIO/S3 for a single stock and store URLs in database.
    
    Uploads:
    - evaluation_png: Model evaluation plot
    - future_png: Future predictions plot  
    - model_pkl: Trained model pickle
    - scaler_pkl: Data scaler pickle
    - future_predictions_csv: Future predictions data
    
    Then creates an ExperimentTickerArtifact record with the URLs so they
    can be displayed in the Models page.
    """
    ti = context['ti']
    stock_symbols = ti.xcom_pull(task_ids='load_training_config', key='stock_symbols') or []
    if stock_symbols and stock_symbol not in stock_symbols:
        logger.info(f"Skipping upload artifacts {stock_symbol} - not in training config")
        return {"status": "skipped", "stock": stock_symbol}
    
    # Generate a unique run ID for this DAG run (use Airflow run_id)
    dag_run_id = context.get('run_id', str(uuid4()))
    
    # Get file paths for this stock's artifacts
    paths = get_stock_file_paths(stock_symbol)
    
    artifact_files = [
        ("evaluation_png", paths["plot"], "image/png"),
        ("future_png", paths["future_plot"], "image/png"),
        ("model_pkl", paths["model"], "application/octet-stream"),
        ("scaler_pkl", paths["scaler"], "application/octet-stream"),
        ("future_predictions_csv", paths["future_csv"], "text/csv"),
    ]
    
    uploaded_urls = {}
    
    for artifact_type, file_path, content_type in artifact_files:
        if os.path.exists(file_path):
            try:
                url = upload_model_artifact(stock_symbol, dag_run_id, artifact_type, file_path)
                uploaded_urls[artifact_type] = url
                logger.info(f"[{stock_symbol}] Uploaded {artifact_type}: {url}")
            except Exception as e:
                logger.warning(f"[{stock_symbol}] Failed to upload {artifact_type}: {e}")
        else:
            logger.debug(f"[{stock_symbol}] Artifact not found: {file_path}")
    
    # Store artifact URLs in database (same as run_training_experiment)
    # Use UPSERT logic: update existing record if it exists, otherwise create new
    if uploaded_urls:
        db = SessionLocal()
        try:
            # Get stock from DB
            stock_stmt = select(Stock).where(Stock.ticker == stock_symbol)
            stock = db.execute(stock_stmt).scalar_one_or_none()
            
            if stock:
                # Check if artifact record already exists for this stock (with null run_id)
                existing_stmt = (
                    select(ExperimentTickerArtifact)
                    .where(ExperimentTickerArtifact.stock_id == stock.id)
                    .where(ExperimentTickerArtifact.run_id.is_(None))
                )
                existing_artifact = db.execute(existing_stmt).scalar_one_or_none()
                
                if existing_artifact:
                    # Update existing record
                    existing_artifact.metrics = {"status": "completed", "source": "airflow_dag"}
                    existing_artifact.evaluation_png_url = uploaded_urls.get("evaluation_png")
                    existing_artifact.future_png_url = uploaded_urls.get("future_png")
                    existing_artifact.model_pkl_url = uploaded_urls.get("model_pkl")
                    existing_artifact.scaler_pkl_url = uploaded_urls.get("scaler_pkl")
                    existing_artifact.future_predictions_csv = uploaded_urls.get("future_predictions_csv")
                    db.commit()
                    logger.info(f"[{stock_symbol}] Updated existing artifact URLs in database")
                else:
                    # Create new record
                    artifact = ExperimentTickerArtifact(
                        stock_id=stock.id,
                        metrics={"status": "completed", "source": "airflow_dag"},
                        evaluation_png_url=uploaded_urls.get("evaluation_png"),
                        future_png_url=uploaded_urls.get("future_png"),
                        model_pkl_url=uploaded_urls.get("model_pkl"),
                        scaler_pkl_url=uploaded_urls.get("scaler_pkl"),
                        future_predictions_csv=uploaded_urls.get("future_predictions_csv"),
                    )
                    db.add(artifact)
                    db.commit()
                    logger.info(f"[{stock_symbol}] Created new artifact record in database")
            else:
                logger.warning(f"[{stock_symbol}] Stock not found in database, skipping artifact record")
        except Exception as e:
            logger.error(f"[{stock_symbol}] Failed to store artifact record: {e}")
            db.rollback()
        finally:
            db.close()
    
    return {
        "status": "success",
        "stock": stock_symbol,
        "artifacts": uploaded_urls,
        "run_id": dag_run_id,
    }


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
    
    # Task 2: Ensure MinIO/S3 bucket exists before uploading artifacts
    ensure_bucket = PythonOperator(
        task_id='ensure_bucket_exists',
        python_callable=ensure_bucket_task,
    )
    
    # Tasks: Train, Evaluate, Predict, Upload for each stock
    # Note: We use settings.vn30_stocks for task generation, but actual processing
    # uses config from DB (via XCom). Unused stock tasks will complete quickly.
    all_upload_tasks = []
    
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
        
        # Upload artifacts to MinIO/S3
        upload_task = PythonOperator(
            task_id=f'upload_artifacts_{stock}',
            python_callable=upload_artifacts_task,
            op_kwargs={'stock_symbol': stock},
        )
        
        # Chain: LoadConfig -> Train -> Evaluate -> Predict -> Upload
        # ensure_bucket runs in parallel with load_config, but before uploads
        load_config >> train_task >> evaluate_task >> predict_task
        [predict_task, ensure_bucket] >> upload_task
        all_upload_tasks.append(upload_task)
    
    # Final task: Send email after all uploads complete
    send_email = PythonOperator(
        task_id='send_email_report',
        python_callable=send_email_task,
    )
    
    # All uploads must complete before email
    all_upload_tasks >> send_email
