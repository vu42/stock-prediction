"""
Experiment training tasks for RQ worker.
"""

import os
import sys
from datetime import datetime
from uuid import UUID

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import (
    ExperimentLog,
    ExperimentRun,
    ExperimentState,
    ExperimentTickerArtifact,
    Stock,
)
from app.db.session import SessionLocal
from app.integrations.storage_s3 import upload_model_artifact
from app.services.data_fetcher import fetch_stock_data, load_stock_data_from_db
from app.services.model_trainer import (
    evaluate_model,
    get_stock_file_paths,
    predict_future_prices,
    train_prediction_model,
)

logger = get_logger(__name__)


def log_to_experiment(db: Session, run_id: str, level: str, message: str):
    """Log a message to the experiment logs table."""
    log_entry = ExperimentLog(
        run_id=UUID(run_id),
        level=level,
        message=message,
    )
    db.add(log_entry)
    db.commit()


def run_training_experiment(run_id: str) -> dict:
    """
    Execute a training experiment.
    
    This is the main task that gets queued via RQ.
    
    Args:
        run_id: UUID of the experiment run
        
    Returns:
        Dict with results summary
    """
    db = SessionLocal()
    
    try:
        # Get the experiment run
        stmt = select(ExperimentRun).where(ExperimentRun.id == UUID(run_id))
        run = db.execute(stmt).scalar_one_or_none()
        
        if not run:
            logger.error(f"Experiment run not found: {run_id}")
            return {"error": "Run not found"}
        
        # Update state to running
        run.state = ExperimentState.RUNNING.value
        run.started_at = datetime.now()
        db.commit()
        
        log_to_experiment(db, run_id, "INFO", "Starting training experiment")
        
        # Get config
        config = run.config.config
        
        # Determine stocks to process
        if config.get("universe", {}).get("useAllVn30"):
            stock_symbols = settings.vn30_stocks
        else:
            stock_symbols = config.get("universe", {}).get("tickers", [])
        
        if not stock_symbols:
            raise ValueError("No stocks to process")
        
        log_to_experiment(
            db, run_id, "INFO",
            f"Processing {len(stock_symbols)} stocks: {', '.join(stock_symbols[:5])}..."
        )
        
        results = {
            "success": [],
            "failed": [],
            "artifacts": [],
        }
        
        total_stocks = len(stock_symbols)
        
        for idx, stock_symbol in enumerate(stock_symbols):
            try:
                progress = (idx / total_stocks) * 100
                run.progress_pct = progress
                db.commit()
                
                log_to_experiment(
                    db, run_id, "INFO",
                    f"[{idx + 1}/{total_stocks}] Processing {stock_symbol}..."
                )
                
                # Step 1: Fetch data (if not skip_refetch)
                if not config.get("dataWindow", {}).get("skipRefetch", False):
                    log_to_experiment(db, run_id, "INFO", f"[{stock_symbol}] Fetching data...")
                    fetch_result = fetch_stock_data(db, stock_symbol)
                    if not fetch_result:
                        log_to_experiment(
                            db, run_id, "WARNING",
                            f"[{stock_symbol}] Data fetch failed, skipping"
                        )
                        results["failed"].append(stock_symbol)
                        continue
                
                # Step 2: Train model
                log_to_experiment(db, run_id, "INFO", f"[{stock_symbol}] Training model...")
                train_result = train_prediction_model(db, stock_symbol)
                
                if not train_result:
                    log_to_experiment(
                        db, run_id, "WARNING",
                        f"[{stock_symbol}] Training failed, skipping"
                    )
                    results["failed"].append(stock_symbol)
                    continue
                
                # Step 3: Evaluate
                log_to_experiment(db, run_id, "INFO", f"[{stock_symbol}] Evaluating model...")
                evaluate_model(db, stock_symbol)
                
                # Step 4: Predict future
                log_to_experiment(db, run_id, "INFO", f"[{stock_symbol}] Generating predictions...")
                predict_future_prices(db, stock_symbol)
                
                # Step 5: Upload artifacts to S3 (optional)
                artifact_urls = upload_artifacts(db, run_id, stock_symbol)
                
                # Store artifact record
                stock_stmt = select(Stock).where(Stock.ticker == stock_symbol)
                stock = db.execute(stock_stmt).scalar_one_or_none()
                
                if stock:
                    artifact = ExperimentTickerArtifact(
                        run_id=UUID(run_id),
                        stock_id=stock.id,
                        metrics={"status": "completed"},
                        evaluation_png_url=artifact_urls.get("evaluation_png"),
                        future_png_url=artifact_urls.get("future_png"),
                        model_pkl_url=artifact_urls.get("model_pkl"),
                        scaler_pkl_url=artifact_urls.get("scaler_pkl"),
                        future_predictions_csv=artifact_urls.get("future_predictions_csv"),
                    )
                    db.add(artifact)
                    db.commit()
                
                results["success"].append(stock_symbol)
                log_to_experiment(
                    db, run_id, "INFO",
                    f"[{stock_symbol}] Completed successfully"
                )
                
            except Exception as e:
                logger.error(f"Error processing {stock_symbol}: {e}")
                log_to_experiment(
                    db, run_id, "ERROR",
                    f"[{stock_symbol}] Error: {str(e)}"
                )
                results["failed"].append(stock_symbol)
        
        # Update run state
        run.state = ExperimentState.SUCCESS.value
        run.finished_at = datetime.now()
        run.progress_pct = 100
        run.summary_metrics = {
            "total": total_stocks,
            "success": len(results["success"]),
            "failed": len(results["failed"]),
        }
        db.commit()
        
        log_to_experiment(
            db, run_id, "INFO",
            f"Experiment completed: {len(results['success'])}/{total_stocks} successful"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment {run_id} failed: {e}")
        
        # Update run state to failed
        if run:
            run.state = ExperimentState.FAILED.value
            run.finished_at = datetime.now()
            db.commit()
            
            log_to_experiment(db, run_id, "ERROR", f"Experiment failed: {str(e)}")
        
        return {"error": str(e)}
        
    finally:
        db.close()


def upload_artifacts(db: Session, run_id: str, stock_symbol: str) -> dict[str, str]:
    """
    Upload model artifacts to S3.
    
    Args:
        db: Database session
        run_id: Experiment run ID
        stock_symbol: Stock ticker symbol
        
    Returns:
        Dict with artifact URLs
    """
    urls = {}
    paths = get_stock_file_paths(stock_symbol)
    
    artifact_files = [
        ("evaluation_png", paths["plot"], "image/png"),
        ("future_png", paths["future_plot"], "image/png"),
        ("model_pkl", paths["model"], "application/octet-stream"),
        ("scaler_pkl", paths["scaler"], "application/octet-stream"),
        ("future_predictions_csv", paths["future_csv"], "text/csv"),
    ]
    
    for artifact_type, file_path, content_type in artifact_files:
        if os.path.exists(file_path):
            try:
                url = upload_model_artifact(stock_symbol, run_id, artifact_type, file_path)
                urls[artifact_type] = url
            except Exception as e:
                logger.warning(f"Failed to upload {artifact_type} for {stock_symbol}: {e}")
    
    return urls


def fetch_stock_data_task(stock_symbol: str, to_date: str | None = None) -> bool:
    """
    Task to fetch stock data.
    
    Args:
        stock_symbol: Stock ticker symbol
        to_date: Optional end date
        
    Returns:
        True if successful
    """
    db = SessionLocal()
    try:
        return fetch_stock_data(db, stock_symbol, to_date=to_date)
    finally:
        db.close()


def train_single_stock_task(stock_symbol: str, continue_training: bool = True) -> bool:
    """
    Task to train a single stock model.
    
    Args:
        stock_symbol: Stock ticker symbol
        continue_training: Whether to continue from existing model
        
    Returns:
        True if successful
    """
    db = SessionLocal()
    try:
        # Fetch data
        fetch_stock_data(db, stock_symbol)
        
        # Train
        result = train_prediction_model(db, stock_symbol, continue_training=continue_training)
        
        if result:
            # Evaluate and predict
            evaluate_model(db, stock_symbol)
            predict_future_prices(db, stock_symbol)
        
        return result
    finally:
        db.close()

