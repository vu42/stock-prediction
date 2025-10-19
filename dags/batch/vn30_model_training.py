"""
VN30 Model Training & Prediction DAG
Train LSTM models and generate predictions

Schedule: Daily at 6:00 PM Vietnam Time (after data crawler)
Duration: 2-3 hours
Purpose: Train models on latest data and predict 30-day future prices
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
from airflow.sensors.external_task import ExternalTaskSensor

from config import STOCK_SYMBOLS
from modules.model_trainer import train_prediction_model, evaluate_model, predict_future_prices
from modules.email_notifier import send_email_notification


def train_stock_task(stock_symbol):
    """Train LSTM model for a single stock."""
    return train_prediction_model(stock_symbol)


def evaluate_stock_task(stock_symbol):
    """Evaluate model for a single stock."""
    return evaluate_model(stock_symbol)


def predict_stock_task(stock_symbol):
    """Generate future predictions for a single stock."""
    return predict_future_prices(stock_symbol)


def send_email_task(**context):
    """Send summary email report."""
    return send_email_notification(STOCK_SYMBOLS)


# Vietnam timezone
local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

# Default arguments
default_args = {
    'owner': 'ml_engineering',
    'email': ['tuph.alex@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# Define DAG
with DAG(
    dag_id='vn30_model_training',
    default_args=default_args,
    description=f'LSTM model training and prediction for {len(STOCK_SYMBOLS)} VN30 stocks',
    # Production schedule: Daily at 6:00 PM Vietnam Time (after data crawler at 5:00 PM)
    # schedule_interval='0 18 * * *',  # Uncomment for production
    schedule_interval=None,  # Manual trigger for demo
    start_date=pendulum.datetime(2025, 10, 1, tz=local_tz),
    catchup=False,
    tags=['vn30', 'ml', 'lstm', 'training', 'prediction', 'batch'],
    max_active_runs=1,
) as dag:
    
    # Task 0: Wait for data crawler to complete
    wait_for_crawler = ExternalTaskSensor(
        task_id='wait_for_data_crawler',
        external_dag_id='vn30_data_crawler',
        external_task_id=None,  # Wait for entire DAG
        timeout=3600,  # 1 hour timeout
        mode='reschedule',
        poke_interval=300,  # Check every 5 minutes
    )
    
    # Tasks: Train, Evaluate, Predict for each stock
    all_predict_tasks = []
    
    for stock in STOCK_SYMBOLS:
        # Train
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
        
        # Chain: Train -> Evaluate -> Predict
        wait_for_crawler >> train_task >> evaluate_task >> predict_task
        all_predict_tasks.append(predict_task)
    
    # Final task: Send email after all predictions complete
    send_email = PythonOperator(
        task_id='send_email_report',
        python_callable=send_email_task,
    )
    
    # All predictions must complete before email
    all_predict_tasks >> send_email

