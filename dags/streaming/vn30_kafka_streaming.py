"""
Airflow DAG for Kafka Real-time Streaming Simulation

This DAG runs Kafka producer and consumer for continuous
stock price streaming from VNDirect API to PostgreSQL.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pendulum

from modules.database import init_database
from modules.kafka_producer import run_producer
from modules.kafka_consumer import run_consumer
from config import STOCK_SYMBOLS

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0,  # Don't retry streaming tasks
}

local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

with DAG(
    dag_id='vn30_kafka_streaming',
    default_args=default_args,
    description=f'Real-time streaming simulation for {len(STOCK_SYMBOLS)} VN30 stocks via Kafka',
    schedule_interval=None,  # Manual trigger or run continuously
    start_date=pendulum.datetime(2024, 10, 1, tz=local_tz),
    catchup=False,
    tags=['stock-prediction', 'kafka', 'streaming', 'vn30', 'real-time'],
) as dag:
    
    # Task 1: Initialize Database
    init_db_task = PythonOperator(
        task_id='initialize_database',
        python_callable=init_database,
    )
    
    # Task 2: Start Kafka Producer (runs for 1 hour as demo)
    # For production, run this as a separate long-running service
    start_producer_task = PythonOperator(
        task_id='start_kafka_producer',
        python_callable=run_producer,
        op_kwargs={
            'duration_minutes': 60  # Run for 60 minutes
        }
    )
    
    # Task 3: Start Kafka Consumer (runs for 1 hour as demo)
    # For production, run this as a separate long-running service
    start_consumer_task = PythonOperator(
        task_id='start_kafka_consumer',
        python_callable=run_consumer,
        op_kwargs={
            'duration_minutes': 60  # Run for 60 minutes
        }
    )
    
    # Note: In production, producer and consumer should run in parallel
    # as separate services (not in Airflow). This DAG is for demonstration.
    
    init_db_task >> [start_producer_task, start_consumer_task]


# Standalone DAG for running producer only (for testing)
with DAG(
    dag_id='kafka_producer_standalone',
    default_args=default_args,
    description='Run Kafka producer standalone for testing',
    schedule_interval=None,
    start_date=pendulum.datetime(2024, 10, 1, tz=local_tz),
    catchup=False,
    tags=['kafka', 'producer', 'test'],
) as producer_dag:
    
    run_producer_task = PythonOperator(
        task_id='run_kafka_producer',
        python_callable=run_producer,
        op_kwargs={
            'duration_minutes': 10  # 10 min test
        }
    )


# Standalone DAG for running consumer only (for testing)
with DAG(
    dag_id='kafka_consumer_standalone',
    default_args=default_args,
    description='Run Kafka consumer standalone for testing',
    schedule_interval=None,
    start_date=pendulum.datetime(2024, 10, 1, tz=local_tz),
    catchup=False,
    tags=['kafka', 'consumer', 'test'],
) as consumer_dag:
    
    run_consumer_task = PythonOperator(
        task_id='run_kafka_consumer',
        python_callable=run_consumer,
        op_kwargs={
            'duration_minutes': 10  # 10 min test
        }
    )

