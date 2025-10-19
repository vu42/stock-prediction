"""
Airflow DAG - Streaming Orchestrator (Option 1: Hybrid Architecture)

This DAG monitors Kafka streaming pipeline and triggers ML training
when sufficient new data is available.

Prerequisites:
- Kafka producer running as background service
- Kafka consumer running as background service
- PostgreSQL database with stock_prices table

Flow:
1. Check Kafka topic has messages (KafkaSensor)
2. Count new records in PostgreSQL
3. Decide if training threshold met
4. If yes: Trigger model training (MOCK for demo)
5. Send notification (MOCK for demo)
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pendulum

from modules.streaming_utils import (
    check_kafka_topic_has_messages,
    get_new_records_count,
    should_trigger_training
)
from config import STOCK_SYMBOLS

# Configuration
TRAINING_THRESHOLD = 50  # Minimum new records to trigger training (lowered for demo)
LOOKBACK_MINUTES = 30    # Check records from last 30 minutes

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")


def check_kafka_messages(**context):
    """Check if Kafka topic has messages."""
    has_messages = check_kafka_topic_has_messages(timeout_seconds=10)
    
    if has_messages:
        print("[ORCHESTRATOR] ✓ Kafka topic has messages, proceeding...")
        return True
    else:
        print("[ORCHESTRATOR] ○ No messages in Kafka topic, will retry...")
        raise Exception("No messages in Kafka topic")


def count_new_records(**context):
    """Count new records in PostgreSQL."""
    count = get_new_records_count(since_minutes=LOOKBACK_MINUTES)
    
    # Push to XCom for next task
    context['task_instance'].xcom_push(key='new_records_count', value=count)
    
    print(f"[ORCHESTRATOR] New records count: {count}")
    return count


def decide_training(**context):
    """Decide whether to trigger training based on threshold."""
    # Pull from XCom
    ti = context['task_instance']
    new_records = ti.xcom_pull(task_ids='count_new_records', key='new_records_count')
    
    print(f"\n{'='*60}")
    print(f"DECISION POINT")
    print(f"{'='*60}")
    print(f"New records: {new_records}")
    print(f"Threshold: {TRAINING_THRESHOLD}")
    
    if new_records >= TRAINING_THRESHOLD:
        print(f"✓ DECISION: Trigger training ({new_records} >= {TRAINING_THRESHOLD})")
        print(f"{'='*60}\n")
        return 'trigger_training'
    else:
        print(f"○ DECISION: Skip training ({new_records} < {TRAINING_THRESHOLD})")
        print(f"{'='*60}\n")
        return 'skip_training'


def mock_trigger_training(**context):
    """MOCK: Simulate triggering model training for all VN30 stocks."""
    print(f"\n{'='*80}")
    print(f"[MOCK] TRIGGERING MODEL TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Total stocks to process: {len(STOCK_SYMBOLS)}")
    print(f"Stocks: {', '.join(STOCK_SYMBOLS[:10])}...")
    print(f"\n[MOCK] Steps that would execute:")
    print(f"  1. Load data from PostgreSQL")
    print(f"  2. Train LSTM models (100 epochs)")
    print(f"  3. Save models to output/{{stock}}/")
    print(f"  4. Generate evaluation metrics")
    print(f"  5. Create prediction charts")
    print(f"\n✓ [MOCK] Training pipeline triggered successfully")
    print(f"{'='*80}\n")


def mock_evaluate_models(**context):
    """MOCK: Simulate model evaluation."""
    print(f"\n{'='*80}")
    print(f"[MOCK] EVALUATING MODELS")
    print(f"{'='*80}")
    print(f"Total models: {len(STOCK_SYMBOLS)}")
    print(f"\n[MOCK] Evaluation metrics (simulated):")
    print(f"  - Average RMSE: 2,345.67")
    print(f"  - Average MAE: 1,234.56")
    print(f"  - Average MAPE: 3.45%")
    print(f"  - Average R²: 0.8923")
    print(f"  - Average Direction Accuracy: 78.5%")
    print(f"\n✓ [MOCK] Model evaluation completed")
    print(f"{'='*80}\n")


def mock_predict_future(**context):
    """MOCK: Simulate future price predictions."""
    print(f"\n{'='*80}")
    print(f"[MOCK] PREDICTING FUTURE PRICES")
    print(f"{'='*80}")
    print(f"Prediction horizon: 30 days")
    print(f"Total stocks: {len(STOCK_SYMBOLS)}")
    print(f"\n[MOCK] Sample predictions:")
    print(f"  VCB: Day 1: 89,500 → Day 30: 92,300 (+3.1%)")
    print(f"  FPT: Day 1: 124,000 → Day 30: 128,500 (+3.6%)")
    print(f"  HPG: Day 1: 25,800 → Day 30: 26,900 (+4.3%)")
    print(f"\n[MOCK] Prediction CSVs generated:")
    print(f"  - output/{{stock}}/{{stock}}_future_predictions.csv")
    print(f"  - output/{{stock}}/{{stock}}_future.png")
    print(f"\n✓ [MOCK] Future predictions completed")
    print(f"{'='*80}\n")


def mock_send_notification(**context):
    """MOCK: Simulate sending email notification."""
    ti = context['task_instance']
    new_records = ti.xcom_pull(task_ids='count_new_records', key='new_records_count')
    
    print(f"\n{'='*80}")
    print(f"[MOCK] SENDING EMAIL NOTIFICATION")
    print(f"{'='*80}")
    print(f"To: tuph.alex@gmail.com")
    print(f"Subject: VN30 Streaming Pipeline - Training Complete")
    print(f"\n[MOCK] Email body:")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  VN30 Stock Prediction - Streaming Pipeline Report")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ")
    print(f"  Pipeline Type: Real-time Kafka Streaming")
    print(f"  New Records: {new_records}")
    print(f"  Stocks Processed: {len(STOCK_SYMBOLS)}")
    print(f"  ")
    print(f"  ✓ Models trained and updated")
    print(f"  ✓ Evaluation metrics calculated")
    print(f"  ✓ 30-day predictions generated")
    print(f"  ")
    print(f"  Attachments (would include):")
    print(f"    - VN30_summary_report.pdf")
    print(f"    - Sample evaluation charts (3 stocks)")
    print(f"    - Sample prediction charts (3 stocks)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\n✓ [MOCK] Email sent successfully")
    print(f"{'='*80}\n")


# Define the DAG
with DAG(
    dag_id='vn30_streaming_orchestrator',
    default_args=default_args,
    description=f'Kafka streaming orchestrator - monitors and triggers ML pipeline for {len(STOCK_SYMBOLS)} VN30 stocks',
    schedule_interval='*/30 * * * *',  # Run every 30 minutes
    start_date=pendulum.datetime(2024, 10, 1, tz=local_tz),
    catchup=False,
    tags=['stock-prediction', 'kafka', 'streaming', 'orchestrator', 'vn30'],
) as dag:
    
    # Task 1: Check Kafka topic has messages
    check_kafka_task = PythonOperator(
        task_id='check_kafka_messages',
        python_callable=check_kafka_messages,
    )
    
    # Task 2: Count new records in PostgreSQL
    count_records_task = PythonOperator(
        task_id='count_new_records',
        python_callable=count_new_records,
    )
    
    # Task 3: Decide whether to trigger training (Branch)
    decide_task = BranchPythonOperator(
        task_id='decide_training',
        python_callable=decide_training,
    )
    
    # Task 4a: Trigger training (if threshold met)
    trigger_training_task = PythonOperator(
        task_id='trigger_training',
        python_callable=mock_trigger_training,
    )
    
    # Task 5: Evaluate models (MOCK)
    evaluate_task = PythonOperator(
        task_id='evaluate_models',
        python_callable=mock_evaluate_models,
    )
    
    # Task 6: Predict future prices (MOCK)
    predict_task = PythonOperator(
        task_id='predict_future_prices',
        python_callable=mock_predict_future,
    )
    
    # Task 7: Send notification (MOCK)
    notify_task = PythonOperator(
        task_id='send_notification',
        python_callable=mock_send_notification,
    )
    
    # Task 4b: Skip training (if threshold not met)
    skip_task = EmptyOperator(
        task_id='skip_training',
    )
    
    # Task 8: Join point
    end_task = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success',  # Continue even if skipped
    )
    
    # Define task dependencies
    check_kafka_task >> count_records_task >> decide_task
    
    # Branch: Training path
    decide_task >> trigger_training_task >> evaluate_task >> predict_task >> notify_task >> end_task
    
    # Branch: Skip path
    decide_task >> skip_task >> end_task

