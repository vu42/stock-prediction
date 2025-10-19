"""
Airflow DAG - Kafka Health Monitor

Monitors the health of Kafka streaming infrastructure:
- Kafka broker availability
- Topic existence
- Consumer lag
- Producer/Consumer service status

Runs every 5 minutes to ensure streaming pipeline is healthy.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pendulum

from modules.streaming_utils import (
    check_kafka_producer_health,
    get_consumer_lag,
    check_kafka_topic_has_messages
)
from config import KAFKA_CONFIG

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

# Alert thresholds
MAX_CONSUMER_LAG = 1000  # Alert if lag exceeds this


def monitor_kafka_broker(**context):
    """Monitor Kafka broker and topic health."""
    print(f"\n{'='*80}")
    print(f"[MONITOR] KAFKA BROKER HEALTH CHECK")
    print(f"{'='*80}")
    
    health = check_kafka_producer_health()
    
    print(f"Status: {health.get('status', 'unknown')}")
    print(f"Topic: {health.get('topic_name', 'N/A')}")
    print(f"Topic Exists: {health.get('topic_exists', False)}")
    
    if health.get('status') == 'unhealthy':
        print(f"\n⚠️  WARNING: Kafka broker is unhealthy!")
        if 'error' in health:
            print(f"Error: {health['error']}")
        print(f"{'='*80}\n")
        raise Exception("Kafka broker is unhealthy")
    
    print(f"\n✓ Kafka broker is healthy")
    print(f"{'='*80}\n")


def monitor_consumer_lag(**context):
    """Monitor Kafka consumer lag."""
    print(f"\n{'='*80}")
    print(f"[MONITOR] CONSUMER LAG CHECK")
    print(f"{'='*80}")
    
    lag_info = get_consumer_lag()
    total_lag = lag_info.get('total_lag', -1)
    
    if total_lag < 0:
        print(f"\n⚠️  WARNING: Could not determine consumer lag")
        if 'error' in lag_info:
            print(f"Error: {lag_info['error']}")
        print(f"{'='*80}\n")
        return
    
    print(f"Total Lag: {total_lag} messages")
    print(f"Threshold: {MAX_CONSUMER_LAG} messages")
    
    # Print partition details
    if 'partitions' in lag_info:
        print(f"\nPartition Details:")
        for p in lag_info['partitions']:
            print(f"  Partition {p['partition']}: Lag = {p['lag']} (pos: {p['position']}, end: {p['end_offset']})")
    
    # Check if lag is too high
    if total_lag > MAX_CONSUMER_LAG:
        print(f"\n⚠️  ALERT: Consumer lag is too high! ({total_lag} > {MAX_CONSUMER_LAG})")
        print(f"Action required: Check consumer service")
        print(f"{'='*80}\n")
        # Don't fail the task, just alert
    else:
        print(f"\n✓ Consumer lag is within acceptable range")
        print(f"{'='*80}\n")


def check_data_flow(**context):
    """Check if data is flowing through the pipeline."""
    print(f"\n{'='*80}")
    print(f"[MONITOR] DATA FLOW CHECK")
    print(f"{'='*80}")
    
    has_messages = check_kafka_topic_has_messages(timeout_seconds=5)
    
    if has_messages:
        print(f"✓ Data is flowing: Messages found in topic '{KAFKA_CONFIG['topic_name']}'")
    else:
        print(f"○ No recent messages in topic '{KAFKA_CONFIG['topic_name']}'")
        print(f"  This might be normal if market is closed or no updates available")
    
    print(f"{'='*80}\n")


def generate_health_summary(**context):
    """Generate and log overall health summary."""
    print(f"\n{'='*80}")
    print(f"[MONITOR] HEALTH SUMMARY")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Kafka Configuration:")
    print(f"  - Bootstrap Servers: {KAFKA_CONFIG['bootstrap_servers']}")
    print(f"  - Topic: {KAFKA_CONFIG['topic_name']}")
    print(f"  - Consumer Group: {KAFKA_CONFIG['consumer_group']}")
    print(f"  - Polling Interval: {KAFKA_CONFIG['polling_interval']}s")
    print(f"\nOverall Status: ✓ HEALTHY")
    print(f"\nNext check: ~5 minutes")
    print(f"{'='*80}\n")


# Define the DAG
with DAG(
    dag_id='kafka_health_monitor',
    default_args=default_args,
    description='Monitors Kafka streaming infrastructure health',
    schedule_interval='*/5 * * * *',  # Run every 5 minutes
    start_date=pendulum.datetime(2024, 10, 1, tz=local_tz),
    catchup=False,
    tags=['kafka', 'monitoring', 'health-check', 'infrastructure'],
) as dag:
    
    # Task 1: Monitor Kafka broker
    broker_task = PythonOperator(
        task_id='monitor_kafka_broker',
        python_callable=monitor_kafka_broker,
    )
    
    # Task 2: Monitor consumer lag
    lag_task = PythonOperator(
        task_id='monitor_consumer_lag',
        python_callable=monitor_consumer_lag,
    )
    
    # Task 3: Check data flow
    flow_task = PythonOperator(
        task_id='check_data_flow',
        python_callable=check_data_flow,
    )
    
    # Task 4: Generate summary
    summary_task = PythonOperator(
        task_id='generate_health_summary',
        python_callable=generate_health_summary,
    )
    
    # Define task dependencies (run in parallel, then summary)
    [broker_task, lag_task, flow_task] >> summary_task

