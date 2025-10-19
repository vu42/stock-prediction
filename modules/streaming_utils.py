"""
Streaming Utilities for Kafka + Airflow Integration

Helper functions for monitoring Kafka topics and PostgreSQL data flow.
"""
import psycopg2
from datetime import datetime, timedelta
from kafka import KafkaConsumer, KafkaAdminClient
from kafka.errors import KafkaError

from config import KAFKA_CONFIG, DB_CONFIG


def check_kafka_topic_has_messages(topic_name=None, timeout_seconds=10):
    """
    Check if Kafka topic has any messages.
    
    Args:
        topic_name (str): Kafka topic name. Defaults to config value.
        timeout_seconds (int): Timeout for checking
        
    Returns:
        bool: True if topic has messages
    """
    if topic_name is None:
        topic_name = KAFKA_CONFIG['topic_name']
    
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            consumer_timeout_ms=timeout_seconds * 1000
        )
        
        # Try to get one message
        for message in consumer:
            consumer.close()
            print(f"[SENSOR] ✓ Topic '{topic_name}' has messages")
            return True
        
        consumer.close()
        print(f"[SENSOR] ○ Topic '{topic_name}' is empty or no new messages")
        return False
        
    except KafkaError as e:
        print(f"[SENSOR] ✗ Kafka error: {str(e)}")
        return False
    except Exception as e:
        print(f"[SENSOR] ✗ Error checking topic: {str(e)}")
        return False


def get_new_records_count(since_minutes=30):
    """
    Count new records in PostgreSQL since last N minutes.
    
    Args:
        since_minutes (int): Look back window in minutes
        
    Returns:
        int: Number of new records
    """
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(minutes=since_minutes)
        
        query = """
            SELECT COUNT(*) 
            FROM stock_prices 
            WHERE created_at >= %s
        """
        cursor.execute(query, (cutoff_time,))
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"[SENSOR] New records in last {since_minutes} min: {count}")
        return count
        
    except Exception as e:
        print(f"[SENSOR] ✗ Error counting records: {str(e)}")
        return 0


def check_kafka_producer_health():
    """
    Check if Kafka producer is healthy by checking topic availability.
    
    Returns:
        dict: Health status with details
    """
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            request_timeout_ms=5000
        )
        
        topics = admin_client.list_topics()
        target_topic = KAFKA_CONFIG['topic_name']
        
        if target_topic in topics:
            print(f"[HEALTH] ✓ Topic '{target_topic}' exists")
            admin_client.close()
            return {
                'status': 'healthy',
                'topic_exists': True,
                'topic_name': target_topic
            }
        else:
            print(f"[HEALTH] ✗ Topic '{target_topic}' does not exist")
            admin_client.close()
            return {
                'status': 'unhealthy',
                'topic_exists': False,
                'topic_name': target_topic
            }
            
    except Exception as e:
        print(f"[HEALTH] ✗ Kafka health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


def get_consumer_lag():
    """
    Get consumer lag (difference between produced and consumed messages).
    
    Returns:
        dict: Consumer lag information
    """
    try:
        consumer = KafkaConsumer(
            KAFKA_CONFIG['topic_name'],
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            group_id=KAFKA_CONFIG['consumer_group'],
            enable_auto_commit=False
        )
        
        # Get partition assignments
        partitions = consumer.assignment()
        
        if not partitions:
            # Trigger partition assignment
            consumer.poll(timeout_ms=1000)
            partitions = consumer.assignment()
        
        total_lag = 0
        partition_info = []
        
        for partition in partitions:
            # Get current position
            position = consumer.position(partition)
            
            # Get end offset (latest)
            end_offsets = consumer.end_offsets([partition])
            end_offset = end_offsets[partition]
            
            lag = end_offset - position
            total_lag += lag
            
            partition_info.append({
                'partition': partition.partition,
                'position': position,
                'end_offset': end_offset,
                'lag': lag
            })
        
        consumer.close()
        
        print(f"[HEALTH] Consumer lag: {total_lag} messages")
        return {
            'total_lag': total_lag,
            'partitions': partition_info
        }
        
    except Exception as e:
        print(f"[HEALTH] ✗ Error getting consumer lag: {str(e)}")
        return {
            'total_lag': -1,
            'error': str(e)
        }


def should_trigger_training(min_records_threshold=100):
    """
    Decide if model training should be triggered based on new data.
    
    Args:
        min_records_threshold (int): Minimum new records needed to trigger
        
    Returns:
        bool: True if should trigger training
    """
    new_records = get_new_records_count(since_minutes=30)
    
    if new_records >= min_records_threshold:
        print(f"[DECISION] ✓ Trigger training: {new_records} >= {min_records_threshold}")
        return True
    else:
        print(f"[DECISION] ○ Skip training: {new_records} < {min_records_threshold}")
        return False

