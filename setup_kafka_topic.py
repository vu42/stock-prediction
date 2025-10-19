#!/usr/bin/env python3
"""
Setup Kafka Topic with Partitions

Creates Kafka topic with specified number of partitions and replication factor.
Run this before starting the streaming demo.
"""

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, KafkaError
from config import KAFKA_CONFIG
import time


def create_kafka_topic():
    """Create Kafka topic with partitions."""
    
    topic_name = KAFKA_CONFIG['topic_name']
    num_partitions = KAFKA_CONFIG['num_partitions']
    replication_factor = KAFKA_CONFIG['replication_factor']
    
    print("=" * 60)
    print("KAFKA TOPIC SETUP")
    print("=" * 60)
    print(f"Topic Name:           {topic_name}")
    print(f"Partitions:           {num_partitions}")
    print(f"Replication Factor:   {replication_factor}")
    print(f"Bootstrap Servers:    {KAFKA_CONFIG['bootstrap_servers']}")
    print("=" * 60)
    print()
    
    try:
        # Connect to Kafka
        print("[1/3] Connecting to Kafka broker...")
        admin_client = KafkaAdminClient(
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            client_id='topic-setup-client'
        )
        print("✓ Connected to Kafka broker")
        print()
        
        # Create topic
        print(f"[2/3] Creating topic '{topic_name}'...")
        topic = NewTopic(
            name=topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor,
            topic_configs={
                'retention.ms': '604800000',  # 7 days retention
                'cleanup.policy': 'delete',
                'compression.type': 'snappy'
            }
        )
        
        admin_client.create_topics(new_topics=[topic], validate_only=False)
        print(f"✓ Topic '{topic_name}' created successfully!")
        print()
        
        # Wait for topic to be ready
        print("[3/3] Waiting for topic to be ready...")
        time.sleep(2)
        
        # Verify topic
        topics = admin_client.list_topics()
        if topic_name in topics:
            print(f"✓ Topic '{topic_name}' is ready!")
            print()
            
            # Get topic details
            topic_metadata = admin_client.describe_topics([topic_name])
            for topic_info in topic_metadata:
                print("Topic Details:")
                print(f"  - Name: {topic_info['topic']}")
                print(f"  - Partitions: {len(topic_info['partitions'])}")
                print(f"  - Partition IDs: {[p['partition'] for p in topic_info['partitions']]}")
                print()
        
        admin_client.close()
        
        print("=" * 60)
        print("✓ SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Start producer: python3 -m modules.kafka_producer")
        print("  2. Start consumer: python3 -m modules.kafka_consumer")
        print("  3. Check Kafka UI: http://localhost:8080")
        print()
        
        return True
        
    except TopicAlreadyExistsError:
        print(f"⚠️  Topic '{topic_name}' already exists!")
        print()
        print("To recreate the topic:")
        print(f"  1. Delete existing topic:")
        print(f"     docker exec -it vn30-kafka kafka-topics \\")
        print(f"       --bootstrap-server localhost:9092 \\")
        print(f"       --delete --topic {topic_name}")
        print(f"  2. Run this script again")
        print()
        
        # Show existing topic details
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
                client_id='topic-setup-client'
            )
            topic_metadata = admin_client.describe_topics([topic_name])
            for topic_info in topic_metadata:
                print("Current Topic Details:")
                print(f"  - Name: {topic_info['topic']}")
                print(f"  - Partitions: {len(topic_info['partitions'])}")
                print(f"  - Partition IDs: {[p['partition'] for p in topic_info['partitions']]}")
            admin_client.close()
        except:
            pass
        
        return False
        
    except KafkaError as e:
        print(f"✗ Kafka error: {str(e)}")
        print()
        print("Make sure Kafka is running:")
        print("  docker-compose up -d")
        print()
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False


def delete_kafka_topic():
    """Delete existing Kafka topic."""
    topic_name = KAFKA_CONFIG['topic_name']
    
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            client_id='topic-setup-client'
        )
        
        print(f"Deleting topic '{topic_name}'...")
        admin_client.delete_topics([topic_name])
        print(f"✓ Topic '{topic_name}' deleted!")
        
        admin_client.close()
        time.sleep(2)
        return True
        
    except Exception as e:
        print(f"✗ Error deleting topic: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--delete":
        delete_kafka_topic()
    else:
        create_kafka_topic()

