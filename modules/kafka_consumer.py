"""
Kafka Consumer - Real-time Stock Price Ingestion with Aggregation

Consumes stock price ticks from Kafka topic, aggregates them
into daily OHLCV format, and stores in PostgreSQL database.
"""
import json
import time
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config import KAFKA_CONFIG
from modules.tick_aggregator import TickAggregator


class StockPriceConsumer:
    """
    Kafka Consumer that reads stock price ticks from Kafka topic,
    aggregates them into daily OHLCV, and saves to PostgreSQL.
    """
    
    def __init__(self, flush_threshold=1000):
        """
        Initialize Kafka consumer with configuration.
        
        Args:
            flush_threshold: Number of ticks to cache before auto-flushing
        """
        self.topic = KAFKA_CONFIG['topic_name']
        self.consumer_group = KAFKA_CONFIG['consumer_group']
        
        # Initialize Tick Aggregator
        self.aggregator = TickAggregator(flush_threshold=flush_threshold)
        
        # Initialize Kafka Consumer
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            group_id=self.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            max_poll_records=KAFKA_CONFIG['max_poll_records']
        )
        
        print(f"[CONSUMER] Connected to Kafka: {KAFKA_CONFIG['bootstrap_servers']}")
        print(f"[CONSUMER] Subscribed to topic: {self.topic}")
        print(f"[CONSUMER] Consumer group: {self.consumer_group}")
        print(f"[CONSUMER] Aggregation enabled with flush threshold: {flush_threshold}")
    
    def process_message(self, message):
        """
        Process a single Kafka message by adding to aggregator.
        
        Args:
            message: Kafka message object
            
        Returns:
            bool: True if successful
        """
        try:
            # Extract message data
            stock_symbol = message.key
            value = message.value
            
            # Extract tick data
            tick_data = value['data']
            
            # Ensure stock_symbol is in tick data
            if 'stock_symbol' not in tick_data:
                tick_data['stock_symbol'] = stock_symbol
            
            # Add tick to aggregator
            self.aggregator.add_tick(tick_data)
            
            # Check if should flush
            if self.aggregator.should_flush():
                flushed = self.aggregator.flush_to_database()
                if flushed > 0:
                    print(f"[CONSUMER] Auto-flushed {flushed} aggregated OHLCV records")
            
            return True
                
        except Exception as e:
            print(f"[CONSUMER] ✗ Error processing message: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_consuming(self, duration_minutes=None):
        """
        Start consuming messages from Kafka topic.
        
        Args:
            duration_minutes (int, optional): Run for specific duration. None = run indefinitely.
        """
        print(f"\n{'='*80}")
        print(f"Starting Stock Price Consumer with Aggregation")
        print(f"{'='*80}")
        print(f"Topic: {self.topic}")
        print(f"Consumer Group: {self.consumer_group}")
        print(f"Aggregation: Ticks → Daily OHLCV")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        else:
            print(f"Duration: Indefinite (until stopped)")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        message_count = 0
        success_count = 0
        
        try:
            for message in self.consumer:
                message_count += 1
                
                # Process message (add to aggregator)
                if self.process_message(message):
                    success_count += 1
                
                # Check duration limit
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        print(f"\n[CONSUMER] Duration limit reached ({duration_minutes} min)")
                        break
                
                # Print stats every 100 messages
                if message_count % 100 == 0:
                    stats = self.aggregator.get_cache_stats()
                    print(f"\n[CONSUMER] Progress: {message_count} messages consumed | "
                          f"{stats['total_ticks_cached']} ticks cached | "
                          f"{stats['total_records_flushed']} OHLCV records flushed")
        
        except KeyboardInterrupt:
            print("\n[CONSUMER] Interrupted by user")
        
        finally:
            # Force flush remaining data
            print(f"\n[CONSUMER] Flushing remaining aggregated data...")
            remaining = self.aggregator.force_flush_all()
            if remaining > 0:
                print(f"[CONSUMER] ✓ Flushed final {remaining} records")
            
            # Print final stats
            print(f"\n[CONSUMER] Final Stats:")
            print(f"   Messages consumed: {message_count}")
            print(f"   Successfully processed: {success_count}")
            self.aggregator.print_stats()
            
            self.close()
    
    def close(self):
        """Close Kafka consumer connection."""
        print("\n[CONSUMER] Closing connection...")
        self.consumer.close()
        print("[CONSUMER] Connection closed")


def run_consumer(duration_minutes=None, flush_threshold=1000):
    """
    Main function to run Kafka consumer.
    
    Args:
        duration_minutes (int, optional): Run duration in minutes
        flush_threshold (int): Number of ticks before auto-flush
    """
    consumer = StockPriceConsumer(flush_threshold=flush_threshold)
    consumer.start_consuming(duration_minutes=duration_minutes)


if __name__ == "__main__":
    # Run indefinitely with default flush threshold
    run_consumer()
