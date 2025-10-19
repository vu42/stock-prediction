"""
Kafka Producer - Real-time Stock Price Streaming

Supports 2 modes:
1. Simulated: Generate fake ticks (1/second) for demo - NO API calls
2. Real: Poll VNDirect API (every 10s) - limited to daily data
"""
import json
import time
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from kafka import KafkaProducer
from kafka.errors import KafkaError

from config import (
    KAFKA_CONFIG, STOCK_SYMBOLS, API_BASE_URL, 
    STREAMING_MODE, SIMULATED_STREAMING
)
from modules.simulated_data_generator import SimulatedDataGenerator


class StockPriceProducer:
    """
    Kafka Producer for stock price streaming.
    
    Modes:
    - 'simulated': Generate fake ticks using random walk (demo mode)
    - 'real': Poll VNDirect API for actual data (limited to daily OHLCV)
    """
    
    def __init__(self, mode=None):
        """
        Initialize Kafka producer with configuration.
        
        Args:
            mode: 'simulated' or 'real'. Defaults to config.STREAMING_MODE
        """
        self.mode = mode or STREAMING_MODE
        self.topic = KAFKA_CONFIG['topic_name']
        self.polling_interval = KAFKA_CONFIG['polling_interval']
        
        # Initialize Kafka Producer
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        
        # Initialize simulator for simulated mode
        if self.mode == 'simulated':
            self.simulator = SimulatedDataGenerator(
                stock_symbols=STOCK_SYMBOLS,
                base_prices=None  # Use defaults
            )
            self.simulator.volatility = SIMULATED_STREAMING['volatility']
            self.simulator.drift = SIMULATED_STREAMING['drift']
            self.simulator.min_volume = SIMULATED_STREAMING['min_volume']
            self.simulator.max_volume = SIMULATED_STREAMING['max_volume']
        
        print(f"[PRODUCER] Mode: {self.mode.upper()}")
        print(f"[PRODUCER] Connected to Kafka: {KAFKA_CONFIG['bootstrap_servers']}")
        print(f"[PRODUCER] Publishing to topic: {self.topic}")
        
        if self.mode == 'simulated':
            print(f"[PRODUCER] Simulated streaming: {SIMULATED_STREAMING['tick_interval']}s interval")
        else:
            print(f"[PRODUCER] Real API polling: {self.polling_interval}s interval")
    
    def fetch_latest_price_from_api(self, stock_symbol):
        """
        Fetch latest stock price from VNDirect API (REAL mode).
        
        Args:
            stock_symbol (str): Stock symbol to fetch
            
        Returns:
            dict: Stock price data or None if error
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            
            query_params = f"sort=date:desc&q=code:{stock_symbol}~date:gte:{start_date}~date:lte:{end_date}&size=1&page=1"
            api_url = f"{API_BASE_URL}?{query_params}"
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            request = Request(api_url, headers=headers)
            response = urlopen(request, timeout=10).read()
            data = json.loads(response)['data']
            
            if data and len(data) > 0:
                return data[0]
            return None
            
        except Exception as e:
            print(f"[PRODUCER] Error fetching {stock_symbol}: {str(e)}")
            return None
    
    def publish_tick(self, tick_data):
        """
        Publish a single tick to Kafka topic.
        
        Args:
            tick_data (dict): Tick data (can be from simulator or API)
            
        Returns:
            bool: True if successful
        """
        try:
            stock_symbol = tick_data.get('stock_symbol')
            
            # Prepare message
            message = {
                'stock_symbol': stock_symbol,
                'timestamp': tick_data.get('timestamp', datetime.now().isoformat()),
                'data': tick_data,
                'mode': self.mode
            }
            
            # Publish to Kafka
            future = self.producer.send(
                self.topic,
                key=stock_symbol,
                value=message
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            # Print compact log (every 30 ticks to avoid spam)
            if self.mode == 'simulated':
                tick_id = tick_data.get('tick_id', 0)
                if tick_id % 30 == 0:
                    print(f"[PRODUCER] ✓ {stock_symbol} | Price: {tick_data.get('price', 0):,.2f} | Tick #{tick_id}")
            else:
                print(f"[PRODUCER] ✓ {stock_symbol} → Partition {record_metadata.partition}")
            
            return True
            
        except KafkaError as e:
            print(f"[PRODUCER] ✗ Kafka error: {str(e)}")
            return False
        except Exception as e:
            print(f"[PRODUCER] ✗ Error publishing: {str(e)}")
            return False
    
    def start_streaming_simulated(self, duration_minutes=None):
        """Start simulated streaming mode (generates fake ticks)."""
        print(f"\n{'='*80}")
        print(f"Starting SIMULATED Streaming Producer")
        print(f"{'='*80}")
        print(f"Stocks: {len(STOCK_SYMBOLS)}")
        print(f"Tick Interval: {SIMULATED_STREAMING['tick_interval']}s")
        print(f"Volatility: {SIMULATED_STREAMING['volatility']*100:.2f}%")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        else:
            print(f"Duration: Indefinite (Ctrl+C to stop)")
        print(f"{'='*80}\n")
        
        try:
            # Stream batches from simulator
            for batch in self.simulator.stream_continuous(
                interval_seconds=SIMULATED_STREAMING['tick_interval'],
                duration_minutes=duration_minutes
            ):
                # Publish each tick in batch
                success_count = 0
                for tick in batch:
                    if self.publish_tick(tick):
                        success_count += 1
                
                # Log batch summary every 10 batches
                batch_num = self.simulator.tick_count // len(STOCK_SYMBOLS)
                if batch_num % 10 == 0 and batch_num > 0:
                    stats = self.simulator.get_statistics()
                    print(f"\n[PRODUCER] Batch {batch_num} | "
                          f"Total: {stats['total_ticks']} ticks | "
                          f"Rate: {stats['ticks_per_second']:.1f} ticks/s")
        
        except KeyboardInterrupt:
            print("\n[PRODUCER] Interrupted by user")
        
        finally:
            self.close()
            self.simulator.print_statistics()
    
    def start_streaming_real(self, duration_minutes=None):
        """Start real streaming mode (polls VNDirect API)."""
        print(f"\n{'='*80}")
        print(f"Starting REAL API Streaming Producer")
        print(f"{'='*80}")
        print(f"Stocks: {len(STOCK_SYMBOLS)}")
        print(f"Polling Interval: {self.polling_interval}s")
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
        else:
            print(f"Duration: Indefinite (Ctrl+C to stop)")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                cycle_start = datetime.now()
                
                print(f"\n--- Cycle {cycle_count} at {cycle_start.strftime('%H:%M:%S')} ---")
                
                success_count = 0
                for stock in STOCK_SYMBOLS:
                    # Fetch from API
                    price_data = self.fetch_latest_price_from_api(stock)
                    
                    if price_data:
                        # Add stock_symbol to data
                        price_data['stock_symbol'] = stock
                        price_data['timestamp'] = datetime.now().isoformat()
                        
                        # Publish
                        if self.publish_tick(price_data):
                            success_count += 1
                    
                    time.sleep(0.1)  # Small delay between stocks
                
                print(f"[PRODUCER] Cycle {cycle_count}: {success_count}/{len(STOCK_SYMBOLS)} published")
                
                # Check duration
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        print(f"\n[PRODUCER] Duration limit reached ({duration_minutes} min)")
                        break
                
                # Wait for next cycle
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, self.polling_interval - cycle_duration)
                
                if sleep_time > 0:
                    print(f"[PRODUCER] Sleeping {sleep_time:.1f}s until next cycle...")
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n[PRODUCER] Interrupted by user")
        
        finally:
            self.close()
    
    def start_streaming(self, duration_minutes=None):
        """
        Start streaming (mode determined by self.mode).
        
        Args:
            duration_minutes: Duration in minutes. None = run indefinitely.
        """
        if self.mode == 'simulated':
            self.start_streaming_simulated(duration_minutes)
        else:
            self.start_streaming_real(duration_minutes)
    
    def close(self):
        """Close Kafka producer connection."""
        print("\n[PRODUCER] Flushing remaining messages...")
        self.producer.flush()
        self.producer.close()
        print("[PRODUCER] Connection closed")


def run_producer(duration_minutes=None, mode=None):
    """
    Main function to run Kafka producer.
    
    Args:
        duration_minutes: Run duration in minutes
        mode: 'simulated' or 'real'
    """
    producer = StockPriceProducer(mode=mode)
    producer.start_streaming(duration_minutes=duration_minutes)


if __name__ == "__main__":
    # Run with mode from config
    run_producer()
