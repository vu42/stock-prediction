"""
Tick Aggregator - Convert Intraday Ticks to Daily OHLCV

Aggregates streaming tick data into daily OHLCV (Open, High, Low, Close, Volume)
format for database storage and model training.
"""
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

from modules.database import insert_stock_data


class TickAggregator:
    """
    In-memory aggregator for converting intraday ticks to daily OHLCV.
    
    Maintains a cache of ticks and periodically flushes aggregated
    data to PostgreSQL database.
    """
    
    def __init__(self, flush_threshold: int = 1000):
        """
        Initialize the aggregator.
        
        Args:
            flush_threshold: Number of ticks to cache before auto-flushing
        """
        # Cache: {(stock_symbol, date): [tick1, tick2, ...]}
        self.daily_cache = {}
        
        # Threshold for auto-flush
        self.flush_threshold = flush_threshold
        
        # Statistics
        self.total_ticks_received = 0
        self.total_records_flushed = 0
        
        print(f"[AGGREGATOR] Initialized with flush threshold: {flush_threshold}")
    
    def add_tick(self, tick: Dict):
        """
        Add a single tick to the cache.
        
        Args:
            tick: Dictionary with keys: stock_symbol, timestamp, price, volume, etc.
        """
        try:
            stock_symbol = tick.get('stock_symbol')
            timestamp = tick.get('timestamp', datetime.now().isoformat())
            
            # Extract date from timestamp (YYYY-MM-DD)
            date = timestamp[:10]
            
            # Cache key
            key = (stock_symbol, date)
            
            # Initialize list if first tick for this stock-date
            if key not in self.daily_cache:
                self.daily_cache[key] = []
            
            # Add tick to cache
            self.daily_cache[key].append(tick)
            
            self.total_ticks_received += 1
            
            # Log every 100 ticks
            if self.total_ticks_received % 100 == 0:
                print(f"[AGGREGATOR] Received {self.total_ticks_received} ticks | "
                      f"Cached: {len(self.daily_cache)} unique (stock, date) pairs")
        
        except Exception as e:
            print(f"[AGGREGATOR] Error adding tick: {str(e)}")
    
    def aggregate_to_ohlcv(self, stock_symbol: str, date: str) -> Dict:
        """
        Aggregate ticks for a specific stock-date into OHLCV format.
        
        Args:
            stock_symbol: Stock symbol
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Dictionary with OHLCV data
        """
        key = (stock_symbol, date)
        ticks = self.daily_cache.get(key, [])
        
        if not ticks:
            return None
        
        # Convert to DataFrame for easy aggregation
        df = pd.DataFrame(ticks)
        
        # Sort by timestamp to ensure correct open/close
        df = df.sort_values('timestamp')
        
        # Extract OHLCV
        ohlcv = {
            'stock_symbol': stock_symbol,
            'date': date,
            'open': float(df['price'].iloc[0]),      # First price
            'high': float(df['price'].max()),         # Highest price
            'low': float(df['price'].min()),          # Lowest price
            'close': float(df['price'].iloc[-1]),     # Last price
            'volume': int(df['volume'].sum())         # Total volume
        }
        
        return ohlcv
    
    def should_flush(self) -> bool:
        """
        Check if cache should be flushed based on threshold.
        
        Returns:
            bool: True if should flush
        """
        total_ticks = sum(len(ticks) for ticks in self.daily_cache.values())
        return total_ticks >= self.flush_threshold
    
    def flush_to_database(self) -> int:
        """
        Aggregate all cached ticks and flush to database.
        
        Returns:
            int: Number of records flushed
        """
        if not self.daily_cache:
            return 0
        
        print(f"\n[AGGREGATOR] Flushing {len(self.daily_cache)} (stock, date) pairs to database...")
        
        flushed_count = 0
        
        try:
            for (stock_symbol, date) in list(self.daily_cache.keys()):
                # Aggregate to OHLCV
                ohlcv = self.aggregate_to_ohlcv(stock_symbol, date)
                
                if ohlcv:
                    # Convert to DataFrame for database insert
                    df = pd.DataFrame([ohlcv])
                    
                    # Insert to database
                    inserted = insert_stock_data(df, stock_symbol)
                    
                    if inserted > 0:
                        flushed_count += 1
                        print(f"[AGGREGATOR] ✓ {stock_symbol} {date} | "
                              f"O:{ohlcv['open']:.2f} H:{ohlcv['high']:.2f} "
                              f"L:{ohlcv['low']:.2f} C:{ohlcv['close']:.2f} "
                              f"V:{ohlcv['volume']:,}")
                
                # Remove from cache after flush
                del self.daily_cache[(stock_symbol, date)]
            
            self.total_records_flushed += flushed_count
            
            print(f"[AGGREGATOR] ✓ Flushed {flushed_count} records to database")
            print(f"[AGGREGATOR] Total stats: {self.total_ticks_received} ticks → "
                  f"{self.total_records_flushed} OHLCV records\n")
            
            return flushed_count
        
        except Exception as e:
            print(f"[AGGREGATOR] ✗ Error flushing to database: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0
    
    def get_cache_stats(self) -> Dict:
        """
        Get current cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_ticks = sum(len(ticks) for ticks in self.daily_cache.values())
        
        return {
            'unique_keys': len(self.daily_cache),
            'total_ticks_cached': total_ticks,
            'total_ticks_received': self.total_ticks_received,
            'total_records_flushed': self.total_records_flushed,
            'flush_threshold': self.flush_threshold
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_cache_stats()
        print(f"\n[AGGREGATOR] Statistics:")
        print(f"  Unique (stock, date) pairs: {stats['unique_keys']}")
        print(f"  Ticks in cache: {stats['total_ticks_cached']}")
        print(f"  Total ticks received: {stats['total_ticks_received']}")
        print(f"  Total records flushed: {stats['total_records_flushed']}")
        print(f"  Flush threshold: {stats['flush_threshold']}")
    
    def force_flush_all(self):
        """Force flush all cached data regardless of threshold."""
        print(f"[AGGREGATOR] Force flushing all cached data...")
        return self.flush_to_database()
    
    def clear_cache(self):
        """Clear all cached data without flushing."""
        self.daily_cache.clear()
        print(f"[AGGREGATOR] Cache cleared")


# Helper function for testing
def test_aggregator():
    """Test the TickAggregator with sample data."""
    print("="*80)
    print("Testing TickAggregator")
    print("="*80)
    
    aggregator = TickAggregator(flush_threshold=10)
    
    # Simulate some ticks
    for i in range(15):
        tick = {
            'stock_symbol': 'VCB',
            'timestamp': f'2025-10-19T09:{i:02d}:00',
            'price': 89000 + (i * 100),  # Price increases
            'volume': 1000 * (i + 1)
        }
        aggregator.add_tick(tick)
    
    # Print stats
    aggregator.print_stats()
    
    # Force flush
    aggregator.force_flush_all()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_aggregator()

