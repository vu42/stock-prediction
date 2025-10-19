"""
Simulated Data Generator for Real-time Streaming Demo

Generates fake stock price ticks to simulate real-time streaming
without calling VNDirect API repeatedly.
"""
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List

from config import STOCK_SYMBOLS


class SimulatedDataGenerator:
    """
    Simulates real-time stock price movements using random walk algorithm.
    
    This generator creates realistic intraday price ticks for demonstration
    purposes without calling external APIs.
    """
    
    def __init__(self, stock_symbols: List[str], base_prices: Dict[str, float] = None):
        """
        Initialize the simulator.
        
        Args:
            stock_symbols: List of stock symbols to simulate
            base_prices: Optional dict of initial prices. If None, uses defaults.
        """
        self.stock_symbols = stock_symbols
        
        # Initialize base prices (typical VN30 price ranges)
        self.current_prices = base_prices or self._get_default_base_prices()
        
        # Simulation parameters
        self.volatility = 0.005  # 0.5% standard deviation per tick
        self.drift = 0.0001      # Slight upward bias (0.01% per tick)
        self.min_volume = 1000
        self.max_volume = 100000
        
        # Trading session info
        self.session_start = datetime.now().replace(hour=9, minute=0, second=0)
        self.session_end = datetime.now().replace(hour=15, minute=0, second=0)
        
        # Statistics tracking
        self.tick_count = 0
        self.start_time = datetime.now()
        
        print(f"[SIMULATOR] Initialized for {len(stock_symbols)} stocks")
        print(f"[SIMULATOR] Volatility: {self.volatility*100:.2f}% | Drift: {self.drift*100:.3f}%")
    
    def _get_default_base_prices(self) -> Dict[str, float]:
        """Get default base prices for VN30 stocks (approximate real prices)."""
        default_prices = {
            "VCB": 89000, "BID": 48500, "CTG": 34200, "TCB": 26500, "VPB": 22800,
            "MBB": 25400, "HDB": 28300, "ACB": 24100, "TPB": 45200, "STB": 36500,
            "VHM": 41200, "VIC": 38900, "VRE": 24600, "KDH": 31500, "NVL": 12800,
            "HPG": 25800, "HSG": 18400, "SSI": 32100, "VCI": 42300, "HCM": 27900,
            "FPT": 124000, "VNM": 82500, "MSN": 67800, "MWG": 52300, "SAB": 185000,
            "GAS": 98500, "PLX": 47200, "POW": 11300, "GVR": 18600, "BVH": 54300,
        }
        
        # Return prices for requested symbols, default to 50000 if not in dict
        return {symbol: default_prices.get(symbol, 50000.0) for symbol in self.stock_symbols}
    
    def generate_tick(self, stock_symbol: str) -> Dict:
        """
        Generate a single price tick for a stock using random walk.
        
        Args:
            stock_symbol: Stock symbol to generate tick for
            
        Returns:
            Dict with tick data (timestamp, price, volume, bid, ask, etc.)
        """
        # Get current price
        current_price = self.current_prices[stock_symbol]
        
        # Random walk: Geometric Brownian Motion
        # dS = S * (drift * dt + volatility * random_normal * sqrt(dt))
        random_shock = random.gauss(0, 1)  # Standard normal distribution
        price_change_pct = self.drift + self.volatility * random_shock
        
        # Calculate new price
        new_price = current_price * (1 + price_change_pct)
        
        # Ensure price doesn't go below floor (10% of base)
        base_price = self._get_default_base_prices()[stock_symbol]
        new_price = max(new_price, base_price * 0.1)
        
        # Round to appropriate precision
        new_price = round(new_price, 2)
        
        # Update current price
        self.current_prices[stock_symbol] = new_price
        
        # Generate bid/ask spread (0.1-0.2%)
        spread_pct = random.uniform(0.001, 0.002)
        bid_price = round(new_price * (1 - spread_pct/2), 2)
        ask_price = round(new_price * (1 + spread_pct/2), 2)
        
        # Generate volume
        volume = random.randint(self.min_volume, self.max_volume)
        
        # Increment tick counter
        self.tick_count += 1
        
        # Create tick data
        tick = {
            'stock_symbol': stock_symbol,
            'timestamp': datetime.now().isoformat(),
            'price': new_price,
            'bid': bid_price,
            'ask': ask_price,
            'volume': volume,
            'change': round(price_change_pct * 100, 3),  # Percentage change
            'tick_id': self.tick_count,
            'is_simulated': True,
            'data_source': 'simulator'
        }
        
        return tick
    
    def generate_batch(self, batch_size: int = None) -> List[Dict]:
        """
        Generate a batch of ticks (one per stock or specified size).
        
        Args:
            batch_size: Number of ticks to generate. If None, generates one per stock.
            
        Returns:
            List of tick dictionaries
        """
        if batch_size is None:
            batch_size = len(self.stock_symbols)
        
        ticks = []
        for i in range(batch_size):
            stock = self.stock_symbols[i % len(self.stock_symbols)]
            tick = self.generate_tick(stock)
            ticks.append(tick)
        
        return ticks
    
    def is_trading_hours(self) -> bool:
        """Check if current time is within trading hours (9 AM - 3 PM)."""
        now = datetime.now()
        current_time = now.time()
        return (9 <= now.hour < 15) or True  # Always True for demo purposes
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics."""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        ticks_per_second = self.tick_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_ticks': self.tick_count,
            'elapsed_seconds': round(elapsed_time, 2),
            'ticks_per_second': round(ticks_per_second, 2),
            'stocks': len(self.stock_symbols),
            'current_prices': self.current_prices.copy()
        }
    
    def print_statistics(self):
        """Print simulation statistics."""
        stats = self.get_statistics()
        print(f"\n[SIMULATOR] Statistics:")
        print(f"  Total Ticks: {stats['total_ticks']}")
        print(f"  Elapsed: {stats['elapsed_seconds']}s")
        print(f"  Rate: {stats['ticks_per_second']} ticks/sec")
        print(f"  Stocks: {stats['stocks']}")
    
    def stream_continuous(self, interval_seconds: float = 1.0, duration_minutes: int = None):
        """
        Stream ticks continuously at specified interval.
        
        Args:
            interval_seconds: Time between batches (default: 1 second)
            duration_minutes: Total duration in minutes. None = infinite.
            
        Yields:
            Batches of tick data
        """
        start_time = datetime.now()
        batch_num = 0
        
        print(f"\n[SIMULATOR] Starting continuous stream...")
        print(f"  Interval: {interval_seconds}s")
        print(f"  Duration: {duration_minutes} min" if duration_minutes else "  Duration: Infinite")
        print(f"  Stocks per batch: {len(self.stock_symbols)}")
        
        try:
            while True:
                batch_start = time.time()
                batch_num += 1
                
                # Generate batch (one tick per stock)
                batch = self.generate_batch()
                
                # Yield batch
                yield batch
                
                # Check duration limit
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        print(f"\n[SIMULATOR] Duration limit reached ({duration_minutes} min)")
                        break
                
                # Print progress every 10 batches
                if batch_num % 10 == 0:
                    print(f"[SIMULATOR] Batch {batch_num} | {len(batch)} ticks | Total: {self.tick_count}")
                
                # Sleep to maintain interval
                elapsed = time.time() - batch_start
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n[SIMULATOR] Interrupted by user")
        
        finally:
            self.print_statistics()


# Helper function for testing
def test_simulator():
    """Test the simulator with a few stocks."""
    print("="*80)
    print("Testing Simulated Data Generator")
    print("="*80)
    
    # Test with 5 stocks
    test_stocks = ["VCB", "FPT", "HPG", "VNM", "VIC"]
    simulator = SimulatedDataGenerator(test_stocks)
    
    # Generate 10 batches
    print("\nGenerating 10 batches...")
    for i, batch in enumerate(simulator.stream_continuous(interval_seconds=0.5, duration_minutes=0.1), 1):
        print(f"\nBatch {i}:")
        for tick in batch:
            print(f"  {tick['stock_symbol']:5s} | Price: {tick['price']:>10,.2f} | "
                  f"Change: {tick['change']:>6.2f}% | Volume: {tick['volume']:>7,}")
        
        if i >= 10:
            break
    
    # Print final statistics
    simulator.print_statistics()
    print("\n" + "="*80)


if __name__ == "__main__":
    test_simulator()

