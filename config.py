"""
Configuration file for Stock Prediction System
"""
import os

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_START_DATE = "2000-01-01"

# ============================================================================
# DATABASE CONFIGURATION - PostgreSQL
# ============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'stock_prediction',
    'user': 'postgres',
    'password': 'postgres'
}

# ============================================================================
# VN30 STOCKS
# ============================================================================
VN30_STOCKS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "KDH", "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SSI",
    "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB"
]

# Select stocks to process
STOCK_SYMBOLS = VN30_STOCKS  # Process all VN30
# STOCK_SYMBOLS = ["VCB", "FPT"]  # Or specific stocks for testing

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_BASE_URL = "https://api-finfo.vndirect.com.vn/v4/stock_prices"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
SEQUENCE_LENGTH = 60      # Days to look back
LSTM_UNITS = 50           # Units per LSTM layer
DROPOUT_RATE = 0.2        # Dropout rate
TRAINING_EPOCHS = 100     # Training epochs
FUTURE_DAYS = 30          # Days to predict ahead

# ============================================================================
# FILE PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# ============================================================================
# EMAIL CONFIGURATION
# ============================================================================
EMAIL_CONFIG = {
    'sender': 'tuph.alex@gmail.com',
    'recipient': 'tu.phancshcmut16@hcmut.edu.vn',
    'sendgrid_api_key': 'YOUR_SENDGRID_API_KEY'  # Replace with actual key
}

# ============================================================================
# KAFKA CONFIGURATION
# ============================================================================
KAFKA_CONFIG = {
    'bootstrap_servers': ['localhost:9092'],
    'topic_name': 'vn30-stock-prices',
    'consumer_group': 'stock-prediction-consumer-group',
    'polling_interval': 10,  # Poll API every 10 seconds for streaming simulation
    'batch_size': 100,
    'max_poll_records': 500
}

# ============================================================================
# STREAMING MODE CONFIGURATION
# ============================================================================
# Mode: 'simulated' or 'real'
# - simulated: Generate fake ticks (1 tick/second) for demo - NO API CALLS
# - real: Call VNDirect API (every 10s) - returns same data (not real-time)
STREAMING_MODE = 'simulated'  # Change to 'real' to use actual API

# Simulated streaming settings
SIMULATED_STREAMING = {
    'tick_interval': 1.0,       # Generate tick every 1 second
    'volatility': 0.005,        # 0.5% price volatility per tick
    'drift': 0.0001,            # 0.01% upward bias per tick
    'min_volume': 1000,         # Minimum trade volume
    'max_volume': 100000,       # Maximum trade volume
}

# ============================================================================
# AIRFLOW SCHEDULE
# ============================================================================
SCHEDULE_TIME = '0 17 * * *'  # 5:00 PM daily (Vietnam time)

