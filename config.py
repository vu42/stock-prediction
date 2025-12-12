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
    "host": "localhost",
    "port": 5432,
    "database": "stock_prediction",
    "user": "postgres",
    "password": "postgres",
}

# ============================================================================
# VN30 STOCKS
# ============================================================================
VN30_STOCKS = [
    "FPT",
    "VCB",
    "VNM",
    "HPG",
    "VIC",
    "VHM",
    "MSN",
    "SAB"
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
SEQUENCE_LENGTH = 60  # Days to look back
FUTURE_DAYS = 30  # Days to predict ahead (for recursive model)

# Direct Multi-Horizon Prediction
# Each horizon gets its own dedicated model trained directly on that target
PREDICTION_HORIZONS = [7, 15, 30]  # Days ahead to predict directly

# Scikit-learn Model Configuration
RANDOM_FOREST_ESTIMATORS = 100
GRADIENT_BOOSTING_ESTIMATORS = 100
SVR_C = 1.0
RIDGE_ALPHA = 1.0

# Incremental Training Configuration
CONTINUE_TRAINING = True  # If True, load existing model and continue training
# If False, train from scratch each time

# ============================================================================
# FILE PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# ============================================================================
# EMAIL CONFIGURATION
# ============================================================================
EMAIL_CONFIG = {
    "sender": "tuph.alex@gmail.com",
    "recipient": "tu.phancshcmut16@hcmut.edu.vn",
    "sendgrid_api_key": "YOUR_SENDGRID_API_KEY",  # Replace with actual key
}

# ============================================================================
# AIRFLOW SCHEDULE
# ============================================================================
SCHEDULE_TIME = "0 17 * * *"  # 5:00 PM daily (Vietnam time)
