# VN30 Stock Price Prediction System

A production-ready machine learning system for predicting stock prices of Vietnam's top 30 blue-chip companies (VN30) using LSTM neural networks and Apache Airflow for workflow orchestration.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [DAG Workflows](#dag-workflows)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This system provides automated stock price prediction for VN30 stocks using deep learning. It features incremental data crawling from VNDirect API, LSTM-based price forecasting, and comprehensive reporting through email notifications.

**Key Capabilities:**
- Incremental data collection (avoids duplicate data)
- LSTM neural network training (100 epochs)
- 30-day future price predictions
- Model performance evaluation with RMSE/MAE metrics
- Automated daily updates via Airflow
- PostgreSQL for reliable data storage

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Apache Airflow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  DAG 1: Data Crawler (5:00 PM Daily)                            │
│  └─> Fetch incremental data from VNDirect API                   │
│      └─> Store in PostgreSQL (UPSERT)                           │
│                                                                   │
│  DAG 2: Model Training (6:00 PM Daily)                          │
│  └─> Load data from PostgreSQL                                  │
│      └─> Train/Update LSTM models (30 stocks)                   │
│          └─> Evaluate & Predict future prices                   │
│              └─> Send email report                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                        ┌──────────────┐
    │ PostgreSQL  │                        │    Output    │
    │  Database   │                        │   Files      │
    │             │                        │  (models,    │
    │ - stock_    │                        │   charts,    │
    │   prices    │                        │   CSVs)      │
    │ - crawl_    │                        └──────────────┘
    │   metadata  │
    └─────────────┘
```

## Features

### Data Collection
- **Incremental Crawling**: Only fetches new data since last update
- **Automatic Deduplication**: PostgreSQL UNIQUE constraint prevents duplicates
- **VNDirect API Integration**: Reliable data source for Vietnamese stocks
- **Metadata Tracking**: Monitors last crawl date and record counts

### Machine Learning
- **LSTM Architecture**: 4-layer LSTM network with dropout
- **100 Epochs Training**: Ensures model accuracy
- **Sequence Length**: 60 days lookback window
- **Future Predictions**: 30-day ahead forecasting
- **Model Persistence**: Saves trained models for reuse

### Evaluation & Reporting
- **Performance Metrics**: RMSE and MAE calculation
- **Visualization**: Comparison charts (actual vs predicted)
- **Email Reports**: Automated summary emails with attachments
- **CSV Exports**: Prediction results in CSV format

## Requirements

### System Requirements
- Python 3.8+
- PostgreSQL 12+
- 8GB RAM minimum (16GB recommended for training)
- 10GB disk space for models and data

### Python Dependencies
```
apache-airflow==2.7.3
pendulum==2.1.2
pandas==2.0.3
numpy==1.24.3
tensorflow==2.13.0
keras==2.13.1
scikit-learn==1.3.0
matplotlib==3.7.2
psycopg2-binary==2.9.9
SQLAlchemy==2.0.23
sendgrid==6.10.0
```

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd stock-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup PostgreSQL

**Option A: Using Docker (Recommended)**
```bash
docker run -d \
  --name stock-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=stock_prediction \
  -p 5432:5432 \
  postgres:15
```

**Option B: Local Installation**
```bash
# macOS
brew install postgresql@15
brew services start postgresql@15
createdb stock_prediction

# Ubuntu/Debian
sudo apt-get install postgresql
sudo -u postgres createdb stock_prediction
```

### 4. Initialize Database
Database tables will be created automatically on first run by the init_database task in the DAG.

### 5. Configure Airflow
```bash
# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Copy DAGs to Airflow folder
cp -r dags/* ~/airflow/dags/
```

## Configuration

Edit `config.py` to customize settings:

```python
# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'stock_prediction',
    'user': 'postgres',
    'password': 'postgres'
}

# Stock Selection
STOCK_SYMBOLS = VN30_STOCKS  # All 30 stocks
# STOCK_SYMBOLS = ["VCB", "FPT"]  # Or specific stocks

# Model Parameters
SEQUENCE_LENGTH = 60      # Days of historical data to use
TRAINING_EPOCHS = 100     # Training iterations
FUTURE_DAYS = 30          # Days ahead to predict

# Email Configuration
EMAIL_CONFIG = {
    'sender': 'your@email.com',
    'recipient': 'recipient@email.com',
    'sendgrid_api_key': 'YOUR_API_KEY'
}
```

## Usage

### Start Airflow
```bash
# Start scheduler and webserver
airflow scheduler &
airflow webserver -p 8080
```

### Access Airflow UI
Open browser: `http://localhost:8080`

Username: `admin`  
Password: `admin`

### Enable DAGs
1. Navigate to DAGs page
2. Toggle ON for both DAGs:
   - `vn30_data_crawler`
   - `vn30_model_training`

### Manual Trigger
Click the "play" button on either DAG to run immediately.

### Schedule
- **Data Crawler**: Runs daily at 5:00 PM Vietnam Time
- **Model Training**: Runs daily at 6:00 PM Vietnam Time (1 hour after crawler)

## Project Structure

```
stock-prediction/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── dags/                    # Airflow DAG definitions
│   ├── vn30_data_crawler.py        # Data collection DAG
│   └── vn30_model_training.py      # Model training DAG
│
├── modules/                 # Core business logic
│   ├── __init__.py
│   ├── database.py                 # PostgreSQL operations
│   ├── data_fetcher.py             # API data collection
│   ├── model_trainer.py            # LSTM training & prediction
│   ├── orchestrator.py             # Workflow coordination
│   └── email_notifier.py           # Email reporting
│
└── output/                  # Generated files (gitignored)
    └── {STOCK_SYMBOL}/
        ├── {STOCK}_price.csv              # Historical data backup
        ├── {STOCK}_model.h5               # Trained LSTM model
        ├── {STOCK}_scaler.pkl             # Data normalization scaler
        ├── {STOCK}_evaluation.png         # Performance chart
        ├── {STOCK}_future.png             # Prediction chart
        └── {STOCK}_future_predictions.csv # Prediction data
```

## Database Schema

### Table: stock_prices
Primary table storing daily stock price data.

```sql
CREATE TABLE stock_prices (
    id SERIAL PRIMARY KEY,
    stock_symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(15, 2),
    high DECIMAL(15, 2),
    low DECIMAL(15, 2),
    close DECIMAL(15, 2) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stock_symbol, date)
);

CREATE INDEX idx_stock_date ON stock_prices(stock_symbol, date DESC);
```

### Table: crawl_metadata
Tracks crawling status for incremental updates.

```sql
CREATE TABLE crawl_metadata (
    stock_symbol VARCHAR(10) PRIMARY KEY,
    last_crawl_date DATE,
    last_data_date DATE,
    total_records INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Useful Queries

**Check data coverage:**
```sql
SELECT 
    stock_symbol,
    MIN(date) as first_date,
    MAX(date) as last_date,
    COUNT(*) as total_records
FROM stock_prices
GROUP BY stock_symbol
ORDER BY stock_symbol;
```

**View recent prices:**
```sql
SELECT * FROM stock_prices 
WHERE stock_symbol = 'VCB' 
ORDER BY date DESC 
LIMIT 30;
```

## DAG Workflows

### DAG 1: vn30_data_crawler

**Schedule**: Daily at 5:00 PM (Vietnam Time)  
**Duration**: 5-10 minutes  
**Purpose**: Incremental data collection

**Tasks:**
1. `init_database`: Create tables if not exist
2. `crawl_{STOCK}`: Fetch new data for each stock (parallel execution)

**Incremental Logic:**
- Queries database for last date
- Only fetches data from (last_date + 1) to today
- Inserts/updates with UPSERT to avoid duplicates

### DAG 2: vn30_model_training

**Schedule**: Daily at 6:00 PM (Vietnam Time)  
**Duration**: 2-3 hours  
**Purpose**: Model training and prediction

**Tasks:**
1. `wait_for_data_crawler`: Waits for crawler DAG to complete
2. `train_{STOCK}`: Train LSTM model for each stock
3. `evaluate_{STOCK}`: Evaluate model performance
4. `predict_{STOCK}`: Generate 30-day predictions
5. `send_email_report`: Email summary with attachments

## API Reference

### modules.database

**get_db_connection()**: Returns PostgreSQL connection  
**init_database()**: Creates database tables  
**get_last_data_date(stock_symbol)**: Returns last date in DB  
**insert_stock_data(df, stock_symbol)**: Inserts data with UPSERT  
**load_stock_data_from_db(stock_symbol)**: Loads all data for a stock  

### modules.data_fetcher

**fetch_stock_data(stock_symbol, **context)**: Crawls data incrementally  
**fetch_all_stocks(stock_symbols, **context)**: Crawls multiple stocks  

### modules.model_trainer

**train_prediction_model(stock_symbol)**: Trains LSTM model  
**evaluate_model(stock_symbol)**: Evaluates model performance  
**predict_future_prices(stock_symbol)**: Generates predictions  

### modules.email_notifier

**send_email_notification(stock_symbols)**: Sends summary email  

## Troubleshooting

### Database Connection Error
```bash
# Check PostgreSQL is running
docker ps | grep stock-postgres

# Restart if needed
docker start stock-postgres
```

### Import Error
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path in DAGs
import sys
print(sys.path)
```

### Training Takes Too Long
- Reduce `TRAINING_EPOCHS` in config.py (e.g., 50 instead of 100)
- Train fewer stocks: `STOCK_SYMBOLS = ["VCB", "FPT"]`

### Email Not Sending
- Verify SendGrid API key in `config.py`
- Check sender email is verified in SendGrid
- Review Airflow logs for error details

### Data Not Updating
- Check crawler DAG ran successfully
- Verify database connection
- Review logs: `docker logs stock-postgres`

## License

MIT License - Free to use and modify

## Author

Original: TuPH  
Enhanced Version: 2025

## Acknowledgments

- VNDirect API for stock market data
- Apache Airflow for workflow orchestration
- TensorFlow/Keras for LSTM implementation
