# VN30 Stock Price Prediction System

Machine learning system for predicting Vietnam's VN30 stock prices using ensemble models and Apache Airflow orchestration.

## Overview

Automated stock prediction system featuring:

- **Incremental data collection** from VNDirect API
- **Ensemble ML models** (Random Forest + Gradient Boosting + SVR + Ridge)
- **Technical indicators** (23 features: RSI, MACD, Bollinger Bands, etc.)
- **30-day predictions** with performance evaluation
- **Airflow automation** for daily updates
- **PostgreSQL storage** with UPSERT deduplication

## Architecture

```
Airflow DAGs
├── Data Crawler (5PM daily)    → PostgreSQL (UPSERT)
└── Model Training (6PM daily)  → Train → Evaluate → Predict → Email
```

**Workflow:**

1. Fetch incremental data from VNDirect API
2. Calculate 23 technical indicators
3. Train ensemble models (4 algorithms)
4. Generate 30-day predictions
5. Send email reports with charts

## Training Architecture

### Ensemble Learning Approach

Uses **4 diverse ML algorithms** that vote together:

1. **Random Forest** (Bagging)
   - 100 decision trees, reduces variance
   - Handles non-linear relationships

2. **Gradient Boosting** (Boosting)
   - Sequential learning, corrects previous errors
   - Strong performance on tabular data

3. **Support Vector Regressor** (Kernel Methods)
   - RBF kernel for high-dimensional mapping
   - Robust to outliers

4. **Ridge Regression** (Linear)
   - L2 regularization baseline
   - Fast and interpretable

### Feature Engineering (23 Features)

**Price Indicators:** SMA (5/20/60), EMA (12/26), MACD, Bollinger Bands  
**Momentum:** RSI, ROC, Momentum  
**Volatility:** ATR, BB_position  
**Volume:** Volume_MA, Volume_ratio  
**Crossovers:** SMA crossovers

### Time Series Processing

- **Sliding Window:** 60-day lookback sequences
- **Normalization:** StandardScaler per feature
- **Train/Test Split:** 80/20 chronological split
- **Ensemble Averaging:** Mean of 4 model predictions

## Requirements

- Python 3.8+, PostgreSQL 12+
- 8GB RAM, 10GB disk space
- See `requirements.txt` for dependencies

## Quick Start

```bash
# Install
pip install -r requirements.txt
# if you encounter an error, try this:
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt 

# Setup PostgreSQL (Docker) or use your system Postgresql
docker run -d --name stock-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=stock_prediction \
  -p 5432:5432 postgres:15

# Configure Airflow
airflow db migrate
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin \
  --email admin@example.com
cp -r dags/* ~/airflow/dags/

# Start
airflow scheduler &
airflow webserver -p 8080
```

Access: `http://localhost:8080` (admin/admin)

## Configuration

Edit `config.py`:

- Database credentials
- Stock symbols (all VN30 or specific stocks)
- Model parameters (sequence length, epochs)
- Email settings (SendGrid API key)

## Usage

### Option 1: Local Training (Without Airflow)

Train models directly on your laptop:

```bash
# Activate environment
source venv/bin/activate

# Initialize database (first time only)
python init_db.py

# Train default stocks (VCB, FPT)
python train_local.py

# Train specific stock
python train_local.py VCB

# Train multiple stocks
python train_local.py VCB FPT VNM

# Train all VN30 stocks
python train_local.py --all

# Skip data fetching (use existing DB)
python train_local.py VCB --no-fetch

# Continue training existing model
python train_local.py VCB --continue
```

### Option 2: Airflow Automation

Enable DAGs in Airflow UI → Toggle ON `vn30_data_crawler` and `vn30_model_training`

**Schedule:**

- Data Crawler: 5:00 PM daily
- Model Training: 6:00 PM daily

## Model Output

All trained models and results are saved in `output/{STOCK_SYMBOL}/`:

```
output/VCB/                                    # Example for VCB stock
├── VCB_sklearn_model.pkl                     # Trained ensemble (4 models)
├── VCB_sklearn_scaler.pkl                    # Data normalization scaler
├── VCB_sklearn_evaluation.png                # Performance chart
├── VCB_sklearn_future.png                    # 30-day prediction chart
└── VCB_sklearn_future_predictions.csv        # Prediction data (CSV)
```

**View results:**
```bash
# Check output files
ls output/VCB/

# View predictions
cat output/VCB/VCB_sklearn_future_predictions.csv

# Open charts
open output/VCB/VCB_sklearn_evaluation.png
open output/VCB/VCB_sklearn_future.png
```

## Database

**Tables:**

- `stock_prices`: Daily OHLCV data (UNIQUE on stock_symbol, date)
- `crawl_metadata`: Tracks last crawl dates for incremental updates

**Key queries:**

```sql
-- Check coverage
SELECT stock_symbol, MIN(date), MAX(date), COUNT(*) 
FROM stock_prices GROUP BY stock_symbol;

-- Recent prices
SELECT * FROM stock_prices WHERE stock_symbol = 'VCB' 
ORDER BY date DESC LIMIT 30;
```  

## Testing

See `tests/README.md` for details.

## License

MIT License
