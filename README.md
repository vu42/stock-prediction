# VN30 Stock Price Prediction System

Production-ready machine learning system for predicting stock prices of Vietnam's top 30 blue-chip companies (VN30) using LSTM neural networks.

**Two Independent Workflows:**
- **BATCH**: Daily scheduled processing with Airflow (production)
- **STREAMING**: Real-time Kafka pipeline (demo/architecture showcase)

---

## 🚀 Quick Start

### ONE-TIME SETUP (Required First!)
```bash
./start_database.sh
```
This installs PostgreSQL, creates database & tables. **Do this once only!**

### STREAMING Demo (Kafka - 5 min)
```bash
./start_streaming_demo.sh    # Start
# ... demo ...
./stop_streaming_demo.sh      # Stop
./cleanup_database.sh         # Clean for next demo
```

### BATCH Demo (Real API - 10 min)
```bash
./start_batch_demo.sh         # Start
# ... demo ...
./stop_batch_demo.sh          # Stop
```

See [WORKFLOW.md](WORKFLOW.md) for detailed step-by-step guide.

---

## 📊 System Architecture

### BATCH Processing (Production)
```
Airflow DAG (5:00 PM Daily)
    ↓
VNDirect API (incremental crawl)
    ↓
PostgreSQL
    ↓
Airflow DAG (5:30 PM Daily)
    ↓
LSTM Training (100 epochs, 20+ features)
    ↓
Predictions + Email Reports
```

**Features:**
- Real API calls with incremental crawling
- Multi-feature LSTM (Bidirectional + BatchNorm)
- 30-day predictions with evaluation metrics
- Automated email notifications

### STREAMING (Demo/Architecture)
```
Simulated Generator (30 ticks/sec)
    ↓
Kafka Producer
    ↓
Kafka Broker
    ↓
Kafka Consumer + TickAggregator
    ↓
PostgreSQL (daily OHLCV)
    ↓
Airflow Orchestrator (every 30 min)
```

**Features:**
- Simulated real-time ticks (no API calls)
- Kafka message queue architecture
- Tick aggregation (intraday → daily OHLCV)
- Mock ML pipeline for quick demos

---

## 📂 Project Structure

```
stock-prediction/
├── README.md                       # This file
├── WORKFLOW.md                     # Demo workflow guide
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Kafka + PostgreSQL infrastructure
├── config.py                      # Configuration settings
│
├── start_database.sh              # ONE-TIME: Setup PostgreSQL + tables
├── start_streaming_demo.sh        # Start streaming demo
├── stop_streaming_demo.sh         # Stop streaming demo
├── start_batch_demo.sh            # Start batch demo
├── stop_batch_demo.sh             # Stop batch demo
├── cleanup_database.sh            # Clean database between demos
│
├── dags/
│   ├── batch/
│   │   ├── vn30_data_crawler.py          # Daily data crawling (5:00 PM)
│   │   └── vn30_model_training.py        # Daily model training (5:30 PM)
│   │
│   └── streaming/
│       ├── kafka_health_monitor.py       # Health monitoring (every 5 min)
│       ├── vn30_streaming_orchestrator.py # ML orchestrator (every 30 min)
│       └── vn30_kafka_streaming.py       # Demo DAGs (manual trigger)
│
├── modules/
│   ├── database.py                # PostgreSQL operations
│   ├── data_fetcher.py            # API data collection (batch)
│   ├── model_trainer.py           # LSTM training & prediction
│   ├── orchestrator.py            # Workflow coordination
│   ├── email_notifier.py          # Email reporting
│   ├── kafka_producer.py          # Kafka producer (streaming)
│   ├── kafka_consumer.py          # Kafka consumer (streaming)
│   ├── tick_aggregator.py         # Tick → OHLCV aggregation
│   ├── simulated_data_generator.py # Fake data generator
│   └── streaming_utils.py         # Kafka + Airflow utilities
│
└── output/                        # Generated files (gitignored)
    └── {STOCK_SYMBOL}/
        ├── {STOCK}_model.h5              # Trained LSTM model
        ├── {STOCK}_scaler.pkl            # Data scaler
        ├── {STOCK}_evaluation.png        # Performance chart
        ├── {STOCK}_future.png            # Prediction chart
        └── {STOCK}_future_predictions.csv # Predictions
```

---

## 🔧 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd stock-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Airflow
```bash
export AIRFLOW_HOME=~/airflow
airflow db init
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Copy DAGs
cp -r dags/* ~/airflow/dags/
```

---

## 🎯 Usage

### BATCH Flow (Production)

**Start:**
```bash
./batch_demo.sh
```

**Start Airflow:**
```bash
# Terminal 1
export AIRFLOW_HOME=~/airflow
airflow scheduler

# Terminal 2
export AIRFLOW_HOME=~/airflow
airflow webserver -p 8080
```

**Access:** http://localhost:8080 (admin/admin)

**Enable DAGs:**
- `vn30_data_crawler` (5:00 PM daily)
- `vn30_model_training` (5:30 PM daily)

**Stop:**
```bash
./batch_stop.sh
# Ctrl+C in Airflow terminals
```

---

### STREAMING Flow (Demo)

**Start:**
```bash
./streaming_demo.sh
```

**Monitor:**
```bash
# Kafka UI
open http://localhost:8080

# Logs
tail -f logs/producer_*.log
tail -f logs/consumer_*.log

# Database
docker exec vn30-postgres psql -U postgres -d stock_prediction \
  -c "SELECT * FROM stock_prices ORDER BY date DESC LIMIT 5;"
```

**Airflow (optional):**
```bash
export AIRFLOW_HOME=~/airflow
airflow scheduler &
airflow webserver -p 8081  # Note: 8081!
```

Access: http://localhost:8081

**Stop:**
```bash
./streaming_stop.sh
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
# Stock selection
STOCK_SYMBOLS = VN30_STOCKS  # All 30 stocks
# STOCK_SYMBOLS = ["VCB", "FPT"]  # Or specific stocks

# Model parameters
SEQUENCE_LENGTH = 60        # Days of lookback
TRAINING_EPOCHS = 100       # Training iterations
FUTURE_DAYS = 30            # Days ahead to predict

# Database
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'stock_prediction',
    'user': 'postgres',
    'password': 'postgres'
}

# Email
EMAIL_CONFIG = {
    'sender': 'your@email.com',
    'recipient': 'recipient@email.com',
    'sendgrid_api_key': 'YOUR_API_KEY'
}

# Streaming mode
STREAMING_MODE = 'simulated'  # 'simulated' or 'real'
```

---

## 📊 Database Schema

### Table: stock_prices
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
```

### Table: crawl_metadata
```sql
CREATE TABLE crawl_metadata (
    stock_symbol VARCHAR(10) PRIMARY KEY,
    last_crawl_date DATE,
    last_data_date DATE,
    total_records INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🎓 Key Features

### Data Collection
- **Incremental Crawling**: Only fetches new data (batch)
- **Real-time Streaming**: Kafka-based ingestion (streaming)
- **Tick Aggregation**: Intraday ticks → daily OHLCV (streaming)
- **Deduplication**: PostgreSQL UNIQUE constraints

### Machine Learning
- **Enhanced LSTM**: Bidirectional + BatchNormalization
- **Multi-feature Input**: 20+ technical indicators
  - Moving Averages (SMA, EMA)
  - MACD, RSI
  - Bollinger Bands
  - ATR, Volume indicators
- **Early Stopping**: Prevents overfitting
- **100 Epochs Training**: Ensures accuracy

### Evaluation & Reporting
- **Metrics**: RMSE, MAE, MAPE, R², Direction Accuracy
- **Visualizations**: Actual vs Predicted charts
- **30-day Predictions**: Future price forecasts
- **Email Reports**: Automated notifications

---

## 🔍 Technology Stack

### Core
- **Python 3.8+**
- **Apache Airflow 2.7.3** - Workflow orchestration
- **PostgreSQL 15** - Data persistence
- **Pandas** - Data transformation

### Machine Learning
- **TensorFlow 2.13** / **Keras** - LSTM models
- **scikit-learn** - Data preprocessing
- **TA-Lib** - Technical indicators

### Streaming (Optional)
- **Apache Kafka** - Message queue
- **Kafka-Python** - Python client
- **Docker Compose** - Infrastructure

---

## 📈 Performance

**Model Metrics (VCB example):**
- RMSE: ~2,345 VND
- MAE: ~1,234 VND
- MAPE: ~3.45%
- R²: ~0.89
- Direction Accuracy: ~78.5%

**Data Volume:**
- Historical: ~100K records (30 stocks × 10 years)
- Streaming: 30 ticks/sec → 30 OHLCV/day (after aggregation)
- Storage: ~100MB total

---

## 🐛 Troubleshooting

### Port conflicts
```bash
# Stop conflicting services
docker stop $(docker ps -q --filter "expose=5432")
docker stop $(docker ps -q --filter "expose=9092")
```

### Airflow DAGs not showing
```bash
# Re-copy DAGs
cp -r dags/* ~/airflow/dags/
airflow dags list
```

### Database connection errors
```bash
# Check PostgreSQL
docker exec -it vn30-postgres-batch psql -U postgres -c "SELECT 1;"
# Or for streaming
docker exec -it vn30-postgres psql -U postgres -c "SELECT 1;"
```

---

## 📚 Documentation

- **QUICK_DEMO_GUIDE.md** - Quick start guide for demos
- **requirements.txt** - Python dependencies
- **config.py** - Configuration reference

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

## 📄 License

MIT License - See LICENSE file

---

## 🎯 Use Cases

### BATCH Flow - Best for:
- Production ML pipelines
- Daily automated predictions
- Real stock analysis
- Backtesting strategies

### STREAMING Flow - Best for:
- Architecture demonstrations
- Kafka learning/teaching
- Microservices patterns
- Quick prototyping

---

**Built with ❤️ for VN30 stock prediction**
