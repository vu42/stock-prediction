# VN30 Stock Price Prediction System

Machine learning system for predicting Vietnam's VN30 stock prices using ensemble models, FastAPI REST API, and Apache Airflow orchestration.

## Overview

Automated stock prediction system featuring:

- **FastAPI REST API** with JWT authentication and role-based access
- **Incremental data collection** from VNDirect API
- **Ensemble ML models** (Random Forest + Gradient Boosting + SVR + Ridge)
- **Technical indicators** (23 features: RSI, MACD, Bollinger Bands, etc.)
- **30-day predictions** with performance evaluation
- **Airflow automation** for daily updates
- **PostgreSQL storage** with SQLAlchemy ORM
- **RQ Workers** for background job processing
- **S3/MinIO** for artifact storage

## Project Structure

```
stock-prediction/
├── backend/                    # Backend monorepo
│   ├── src/
│   │   ├── app/               # FastAPI Application
│   │   │   ├── api/v1/        # REST API endpoints
│   │   │   ├── core/          # Config, security, logging
│   │   │   ├── db/            # SQLAlchemy models
│   │   │   ├── schemas/       # Pydantic schemas
│   │   │   ├── services/      # Business logic
│   │   │   └── integrations/  # External services
│   │   └── worker/            # RQ background worker
│   ├── migrations/            # Alembic migrations
│   ├── tests/                 # Test suite
│   ├── docker/                # Docker configuration
│   └── pyproject.toml         # Python dependencies
├── dags/                       # Airflow DAGs
│   ├── vn30_data_crawler.py
│   └── vn30_model_training.py
├── docs/                       # Documentation
├── output/                     # Model outputs
├── modules/                    # Legacy modules (deprecated)
└── tests/                      # Legacy tests (deprecated)
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Start all services
cd backend
docker-compose -f docker/docker-compose.dev.yml up -d

# Run migrations
docker exec stock-prediction-api alembic upgrade head

# Access API
open http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# 1. Setup backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Start dependencies
docker-compose -f docker/docker-compose.dev.yml up -d postgres redis minio

# 3. Configure
cp env.example .env
# Edit .env with your settings

# 4. Run migrations
alembic upgrade head

# 5. Start API
uvicorn app.main:app --reload

# 6. Start worker (another terminal)
python -m worker.main
```

### Option 3: Local Training (Without API)

Train models directly using the legacy modules:

```bash
# Activate environment
source venv/bin/activate

# Initialize database
python init_db.py

# Train specific stocks
python train_local.py VCB FPT VNM

# Train all VN30 stocks
python train_local.py --all
```

## API Endpoints

See full API documentation at `http://localhost:8000/docs` after starting the server.

### Key Endpoints

| Category | Endpoint | Description |
|----------|----------|-------------|
| Auth | `POST /api/v1/auth/login` | Get JWT tokens |
| Stocks | `GET /api/v1/stocks/top-picks` | Should Buy/Sell picks |
| Stocks | `GET /api/v1/stocks/market-table` | Market overview |
| Predictions | `GET /api/v1/stocks/{ticker}/chart` | Price + forecast chart |
| Training | `POST /api/v1/experiments/run` | Start training experiment |
| Pipelines | `GET /api/v1/pipeline/dags` | List Airflow DAGs |

## Architecture

### Data Flow

```
VNDirect API → Data Crawler DAG → PostgreSQL
                                       ↓
                              Training DAG / API
                                       ↓
                              ML Models (Ensemble)
                                       ↓
                              Predictions → S3
                                       ↓
                              REST API → Frontend
```

### Ensemble Learning

Uses **4 diverse ML algorithms** that vote together:

1. **Random Forest** - 100 decision trees, reduces variance
2. **Gradient Boosting** - Sequential learning, corrects errors
3. **Support Vector Regressor** - RBF kernel, robust to outliers
4. **Ridge Regression** - L2 regularized linear baseline

### Feature Engineering (23 Features)

- **Price Indicators:** SMA (5/20/60), EMA (12/26), MACD, Bollinger Bands
- **Momentum:** RSI, ROC, Momentum
- **Volatility:** ATR, BB_position
- **Volume:** Volume_MA, Volume_ratio
- **Crossovers:** SMA crossovers

## Configuration

### Environment Variables

Key settings in `backend/.env`:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection |
| `REDIS_URL` | Redis for task queue |
| `JWT_SECRET_KEY` | JWT signing key |
| `SENDGRID_API_KEY` | Email notifications |
| `S3_ENDPOINT_URL` | S3/MinIO for artifacts |

### Airflow DAGs

**Schedule:**
- Data Crawler: 5:00 PM daily (Vietnam time)
- Model Training: 6:00 PM daily (Vietnam time)

## Model Output

Trained models saved in `output/{STOCK_SYMBOL}/`:

```
output/VCB/
├── VCB_model.pkl           # Trained ensemble
├── VCB_scaler.pkl          # Feature scaler
├── VCB_evaluation.png      # Performance chart
├── VCB_future.png          # Prediction chart
└── VCB_future_predictions.csv
```

## Testing

```bash
# Backend tests
cd backend
pytest

# With coverage
pytest --cov=app --cov-report=html

# Legacy tests
cd ..
pytest tests/
```

## Database

### Tables

- `users` - User accounts with roles
- `stocks` - Stock master data
- `stock_prices` - Daily OHLCV data
- `stock_prediction_summaries` - Predicted % changes
- `training_configs` - Saved training configurations
- `experiment_runs` - Training run history
- `pipeline_dags` - Airflow DAG metadata

### Migrations

```bash
cd backend

# Generate migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Requirements

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- 8GB RAM, 10GB disk space

## Documentation

- [Backend API](backend/README.md)
- [Specifications](docs/SPECS.md)
- [Troubleshooting](_docs/TROUBLESHOOTING.md)

## License

MIT License
