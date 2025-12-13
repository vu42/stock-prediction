# VN30 Stock Price Prediction System

Machine learning system for predicting Vietnam's VN30 stock prices using ensemble models, FastAPI REST API, and Apache Airflow orchestration.

## Live Demo

The system is deployed and publicly accessible at:

| | URL |
|---|---|
| **Application** | http://13.215.172.15 |

**Demo Accounts:**
- Data Scientist: `ds1` / `pass1234`
- End User: `enduser1` / `pass1234`
- Admin: `admin` / `pass1234`

> **Note:** The server will be kept live until January 16, 2026

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

### Docker

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Start all services (API, Worker, PostgreSQL, Redis, MinIO, Airflow)
cd backend
docker-compose -f docker/docker-compose.dev.yml up -d

# 2. Wait for services to be healthy, then run database migrations
docker exec stock-prediction-api alembic upgrade head

# 3. Seed data (required for the app to function)
docker exec stock-prediction-api python -m scripts.seed_users
docker exec stock-prediction-api python -m scripts.seed_stocks
docker exec stock-prediction-api python -m scripts.seed_stock_prices  # Load historical data from CSV files
docker exec stock-prediction-api python -m scripts.seed_mock_predictions      # Mock predictions for demo
docker exec stock-prediction-api python -m scripts.seed_mock_prediction_points # Mock prediction points for charts

# 4. Configure MinIO for public access (required for evaluation plots)
# Install MinIO client first: brew install minio/stable/mc (macOS) or see https://min.io/docs/minio/linux/reference/minio-mc.html
mc alias set local http://localhost:9000 minioadmin minioadmin
mc anonymous set download local/stock-prediction-artifacts

# 5. Train ML Models
# Option A: Use Airflow UI
#   - Open http://localhost:8080 (login: airflow_api / AirflowApi@2025!)
#   - Navigate to DAGs → vn30_model_training → Trigger DAG
#   - Training takes about 5-15 minutes per stock depending on configuration
#
# Option B: Use App UI (requires frontend)
#   - Login with Data Scientist role → Training → Configure & Run
#   - Or: Pipelines → VN30 Model Training → Trigger Run

# 6. Access API documentation
open http://localhost:8000/docs
```

**Test User Credentials (seeded by scripts.seed_users):**
- Data Scientist: `ds1` / `pass1234`
- End User: `enduser1` / `pass1234`
- Admin: `admin` / `pass1234`

**Services Started:**

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Swagger UI |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Task queue |
| MinIO | http://localhost:9001 | Object storage (admin: minioadmin/minioadmin) |
| Airflow | http://localhost:8080 | DAG orchestration (airflow_api/AirflowApi@2025!) |

**Useful Commands:**

```bash
# View logs
docker logs stock-prediction-api -f
docker logs stock-prediction-airflow -f
docker logs stock-prediction-worker -f

# Restart services after code changes
docker restart stock-prediction-api stock-prediction-airflow

# Rebuild and restart (after docker-compose.yml changes)
docker-compose -f docker/docker-compose.dev.yml up -d --build

# Stop all services
docker-compose -f docker/docker-compose.dev.yml down

# Trigger training DAG manually
docker exec stock-prediction-airflow airflow dags trigger vn30_model_training

# Check DAG status
docker exec stock-prediction-airflow airflow dags list
docker exec stock-prediction-airflow airflow dags list-import-errors

# Reset database (warning: deletes all data)
docker exec stock-prediction-postgres psql -U postgres -d stock_prediction -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
docker exec stock-prediction-api alembic upgrade head
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
                              Predictions → MinIO
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
| `S3_ENDPOINT_URL` | MinIO for artifacts |

### Airflow DAGs

**DAGs:**

| DAG | Schedule | Description |
|-----|----------|-------------|
| `vn30_data_crawler` | 5:00 PM daily (VN time) | Fetches latest stock prices from VNDirect API |
| `vn30_model_training` | 6:00 PM daily (VN time) | Trains models, evaluates, generates predictions, uploads artifacts to MinIO |

**Model Training DAG Features:**
- Loads training configuration from database (models enabled/disabled, hyperparameters)
- Trains only configured stocks and models (respects config changes)
- Automatically creates MinIO bucket if not exists
- Uploads artifacts (model.pkl, scaler.pkl, evaluation.png, future.png, predictions.csv) to MinIO
- Stores artifact URLs in database for API access
- Sends email notification on completion (if configured)

## Model Output

### Local Files
Trained models saved locally in `output/{STOCK_SYMBOL}/`:

```
output/VCB/
├── VCB_model.pkl           # Trained ensemble
├── VCB_scaler.pkl          # Feature scaler
├── VCB_evaluation.png      # Performance chart
├── VCB_future.png          # Prediction chart
└── VCB_future_predictions.csv
```

### MinIO/S3 Artifacts
When training via Airflow DAG or Worker, artifacts are also uploaded to MinIO:

```
s3://stock-prediction-artifacts/artifacts/{run_id}/{STOCK}/
├── evaluation_png.png      # Model evaluation plot
├── future_png.png          # Future predictions plot
├── model_pkl.pkl           # Trained model
├── scaler_pkl.pkl          # Feature scaler
└── future_predictions_csv.csv
```

The artifact URLs are stored in the `experiment_ticker_artifacts` table and returned via the `/api/v1/models` endpoint as `plotUrl`.

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
- `stock_prediction_points` - Individual prediction data points
- `model_statuses` - Model training status and timestamps
- `model_horizon_metrics` - MAPE metrics per horizon (7d, 15d, 30d)
- `training_configs` - Saved training configurations
- `experiment_runs` - Training run history
- `experiment_logs` - Training run logs
- `experiment_ticker_artifacts` - Artifact URLs per stock (MinIO links)
- `pipeline_dags` - Airflow DAG metadata
- `user_saved_stocks` - User's saved stock watchlist

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
