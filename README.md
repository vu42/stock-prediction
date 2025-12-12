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

### Docker

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Start all services (API, Worker, PostgreSQL, Redis, MinIO, Airflow)
cd backend
docker-compose -f docker/docker-compose.dev.yml up -d

# 2. Run database migrations
docker exec stock-prediction-api alembic upgrade head

# 3. Create a test user (optional)
docker exec stock-prediction-postgres psql -U postgres -d stock_prediction -c "
INSERT INTO users (id, username, password_hash, display_name, role, email, is_active, created_at, updated_at)
VALUES 
    (gen_random_uuid(), 'admin', '\$2b\$12\$LQv3c1yqBwlVkxO/iNqH.OaGgMpJmR3u8KJgHM.qD7mP9eqVTpMGi', 'Admin User', 'data_scientist', 'admin@example.com', true, NOW(), NOW()),
    (gen_random_uuid(), 'enduser1', '\$2b\$12\$ovIPTyXDwvQ10D5cA64ZBux.omXHLtMaBD4dJsAf6jEnLODRg8KPe', 'End User 1', 'end_user', 'enduser1@example.com', true, NOW(), NOW()),
    (gen_random_uuid(), 'enduser2', '\$2b\$12\$ovIPTyXDwvQ10D5cA64ZBux.omXHLtMaBD4dJsAf6jEnLODRg8KPe', 'End User 2', 'end_user', 'enduser2@example.com', true, NOW(), NOW());
"
# Test user credentials: admin / admin123

# 3.1. Saved Stocks
docker exec stock-prediction-api alembic revision --autogenerate -m "add_user_saved_stocks_table" 

docker exec stock-prediction-api alembic upgrade head

# 4. Seed VN30 stocks (optional)
docker restart stock-prediction-api
docker exec stock-prediction-api python -m scripts.seed_users
docker exec stock-prediction-api python -m scripts.seed_stocks
docker exec stock-prediction-api python -m scripts.seed_mock_prices
docker exec stock-prediction-api python -m scripts.seed_mock_predictions
docker exec stock-prediction-api python -m scripts.seed_mock_prediction_points

# 5. Access API documentation
open http://localhost:8000/docs

# 6. Configure MinIO for public access (required for evaluation plots)
mc alias set local http://localhost:9000 minioadmin minioadmin
mc anonymous set download local/stock-prediction-artifacts
```

**Services Started:**

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Swagger UI |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Task queue |
| MinIO | http://localhost:9001 | S3 storage (admin: minioadmin/minioadmin) |
| Airflow | http://localhost:8080 | DAG orchestration (admin/admin) |

**Useful Commands:**

```bash
# View logs
docker logs stock-prediction-api -f

# Restart API after code changes
docker restart stock-prediction-api

# Stop all services
docker-compose -f docker/docker-compose.dev.yml down

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
