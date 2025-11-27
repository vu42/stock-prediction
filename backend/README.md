# Stock Prediction Backend

FastAPI-based backend for the VN30 Stock Price Prediction System.

## Architecture

```
backend/
├── src/
│   ├── app/                    # FastAPI Application
│   │   ├── api/v1/            # REST API endpoints
│   │   ├── core/              # Config, security, logging, errors
│   │   ├── db/                # SQLAlchemy models and session
│   │   ├── schemas/           # Pydantic request/response schemas
│   │   ├── services/          # Business logic
│   │   └── integrations/      # External services (Airflow, S3, RQ)
│   └── worker/                # RQ background worker
│       └── tasks/             # Background tasks
├── migrations/                # Alembic migrations
├── tests/                     # Test suite
│   ├── api/                   # API endpoint tests
│   ├── services/              # Service layer tests
│   └── integrations/          # Integration tests
├── docker/                    # Docker configuration
├── pyproject.toml             # Python dependencies
└── alembic.ini               # Alembic configuration
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+

### Development Setup

1. **Create virtual environment:**
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Setup environment:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start dependencies with Docker:**
   ```bash
   docker-compose -f docker/docker-compose.dev.yml up -d postgres redis minio
   ```

5. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

6. **Start the API server:**
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Start the worker (in another terminal):**
   ```bash
   python -m worker.main
   ```

### Using Docker

Start all services:

```bash
docker-compose -f docker/docker-compose.dev.yml up
```

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | Login and get tokens |
| `/api/v1/auth/me` | GET | Get current user |
| `/api/v1/auth/refresh` | POST | Refresh access token |
| `/api/v1/auth/register` | POST | Register new user |

### Stocks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/stocks/top-picks` | GET | Get Should Buy/Sell picks |
| `/api/v1/stocks/market-table` | GET | Get market table data |
| `/api/v1/stocks/{ticker}` | GET | Get stock details |
| `/api/v1/stocks` | GET | List all stocks |

### Predictions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/stocks/{ticker}/predictions` | GET | Get predicted % changes |
| `/api/v1/stocks/{ticker}/chart` | GET | Get chart data |
| `/api/v1/models/{ticker}/status` | GET | Get model status |

### Training (Data Scientist)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/features/config` | GET/POST | Manage training config |
| `/api/v1/features/validate` | POST | Validate config |
| `/api/v1/experiments/run` | POST | Start training run |
| `/api/v1/experiments/{runId}` | GET | Get run status |

### Pipelines (Data Scientist)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/pipeline/dags` | GET | List DAGs |
| `/api/v1/pipeline/dags/{dagId}/trigger` | POST | Trigger DAG run |
| `/api/v1/pipeline/dags/{dagId}/runs` | GET | List DAG runs |

## Testing

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app --cov-report=html
```

Run specific test file:

```bash
pytest tests/api/test_auth.py -v
```

## Database Migrations

Generate a new migration:

```bash
alembic revision --autogenerate -m "Description of changes"
```

Apply migrations:

```bash
alembic upgrade head
```

Rollback one version:

```bash
alembic downgrade -1
```

## Project Structure Details

### Services Layer (`app/services/`)

- `data_fetcher.py` - VNDirect API integration, incremental crawling
- `model_trainer.py` - Ensemble ML model training
- `predictions.py` - Prediction query logic
- `stock_service.py` - Stock CRUD operations
- `email_service.py` - SendGrid email notifications

### Database Models (`app/db/models/`)

- `users.py` - User authentication and roles
- `stocks.py` - Stock data and predictions
- `training.py` - Training configs and experiments
- `pipelines.py` - Airflow DAG metadata

### Integrations (`app/integrations/`)

- `airflow_client.py` - Airflow REST API wrapper
- `storage_s3.py` - S3/MinIO artifact storage
- `queue.py` - Redis Queue (RQ) helpers

## Environment Variables

See `env.example` for all configuration options. Key variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `JWT_SECRET_KEY` | Secret for JWT tokens |
| `AIRFLOW_BASE_URL` | Airflow API URL |
| `S3_ENDPOINT_URL` | S3/MinIO endpoint |
| `SENDGRID_API_KEY` | SendGrid API key |

