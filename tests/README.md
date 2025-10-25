# Test Suite for Stock Prediction System

Comprehensive testing with unit tests (mocked) and integration tests (real API/model training).

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run fast tests only (skip integration)
pytest tests/ -v -m "not slow"

# Run with coverage
pytest tests/ -v --cov=modules --cov-report=term-missing

# Run specific test file
pytest tests/test_data_fetcher.py -v
```

## Test Structure

```
tests/
├── test_data_fetcher.py              # Unit tests (fast, mocked)
├── test_data_fetcher_integration.py  # Integration tests (real API)
├── test_model_trainer.py             # Model unit tests
└── test_model_trainer_integration.py # End-to-end training tests
```

## Test Coverage

| Module | Unit Tests | Integration Tests |
|--------|------------|-------------------|
| Data Fetcher | ✅ 14 tests | ✅ 7 tests |
| Model Trainer | ✅ 20 tests | ✅ 8 tests |
| **Total** | **34 tests** | **15 tests** |

## Test Markers

Markers defined in `pytest.ini`:
- `@pytest.mark.integration` - Real API/database calls
- `@pytest.mark.slow` - Long-running tests (model training)
- `@pytest.mark.unit` - Fast unit tests (default)

## Key Features Tested

**Data Fetching:**
- ✅ Incremental crawling (fetch only new data)
- ✅ API field mapping (`nmvolume` → `volume`)
- ✅ Error handling (timeouts, invalid data)
- ✅ Multi-stock fetching

**Model Training:**
- ✅ Technical indicator calculation (20+ indicators)
- ✅ Ensemble modeling (4 algorithms)
- ✅ Feature importance analysis
- ✅ Performance metrics (RMSE, MAE, R², direction accuracy)

**Integration:**
- ✅ End-to-end workflow (fetch → train → evaluate → predict)
- ✅ Real VNDirect API calls
- ✅ Actual model training with real data

## Special Tests

### Indicator Impact Analysis
Validates that technical indicators improve model performance:

```bash
# Compare baseline vs full model
pytest tests/test_model_trainer_integration.py::TestIndicatorImpact -v -s -m integration
```

**What it tests:**
- **Baseline**: OHLCV only (5 features)
- **Full Model**: OHLCV + 20 indicators (25+ features)
- **Metrics**: RMSE, MAE, R², direction accuracy
- **Duration**: ~3-5 minutes (trains real models)

**Key Findings:**
- Indicators improve RMSE by ~67%
- Indicators improve MAE by ~73%
- Most important: `MACD_signal`, `volume_MA`, `high`, `SMA_cross_20_60`

## Troubleshooting

**Import errors:** Ensure virtual environment is activated and you're in project root
**Test not found:** Use full path: `pytest tests/test_data_fetcher.py::TestClassName::test_name`
**Integration tests fail:** Check internet connection and VNDirect API accessibility

