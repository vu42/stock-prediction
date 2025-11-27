"""
Tests for model trainer service.
Migrated from original tests/test_model_trainer.py to use new structure.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from app.db.models import Stock, StockPrice
from app.services.model_trainer import (
    build_ensemble_model,
    calculate_technical_indicators,
    create_feature_matrix,
    get_stock_file_paths,
)


@pytest.fixture
def test_stock_with_prices(db: Session) -> Stock:
    """Create a test stock with price data."""
    stock = Stock(
        ticker="VCB",
        name="Vietcombank",
        is_active=True,
    )
    db.add(stock)
    db.commit()
    
    # Add price data (at least 300 rows for training)
    base_price = 100.0
    for i in range(300):
        price = StockPrice(
            stock_id=stock.id,
            price_date=pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
            open_price=base_price + np.random.randn() * 2,
            high_price=base_price + 2 + np.random.randn(),
            low_price=base_price - 2 + np.random.randn(),
            close_price=base_price + np.random.randn() * 2,
            volume=int(1000000 + np.random.randn() * 100000),
        )
        db.add(price)
        base_price += np.random.randn() * 0.5
    
    db.commit()
    db.refresh(stock)
    return stock


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV dataframe."""
    np.random.seed(42)
    n = 200
    
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n))
    
    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n),
        "high": close + np.abs(np.random.randn(n)) + 1,
        "low": close - np.abs(np.random.randn(n)) - 1,
        "close": close,
        "volume": np.random.randint(100000, 1000000, n),
    })


class TestGetStockFilePaths:
    """Test cases for get_stock_file_paths function."""

    def test_returns_correct_paths(self):
        """Test that correct file paths are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.model_trainer.settings") as mock_settings:
                mock_settings.base_output_dir = tmpdir
                
                paths = get_stock_file_paths("VCB")
                
                assert "dir" in paths
                assert "model" in paths
                assert "scaler" in paths
                assert "plot" in paths
                assert "VCB" in paths["model"]
                assert paths["model"].endswith(".pkl")

    def test_creates_directory(self):
        """Test that stock directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.model_trainer.settings") as mock_settings:
                mock_settings.base_output_dir = tmpdir
                
                paths = get_stock_file_paths("VCB")
                
                assert os.path.exists(paths["dir"])


class TestCalculateTechnicalIndicators:
    """Test cases for calculate_technical_indicators function."""

    def test_adds_sma_columns(self, sample_ohlcv_df):
        """Test that SMA columns are added."""
        result = calculate_technical_indicators(sample_ohlcv_df)
        
        assert "SMA_5" in result.columns
        assert "SMA_20" in result.columns
        assert "SMA_60" in result.columns

    def test_adds_ema_columns(self, sample_ohlcv_df):
        """Test that EMA columns are added."""
        result = calculate_technical_indicators(sample_ohlcv_df)
        
        assert "EMA_12" in result.columns
        assert "EMA_26" in result.columns

    def test_adds_macd(self, sample_ohlcv_df):
        """Test that MACD is calculated."""
        result = calculate_technical_indicators(sample_ohlcv_df)
        
        assert "MACD" in result.columns
        assert "MACD_signal" in result.columns

    def test_adds_rsi(self, sample_ohlcv_df):
        """Test that RSI is calculated."""
        result = calculate_technical_indicators(sample_ohlcv_df)
        
        assert "RSI" in result.columns
        # RSI should be between 0 and 100
        assert result["RSI"].min() >= 0
        assert result["RSI"].max() <= 100

    def test_adds_bollinger_bands(self, sample_ohlcv_df):
        """Test that Bollinger Bands are calculated."""
        result = calculate_technical_indicators(sample_ohlcv_df)
        
        assert "BB_middle" in result.columns
        assert "BB_upper" in result.columns
        assert "BB_lower" in result.columns
        assert "BB_position" in result.columns

    def test_drops_nan_rows(self, sample_ohlcv_df):
        """Test that NaN rows are dropped."""
        result = calculate_technical_indicators(sample_ohlcv_df)
        
        assert not result.isnull().any().any()

    def test_output_length(self, sample_ohlcv_df):
        """Test output length is reasonable (accounting for indicator warmup)."""
        result = calculate_technical_indicators(sample_ohlcv_df)
        
        # Should have fewer rows due to NaN dropping from rolling windows
        assert len(result) < len(sample_ohlcv_df)
        # But should still have reasonable amount (60 day rolling is longest)
        assert len(result) > len(sample_ohlcv_df) - 70


class TestCreateFeatureMatrix:
    """Test cases for create_feature_matrix function."""

    def test_output_shapes(self):
        """Test output shapes are correct."""
        data = np.random.randn(100, 5)
        seq_length = 10
        
        X, y = create_feature_matrix(data, seq_length)
        
        # Should have 100 - 10 = 90 samples
        assert X.shape[0] == 90
        assert y.shape[0] == 90
        # Features should be flattened: seq_length * num_features
        assert X.shape[1] == seq_length * 5

    def test_target_is_close_price(self):
        """Test that target is the close price (first column)."""
        data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ])
        
        X, y = create_feature_matrix(data, 2)
        
        # y should be first column of each target row
        assert y[0] == 7  # Row 2, first column
        assert y[1] == 10  # Row 3, first column


class TestBuildEnsembleModel:
    """Test cases for build_ensemble_model function."""

    def test_returns_dict_of_models(self):
        """Test that function returns dict of models."""
        models = build_ensemble_model()
        
        assert isinstance(models, dict)
        assert "random_forest" in models
        assert "gradient_boosting" in models
        assert "svr" in models
        assert "ridge" in models

    def test_models_are_sklearn_estimators(self):
        """Test that models have fit and predict methods."""
        models = build_ensemble_model()
        
        for name, model in models.items():
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    def test_models_can_fit_and_predict(self):
        """Test that models can fit and predict on sample data."""
        models = build_ensemble_model()
        
        # Create small sample data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_test = np.random.randn(10, 10)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            assert len(predictions) == 10

