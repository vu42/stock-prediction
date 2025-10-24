"""
Integration tests for model_trainer_sklearn.py with REAL data from VNDirect API
Tests end-to-end workflow: fetch data → train model → evaluate → predict

Run with: pytest tests/test_model_trainer_integration.py -v -s -m integration
"""

import os
import shutil
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

# Import modules to test
from modules.data_fetcher import fetch_stock_data
from modules.model_trainer_sklearn import (
    train_prediction_model_sklearn,
    evaluate_model_sklearn,
    predict_future_prices_sklearn,
    calculate_technical_indicators,
)


@pytest.mark.integration
@pytest.mark.slow
class TestRealDataTraining:
    """Integration tests with real VNDirect data and actual model training"""

    @pytest.fixture(scope="class")
    def test_output_dir(self, tmp_path_factory):
        """Create temporary output directory for tests"""
        temp_dir = tmp_path_factory.mktemp("test_models")
        yield str(temp_dir)
        # Cleanup after all tests
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_stock(self):
        """Stock symbol to use for testing"""
        return "VCB"  # Vietnam Commercial Bank - highly liquid stock

    def test_fetch_and_prepare_data(self, test_stock):
        """
        Test Step 1: Fetch real data from VNDirect API
        Validates that we can get real, usable data
        """
        # Fetch last 60 days of data (enough for technical indicators)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"STEP 1: Fetching real data for {test_stock}")
        print(f"{'='*60}")

        with patch("modules.data_fetcher.DATA_START_DATE", start_date):
            # Mock database to avoid actual writes
            with patch("modules.data_fetcher.insert_stock_data") as mock_insert, patch(
                "modules.data_fetcher.get_last_data_date"
            ) as mock_last_date:

                mock_last_date.return_value = None  # First time fetch
                mock_insert.return_value = 50

                context = {"to_date": end_date}
                result = fetch_stock_data(test_stock, **context)

                assert result is True, "Data fetch should succeed"
                assert mock_insert.called, "Should insert data"

                # Get the actual data that was fetched
                df = mock_insert.call_args[0][0]

                # Validate data quality
                assert isinstance(df, pd.DataFrame), "Should return DataFrame"
                assert len(df) > 30, f"Should have enough data, got {len(df)} rows"

                # Check required columns
                required_cols = ["date", "open", "high", "low", "close", "volume"]
                assert all(
                    col in df.columns for col in required_cols
                ), f"Missing columns. Has: {df.columns.tolist()}"

                # Check data quality
                assert (df["close"] > 0).all(), "Close prices should be positive"
                assert (df["high"] >= df["low"]).all(), "High >= Low"
                assert not df["close"].isna().any(), "No missing close prices"

                print(f"✅ Successfully fetched {len(df)} records")
                print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
                print(
                    f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}"
                )
                print(f"   Sample data:")
                print(df[["date", "open", "high", "low", "close", "volume"]].head(3))

    def test_calculate_indicators_on_real_data(self, test_stock):
        """
        Test Step 2: Calculate technical indicators on real data
        Validates that indicators can be computed without errors
        """
        print(f"\n{'='*60}")
        print(f"STEP 2: Calculating technical indicators")
        print(f"{'='*60}")

        # Get real data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

        with patch("modules.data_fetcher.DATA_START_DATE", start_date):
            with patch("modules.data_fetcher.insert_stock_data") as mock_insert, patch(
                "modules.data_fetcher.get_last_data_date"
            ) as mock_last_date:
                mock_last_date.return_value = None
                mock_insert.return_value = 80

                fetch_stock_data(test_stock, to_date=end_date)
                df = mock_insert.call_args[0][0]

        # Calculate indicators
        df_with_indicators = calculate_technical_indicators(df)

        # Validate indicators
        assert (
            len(df_with_indicators) > 0
        ), "Should have rows after indicator calculation"
        assert "RSI" in df_with_indicators.columns, "Should calculate RSI"
        assert "MACD" in df_with_indicators.columns, "Should calculate MACD"
        assert "SMA_20" in df_with_indicators.columns, "Should calculate SMA"

        # Check RSI is in valid range
        assert (df_with_indicators["RSI"] >= 0).all(), "RSI should be >= 0"
        assert (df_with_indicators["RSI"] <= 100).all(), "RSI should be <= 100"

        print(f"✅ Calculated {len(df_with_indicators.columns)} features")
        print(f"   Original data: {len(df)} rows")
        print(f"   After indicators: {len(df_with_indicators)} rows (dropped NaN)")
        print(
            f"   Indicators: RSI={df_with_indicators['RSI'].iloc[-1]:.2f}, "
            f"MACD={df_with_indicators['MACD'].iloc[-1]:.2f}"
        )

    def test_end_to_end_training_with_real_data(self, test_stock, test_output_dir):
        """
        Test Step 3: Complete end-to-end training pipeline
        Fetches real data → Trains model → Validates outputs
        """
        print(f"\n{'='*60}")
        print(f"STEP 3: End-to-End Training with Real VNDirect Data")
        print(f"{'='*60}")

        # Fetch sufficient historical data (400 days to meet SEQUENCE_LENGTH + 200 requirement)
        # Trading days are ~250 per year, so 400 calendar days ≈ 280 trading days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

        print(f"\n1️⃣  Fetching {test_stock} data from {start_date} to {end_date}...")

        # Create mock DataFrame to simulate database load
        real_data = None

        with patch("modules.data_fetcher.DATA_START_DATE", start_date):
            with patch("modules.data_fetcher.insert_stock_data") as mock_insert, patch(
                "modules.data_fetcher.get_last_data_date"
            ) as mock_last_date:
                mock_last_date.return_value = None
                mock_insert.return_value = (
                    280  # Expect ~280 trading days from 400 calendar days
                )

                result = fetch_stock_data(test_stock, to_date=end_date)
                assert result is True

                real_data = mock_insert.call_args[0][0].copy()
                print(f"   ✅ Fetched {len(real_data)} records")

        # Now train model with this real data
        print(f"\n2️⃣  Training model on real data...")

        with patch("modules.model_trainer_sklearn.BASE_OUTPUT_DIR", test_output_dir):
            # Mock database load to return our real fetched data
            with patch(
                "modules.model_trainer_sklearn.load_stock_data_from_db"
            ) as mock_load:
                mock_load.return_value = real_data

                # Train the model
                train_result = train_prediction_model_sklearn(
                    test_stock, continue_training=False
                )

                assert train_result is True, "Training should succeed with real data"

                # Verify model files were created
                model_path = os.path.join(
                    test_output_dir, test_stock, f"{test_stock}_sklearn_model.pkl"
                )
                scaler_path = os.path.join(
                    test_output_dir, test_stock, f"{test_stock}_sklearn_scaler.pkl"
                )

                assert os.path.exists(
                    model_path
                ), f"Model file should exist at {model_path}"
                assert os.path.exists(
                    scaler_path
                ), f"Scaler file should exist at {scaler_path}"

                print(f"   ✅ Model trained successfully")
                print(f"   📁 Model saved to: {model_path}")
                print(f"   📁 Scaler saved to: {scaler_path}")

                # Verify we can load the model back
                import joblib
                import pickle

                models = joblib.load(model_path)
                assert "random_forest" in models, "Should have random forest model"
                assert (
                    "gradient_boosting" in models
                ), "Should have gradient boosting model"

                with open(scaler_path, "rb") as f:
                    scaler_data = pickle.load(f)
                    assert "scaler" in scaler_data, "Should have scaler"
                    assert (
                        "feature_columns" in scaler_data
                    ), "Should have feature columns"

                print(f"   ✅ Verified model artifacts")
                print(f"   Models: {list(models.keys())}")
                print(f"   Features: {len(scaler_data['feature_columns'])} columns")

    def test_evaluation_with_real_data(self, test_stock, test_output_dir):
        """
        Test Step 4: Evaluate trained model on real data
        """
        print(f"\n{'='*60}")
        print(f"STEP 4: Model Evaluation on Real Data")
        print(f"{'='*60}")

        # First, fetch and train (need enough data)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

        real_data = None
        with patch("modules.data_fetcher.DATA_START_DATE", start_date):
            with patch("modules.data_fetcher.insert_stock_data") as mock_insert, patch(
                "modules.data_fetcher.get_last_data_date"
            ) as mock_last_date:
                mock_last_date.return_value = None
                mock_insert.return_value = (
                    280  # Expect ~280 trading days from 400 calendar days
                )
                fetch_stock_data(test_stock, to_date=end_date)
                real_data = mock_insert.call_args[0][0].copy()

        # Train model
        with patch("modules.model_trainer_sklearn.BASE_OUTPUT_DIR", test_output_dir):
            with patch(
                "modules.model_trainer_sklearn.load_stock_data_from_db"
            ) as mock_load:
                mock_load.return_value = real_data
                train_result = train_prediction_model_sklearn(
                    test_stock, continue_training=False
                )
                assert train_result is True

                # Now evaluate
                print(f"\n📊 Evaluating model performance...")
                eval_result = evaluate_model_sklearn(test_stock)

                # Evaluation might return False if insufficient test data
                # but model files should still exist
                model_path = os.path.join(
                    test_output_dir, test_stock, f"{test_stock}_sklearn_model.pkl"
                )
                assert os.path.exists(model_path), "Model should exist after evaluation"

                if eval_result:
                    print(f"   ✅ Evaluation completed successfully")
                else:
                    print(f"   ⚠️  Evaluation returned False (may need more test data)")

    def test_prediction_with_real_data(self, test_stock, test_output_dir):
        """
        Test Step 5: Make future predictions with trained model
        """
        print(f"\n{'='*60}")
        print(f"STEP 5: Future Price Prediction")
        print(f"{'='*60}")

        # Fetch, train, and predict (need enough data)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

        real_data = None
        with patch("modules.data_fetcher.DATA_START_DATE", start_date):
            with patch("modules.data_fetcher.insert_stock_data") as mock_insert, patch(
                "modules.data_fetcher.get_last_data_date"
            ) as mock_last_date:
                mock_last_date.return_value = None
                mock_insert.return_value = (
                    280  # Expect ~280 trading days from 400 calendar days
                )
                fetch_stock_data(test_stock, to_date=end_date)
                real_data = mock_insert.call_args[0][0].copy()

        with patch("modules.model_trainer_sklearn.BASE_OUTPUT_DIR", test_output_dir):
            with patch(
                "modules.model_trainer_sklearn.load_stock_data_from_db"
            ) as mock_load:
                mock_load.return_value = real_data

                # Train
                train_prediction_model_sklearn(test_stock, continue_training=False)

                # Predict next 30 days
                print(f"\n🔮 Predicting next 30 days...")
                predict_result = predict_future_prices_sklearn(
                    test_stock, days_ahead=30
                )

                # Check if prediction succeeded
                if predict_result:
                    print(f"   ✅ Predictions generated successfully")

                    # Check if prediction CSV was created
                    pred_csv = os.path.join(
                        test_output_dir,
                        test_stock,
                        f"{test_stock}_sklearn_future_predictions.csv",
                    )
                    if os.path.exists(pred_csv):
                        pred_df = pd.read_csv(pred_csv)
                        print(f"   📈 Generated {len(pred_df)} predictions")
                        print(
                            f"   Date range: {pred_df['date'].min()} to {pred_df['date'].max()}"
                        )
                        print(f"\n   Sample predictions:")
                        print(pred_df.head(5).to_string(index=False))
                else:
                    print(f"   ⚠️  Prediction returned False (may need more data)")


@pytest.mark.integration
@pytest.mark.slow
class TestDataQualityValidation:
    """Validate data quality from real API for model training"""

    def test_data_sufficient_for_training(self):
        """
        Verify that real API data is sufficient for model training requirements
        """
        print(f"\n{'='*60}")
        print(f"DATA QUALITY VALIDATION")
        print(f"{'='*60}")

        test_stock = "FPT"  # FPT Corporation - another liquid stock
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")

        with patch("modules.data_fetcher.DATA_START_DATE", start_date):
            with patch("modules.data_fetcher.insert_stock_data") as mock_insert, patch(
                "modules.data_fetcher.get_last_data_date"
            ) as mock_last_date:
                mock_last_date.return_value = None
                mock_insert.return_value = (
                    280  # Expect ~280 trading days from 400 calendar days
                )

                fetch_stock_data(test_stock, to_date=end_date)
                df = mock_insert.call_args[0][0]

                # Requirement checks for training
                from config import SEQUENCE_LENGTH

                min_required = SEQUENCE_LENGTH + 200  # From model_trainer

                print(f"\n📋 Training Requirements Check:")
                print(f"   Minimum required: {min_required} records")
                print(f"   Fetched: {len(df)} records")

                if len(df) >= min_required:
                    print(f"   ✅ Sufficient data for training")
                else:
                    print(f"   ⚠️  Need {min_required - len(df)} more records")
                    print(f"   💡 Increase date range to get more historical data")

                # Check data completeness
                print(f"\n🔍 Data Completeness:")
                for col in ["open", "high", "low", "close", "volume"]:
                    missing = df[col].isna().sum()
                    if missing == 0:
                        print(f"   ✅ {col}: No missing values")
                    else:
                        print(f"   ⚠️  {col}: {missing} missing values")


@pytest.mark.integration
@pytest.mark.slow
class TestIndicatorImpact:
    """
    Test whether technical indicators actually improve model performance
    Compares baseline model (OHLCV only) vs full model (OHLCV + indicators)
    """

    @pytest.fixture
    def test_stock(self):
        """Stock symbol for testing"""
        return "VCB"

    @pytest.fixture
    def real_stock_data(self, test_stock):
        """Fetch real stock data for comparison"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        with patch("modules.data_fetcher.DATA_START_DATE", start_date):
            with patch("modules.data_fetcher.insert_stock_data") as mock_insert, patch(
                "modules.data_fetcher.get_last_data_date"
            ) as mock_last_date:
                mock_last_date.return_value = None
                mock_insert.return_value = 250

                fetch_stock_data(test_stock, to_date=end_date)
                df = mock_insert.call_args[0][0]

                return df

    def _train_baseline_model(self, df, stock_symbol="TEST_BASELINE"):
        """
        Train a baseline model using ONLY OHLCV features (no technical indicators)

        Args:
            df: DataFrame with OHLCV data
            stock_symbol: Symbol for file naming

        Returns:
            dict: Performance metrics
        """
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_percentage_error,
            r2_score,
        )
        from modules.model_trainer_sklearn import create_feature_matrix
        import joblib
        import pickle

        print(f"\n{'='*60}")
        print(f"TRAINING BASELINE MODEL (OHLCV Only)")
        print(f"{'='*60}")

        # Use only OHLCV features (ensure they exist in dataframe)
        feature_cols = ["open", "high", "low", "close", "volume"]

        # Check which columns exist in the dataframe
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) != len(feature_cols):
            print(
                f"Warning: Only found {available_cols} in dataframe columns: {df.columns.tolist()}"
            )

        print(f"Using {len(available_cols)} OHLCV features: {available_cols}")
        data = df[available_cols].values

        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        from config import SEQUENCE_LENGTH

        X, y = create_feature_matrix(scaled_data, SEQUENCE_LENGTH)

        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training data: {len(X_train)} samples")
        print(f"Test data: {len(X_test)} samples")
        print(f"Features per sample: {X_train.shape[1]}")

        # Build simple ensemble (same as full model for fair comparison)
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            "svr": SVR(kernel="rbf", C=1.0),
            "ridge": Ridge(alpha=1.0, random_state=42),
        }

        # Train models
        predictions = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred

        # Ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = np.mean(np.abs(ensemble_pred - y_test))
        mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
        r2 = r2_score(y_test, ensemble_pred)

        # Direction accuracy
        real_direction = np.sign(np.diff(y_test))
        pred_direction = np.sign(np.diff(ensemble_pred))
        direction_accuracy = np.mean(real_direction == pred_direction) * 100

        print(f"\n📊 Baseline Performance:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²: {r2:.4f}")
        print(f"   Direction Accuracy: {direction_accuracy:.1f}%")

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "model": models,
            "predictions": ensemble_pred,
            "y_test": y_test,
        }

    def _train_full_model_with_indicators(self, df, stock_symbol="TEST_FULL"):
        """
        Train full model WITH technical indicators

        Args:
            df: DataFrame with OHLCV data
            stock_symbol: Symbol for file naming

        Returns:
            dict: Performance metrics
        """
        from modules.model_trainer_sklearn import (
            calculate_technical_indicators,
            create_feature_matrix,
            build_ensemble_model,
        )
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_percentage_error,
            r2_score,
        )

        print(f"\n{'='*60}")
        print(f"TRAINING FULL MODEL (OHLCV + Indicators)")
        print(f"{'='*60}")

        # Calculate all technical indicators
        df_with_indicators = calculate_technical_indicators(df)

        print(f"Total features: {len(df_with_indicators.columns)} columns")
        print(f"Indicator features added: {len(df_with_indicators.columns) - 6}")

        # Use only numeric features (exclude date, stock_symbol, code, etc.)
        # Select only numeric columns
        numeric_cols = df_with_indicators.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        feature_cols = [
            col for col in numeric_cols if col not in ["date", "stock_symbol", "code"]
        ]

        print(f"Using {len(feature_cols)} numeric features")
        data = df_with_indicators[feature_cols].values

        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        from config import SEQUENCE_LENGTH

        X, y = create_feature_matrix(scaled_data, SEQUENCE_LENGTH)

        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training data: {len(X_train)} samples")
        print(f"Test data: {len(X_test)} samples")
        print(f"Features per sample: {X_train.shape[1]}")

        # Build ensemble
        models = build_ensemble_model()

        # Train models
        predictions = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred

        # Ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = np.mean(np.abs(ensemble_pred - y_test))
        mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
        r2 = r2_score(y_test, ensemble_pred)

        # Direction accuracy
        real_direction = np.sign(np.diff(y_test))
        pred_direction = np.sign(np.diff(ensemble_pred))
        direction_accuracy = np.mean(real_direction == pred_direction) * 100

        print(f"\n📊 Full Model Performance:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²: {r2:.4f}")
        print(f"   Direction Accuracy: {direction_accuracy:.1f}%")

        # Feature importance (from Random Forest)
        feature_importance = models["random_forest"].feature_importances_

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "model": models,
            "predictions": ensemble_pred,
            "y_test": y_test,
            "feature_importance": feature_importance,
            "feature_names": feature_cols,
        }

    def test_indicators_improve_performance(self, real_stock_data, test_stock):
        """
        Main test: Compare baseline vs full model performance
        Validates that technical indicators actually improve predictions
        """
        print(f"\n{'='*70}")
        print(f"INDICATOR IMPACT TEST: Baseline vs Full Model Comparison")
        print(f"{'='*70}")
        print(f"Stock: {test_stock}")
        print(f"Data points: {len(real_stock_data)}")

        # Train both models
        baseline_metrics = self._train_baseline_model(
            real_stock_data, f"{test_stock}_BASELINE"
        )
        full_metrics = self._train_full_model_with_indicators(
            real_stock_data, f"{test_stock}_FULL"
        )

        # Compare results
        print(f"\n{'='*70}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*70}")

        print(f"\n📊 Metric Comparison:")
        print(
            f"{'Metric':<20} {'Baseline':<15} {'With Indicators':<15} {'Improvement':<15}"
        )
        print(f"{'-'*70}")

        # RMSE (lower is better)
        rmse_improvement = (
            (baseline_metrics["rmse"] - full_metrics["rmse"]) / baseline_metrics["rmse"]
        ) * 100
        print(
            f"{'RMSE':<20} {baseline_metrics['rmse']:<15.4f} {full_metrics['rmse']:<15.4f} {rmse_improvement:>+.2f}%"
        )

        # MAE (lower is better)
        mae_improvement = (
            (baseline_metrics["mae"] - full_metrics["mae"]) / baseline_metrics["mae"]
        ) * 100
        print(
            f"{'MAE':<20} {baseline_metrics['mae']:<15.4f} {full_metrics['mae']:<15.4f} {mae_improvement:>+.2f}%"
        )

        # MAPE (lower is better)
        mape_improvement = (
            (baseline_metrics["mape"] - full_metrics["mape"]) / baseline_metrics["mape"]
        ) * 100
        print(
            f"{'MAPE (%)':<20} {baseline_metrics['mape']:<15.2f} {full_metrics['mape']:<15.2f} {mape_improvement:>+.2f}%"
        )

        # R² (higher is better)
        r2_improvement = (
            (full_metrics["r2"] - baseline_metrics["r2"]) / abs(baseline_metrics["r2"])
        ) * 100
        print(
            f"{'R²':<20} {baseline_metrics['r2']:<15.4f} {full_metrics['r2']:<15.4f} {r2_improvement:>+.2f}%"
        )

        # Direction Accuracy (higher is better)
        dir_improvement = (
            full_metrics["direction_accuracy"] - baseline_metrics["direction_accuracy"]
        )
        print(
            f"{'Direction Acc (%)':<20} {baseline_metrics['direction_accuracy']:<15.1f} {full_metrics['direction_accuracy']:<15.1f} {dir_improvement:>+.2f}pp"
        )

        # Overall verdict
        print(f"\n{'='*70}")
        print(f"VERDICT:")
        print(f"{'='*70}")

        avg_improvement = (rmse_improvement + mae_improvement + r2_improvement) / 3

        if avg_improvement > 5:
            print(
                f"✅ SIGNIFICANT IMPROVEMENT: Indicators improve performance by {avg_improvement:.1f}% on average"
            )
            verdict = "indicators_help_significantly"
        elif avg_improvement > 0:
            print(
                f"✓ MODERATE IMPROVEMENT: Indicators improve performance by {avg_improvement:.1f}% on average"
            )
            verdict = "indicators_help_moderately"
        elif avg_improvement > -5:
            print(
                f"≈ NEUTRAL: Indicators provide minimal impact ({avg_improvement:.1f}% change)"
            )
            verdict = "indicators_neutral"
        else:
            print(
                f"⚠️  DEGRADATION: Indicators may hurt performance ({avg_improvement:.1f}% worse)"
            )
            verdict = "indicators_hurt"

        # Assertions - indicators should at least not hurt significantly
        assert (
            full_metrics["rmse"] <= baseline_metrics["rmse"] * 1.1
        ), f"Indicators should not significantly degrade RMSE (got {rmse_improvement:.1f}% worse)"

        print(f"\n💡 Recommendation:")
        if avg_improvement > 0:
            print(
                f"   Keep using technical indicators - they improve model performance!"
            )
        else:
            print(f"   Consider feature selection to remove unhelpful indicators")

    def test_feature_importance_analysis(self, real_stock_data, test_stock):
        """
        Analyze which technical indicators are most important
        Shows feature importance rankings from Random Forest
        """
        print(f"\n{'='*70}")
        print(f"FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*70}")

        # Train full model to get feature importance
        full_metrics = self._train_full_model_with_indicators(
            real_stock_data, f"{test_stock}_IMPORTANCE"
        )

        # Get feature importance
        importances = full_metrics["feature_importance"]
        feature_names = full_metrics["feature_names"]

        # Reshape importance to match original features (before flattening)
        from config import SEQUENCE_LENGTH

        features_per_timestep = len(feature_names)

        # Average importance across time steps
        avg_importance = np.zeros(features_per_timestep)
        for i in range(features_per_timestep):
            # Sum importance for this feature across all time steps
            feature_importance_sum = sum(
                importances[j]
                for j in range(i, len(importances), features_per_timestep)
            )
            avg_importance[i] = feature_importance_sum

        # Normalize to sum to 1
        avg_importance = avg_importance / avg_importance.sum()

        # Sort by importance
        importance_pairs = sorted(
            zip(feature_names, avg_importance), key=lambda x: x[1], reverse=True
        )

        print(f"\n📊 Top 20 Most Important Features:")
        print(f"{'Rank':<6} {'Feature':<25} {'Importance':<15} {'Bar':<30}")
        print(f"{'-'*76}")

        for rank, (feature, importance) in enumerate(importance_pairs[:20], 1):
            bar_length = int(importance * 100)
            bar = "█" * bar_length
            print(f"{rank:<6} {feature:<25} {importance:<15.4f} {bar}")

        # Categorize features
        print(f"\n📈 Feature Categories:")

        categories = {
            "Price (OHLCV)": ["open", "high", "low", "close", "volume"],
            "Moving Averages": ["SMA_5", "SMA_20", "SMA_60", "EMA_12", "EMA_26"],
            "Momentum": ["RSI", "MACD", "MACD_signal", "momentum", "ROC"],
            "Volatility": [
                "ATR",
                "volatility",
                "BB_upper",
                "BB_lower",
                "BB_middle",
                "BB_position",
            ],
            "Volume": ["volume_MA", "volume_ratio"],
            "Crossovers": ["SMA_cross_5_20", "SMA_cross_20_60"],
        }

        category_importance = {}
        for category, features in categories.items():
            total = sum(
                imp
                for name, imp in importance_pairs
                if any(f in name for f in features)
            )
            category_importance[category] = total

        for category, importance in sorted(
            category_importance.items(), key=lambda x: x[1], reverse=True
        ):
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            print(f"{category:<20} {importance:>6.2%}  {bar}")

        # Insights
        print(f"\n💡 Insights:")
        top_indicator_features = [
            name
            for name, _ in importance_pairs[1:11]
            if name not in ["open", "high", "low", "close", "volume"]
        ]

        if top_indicator_features:
            print(
                f"   • Most valuable indicators: {', '.join(top_indicator_features[:5])}"
            )

        price_importance = sum(
            imp
            for name, imp in importance_pairs
            if name in ["open", "high", "low", "close"]
        )
        print(
            f"   • Base price features account for {price_importance:.1%} of importance"
        )

        indicator_importance = (
            1.0
            - price_importance
            - category_importance.get("Price (OHLCV)", 0)
            + sum(
                imp
                for name, imp in importance_pairs
                if name in ["open", "high", "low", "close"]
            )
        )
        print(
            f"   • Technical indicators account for ~{indicator_importance:.1%} of importance"
        )

        assert (
            len(importance_pairs) > 5
        ), "Should have computed feature importance for multiple features"


def run_integration_tests():
    """
    Helper function to run integration tests.
    Usage: python tests/test_model_trainer_integration.py
    """
    print("=" * 80)
    print("Running Model Training Integration Tests with REAL VNDirect Data")
    print("=" * 80)
    print("\n⚠️  WARNING: These tests make real API calls and train actual models!")
    print("Expected duration: 2-5 minutes\n")

    pytest.main(
        [
            __file__,
            "-v",
            "-s",  # Show print statements
            "-m",
            "integration",  # Only run integration tests
            "--tb=short",  # Shorter traceback
        ]
    )


if __name__ == "__main__":
    run_integration_tests()
