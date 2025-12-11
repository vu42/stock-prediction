"""
Model Training and Prediction Service.

Ensemble machine learning models for stock prediction using Random Forest,
Gradient Boosting, SVR, and Ridge regression.

Migrated from modules/model_trainer.py to use SQLAlchemy.
"""

import os
import pickle
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sqlalchemy.orm import Session

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Stock, StockPredictionPoint, StockPredictionSummary
from app.schemas.training import ModelsConfig
from app.services.data_fetcher import load_stock_data_from_db

# Standard horizons for prediction summaries
PREDICTION_HORIZONS = [3, 7, 15, 30]

logger = get_logger(__name__)


def get_stock_file_paths(stock_symbol: str) -> dict[str, str]:
    """
    Get file paths for a stock's model, scaler, and outputs.

    Args:
        stock_symbol: Stock symbol

    Returns:
        Dictionary with file paths
    """
    stock_dir = os.path.join(settings.base_output_dir, stock_symbol)
    os.makedirs(stock_dir, exist_ok=True)

    return {
        "dir": stock_dir,
        "csv": os.path.join(stock_dir, f"{stock_symbol}_price.csv"),
        "model": os.path.join(stock_dir, f"{stock_symbol}_model.pkl"),
        "scaler": os.path.join(stock_dir, f"{stock_symbol}_scaler.pkl"),
        "plot": os.path.join(stock_dir, f"{stock_symbol}_evaluation.png"),
        "future_plot": os.path.join(stock_dir, f"{stock_symbol}_future.png"),
        "future_csv": os.path.join(stock_dir, f"{stock_symbol}_future_predictions.csv"),
    }


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with additional indicator columns
    """
    df = df.copy()

    # Simple Moving Averages
    df["SMA_5"] = df["close"].rolling(window=5).mean()
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["SMA_60"] = df["close"].rolling(window=60).mean()

    # Exponential Moving Average
    df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_middle"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
    df["BB_lower"] = df["BB_middle"] - (bb_std * 2)

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()

    # Volume indicators
    df["volume_MA"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_MA"]

    # Price momentum
    df["momentum"] = df["close"] - df["close"].shift(10)

    # Price rate of change
    df["ROC"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100

    # Additional features
    df["volatility"] = df["close"].rolling(window=20).std()

    # Price position within Bollinger Bands
    df["BB_position"] = (df["close"] - df["BB_lower"]) / (
        df["BB_upper"] - df["BB_lower"]
    )

    # Moving average crossovers
    df["SMA_cross_5_20"] = (df["SMA_5"] - df["SMA_20"]) / df["SMA_20"]
    df["SMA_cross_20_60"] = (df["SMA_20"] - df["SMA_60"]) / df["SMA_60"]

    # Drop NaN rows
    df = df.dropna()

    return df


def create_feature_matrix(
    data: np.ndarray, seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create feature matrix for scikit-learn models by flattening sequences.

    Args:
        data: Scaled data with multiple features
        seq_length: Number of time steps to look back

    Returns:
        Tuple of (X_features, y_targets)
    """
    features, targets = [], []

    for idx in range(seq_length, len(data)):
        sequence_features = data[idx - seq_length : idx].flatten()
        features.append(sequence_features)
        targets.append(data[idx, 0])  # Predict close price (first column)

    return np.array(features), np.array(targets)


def build_ensemble_model(models_config: ModelsConfig | None = None) -> dict:
    """
    Build ensemble of scikit-learn models.

    Args:
        models_config: Optional ModelsConfig from training_config table.
                       If provided, uses those values; otherwise falls back to settings.

    Returns:
        Dictionary of models (only enabled ones if config is provided)
    """
    models = {}

    # Random Forest
    rf_enabled = True
    rf_n_estimators = settings.random_forest_estimators
    rf_max_depth = 10
    if models_config and models_config.random_forest:
        rf_enabled = models_config.random_forest.enabled
        if models_config.random_forest.n_estimators is not None:
            rf_n_estimators = models_config.random_forest.n_estimators
        if models_config.random_forest.max_depth is not None:
            rf_max_depth = models_config.random_forest.max_depth

    if rf_enabled:
        models["random_forest"] = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    # Gradient Boosting
    gb_enabled = True
    gb_n_estimators = settings.gradient_boosting_estimators
    gb_learning_rate = 0.1
    gb_max_depth = 6
    if models_config and models_config.gradient_boosting:
        gb_enabled = models_config.gradient_boosting.enabled
        if models_config.gradient_boosting.n_estimators is not None:
            gb_n_estimators = models_config.gradient_boosting.n_estimators
        if models_config.gradient_boosting.learning_rate is not None:
            gb_learning_rate = models_config.gradient_boosting.learning_rate
        if models_config.gradient_boosting.max_depth is not None:
            gb_max_depth = models_config.gradient_boosting.max_depth

    if gb_enabled:
        models["gradient_boosting"] = GradientBoostingRegressor(
            n_estimators=gb_n_estimators,
            learning_rate=gb_learning_rate,
            max_depth=gb_max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )

    # SVR
    svr_enabled = True
    svr_c = settings.svr_c
    svr_gamma = "scale"
    svr_epsilon = 0.1
    if models_config and models_config.svr:
        svr_enabled = models_config.svr.enabled
        if models_config.svr.c is not None:
            svr_c = models_config.svr.c
        if models_config.svr.gamma is not None:
            svr_gamma = models_config.svr.gamma
        if models_config.svr.epsilon is not None:
            svr_epsilon = models_config.svr.epsilon

    if svr_enabled:
        models["svr"] = SVR(
            kernel="rbf",
            C=svr_c,
            gamma=svr_gamma,
            epsilon=svr_epsilon,
        )

    # Ridge
    ridge_enabled = True
    ridge_alpha = settings.ridge_alpha
    if models_config and models_config.ridge:
        ridge_enabled = models_config.ridge.enabled
        if models_config.ridge.alpha is not None:
            ridge_alpha = models_config.ridge.alpha

    if ridge_enabled:
        models["ridge"] = Ridge(alpha=ridge_alpha, random_state=42)

    return models


def train_prediction_model(
    db: Session,
    stock_symbol: str,
    continue_training: bool | None = None,
    models_config: ModelsConfig | None = None,
) -> bool:
    """
    Train ensemble model for stock prediction.

    Args:
        db: Database session
        stock_symbol: Stock symbol to train
        continue_training: If True, load existing model and continue training
        models_config: Optional ModelsConfig from training_config table

    Returns:
        True if successful
    """
    if continue_training is None:
        continue_training = settings.continue_training

    paths = get_stock_file_paths(stock_symbol)

    try:
        logger.info(f"[{stock_symbol}] Loading data from DATABASE...")

        # Load data
        df = load_stock_data_from_db(db, stock_symbol)

        if df.empty:
            logger.error(f"[{stock_symbol}] ERROR: No data in database")
            return False

        if len(df) < settings.sequence_length + 200:
            logger.warning(
                f"[{stock_symbol}] WARNING: Insufficient data ({len(df)} records)"
            )
            return False

        # Calculate technical indicators
        logger.info(f"[{stock_symbol}] Calculating technical indicators...")
        df_features = calculate_technical_indicators(df)

        # Select features for training
        feature_columns = [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "SMA_5",
            "SMA_20",
            "SMA_60",
            "EMA_12",
            "EMA_26",
            "MACD",
            "MACD_signal",
            "RSI",
            "BB_middle",
            "BB_upper",
            "BB_lower",
            "BB_position",
            "ATR",
            "volume_ratio",
            "momentum",
            "ROC",
            "volatility",
            "SMA_cross_5_20",
            "SMA_cross_20_60",
        ]

        data = df_features[feature_columns].values

        # Check if we can reuse existing scaler
        scaler = None
        if continue_training and os.path.exists(paths["scaler"]):
            try:
                logger.info(f"[{stock_symbol}] Loading existing scaler...")
                with open(paths["scaler"], "rb") as f:
                    scaler_data = pickle.load(f)
                    scaler = scaler_data["scaler"]
                    saved_features = scaler_data["feature_columns"]
                    if saved_features != feature_columns:
                        logger.warning(
                            f"[{stock_symbol}] WARNING: Feature columns changed, creating new scaler"
                        )
                        scaler = None
            except Exception as e:
                logger.warning(
                    f"[{stock_symbol}] Could not load scaler: {e}, creating new one"
                )
                scaler = None

        # Create new scaler if needed
        if scaler is None:
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data)
        else:
            normalized_data = scaler.transform(data)

        # Create feature matrix
        X, y = create_feature_matrix(normalized_data, settings.sequence_length)

        # Split: 70% train, 15% validation, 15% test
        train_size = int(len(X) * 0.70)
        val_size = int(len(X) * 0.85)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]

        logger.info(
            f"[{stock_symbol}] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Load existing models or build new ones
        models = None
        if continue_training and os.path.exists(paths["model"]):
            try:
                logger.info(
                    f"[{stock_symbol}] Loading existing models for incremental training..."
                )
                with open(paths["model"], "rb") as f:
                    models = pickle.load(f)
                logger.info(f"[{stock_symbol}] Successfully loaded existing models")
            except Exception as e:
                logger.warning(f"[{stock_symbol}] Could not load existing models: {e}")
                models = None

        # Build new models if loading failed
        if models is None:
            logger.info(f"[{stock_symbol}] Building new ensemble models...")
            models = build_ensemble_model(models_config)

        # Train each model
        logger.info(f"[{stock_symbol}] Training ensemble models...")
        trained_models = {}

        for name, model in models.items():
            logger.info(f"[{stock_symbol}] Training {name}...")
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            val_score = r2_score(y_val, val_pred)
            logger.info(f"[{stock_symbol}] {name} validation R²: {val_score:.4f}")

            trained_models[name] = model

        # Evaluate ensemble on test set
        test_predictions = {}
        for name, model in trained_models.items():
            test_predictions[name] = model.predict(X_test)

        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(list(test_predictions.values()), axis=0)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = np.mean(np.abs(ensemble_pred - y_test))
        mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
        r2 = r2_score(y_test, ensemble_pred)

        logger.info(
            f"[{stock_symbol}] Ensemble Test - RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
            f"MAPE: {mape:.2f}%, R²: {r2:.4f}"
        )

        # Save models and scaler
        with open(paths["model"], "wb") as f:
            pickle.dump(trained_models, f)

        scaler_data = {"scaler": scaler, "feature_columns": feature_columns}
        with open(paths["scaler"], "wb") as f:
            pickle.dump(scaler_data, f)

        logger.info(f"[{stock_symbol}] Models and scaler saved")
        return True

    except Exception as e:
        logger.error(f"[{stock_symbol}] ERROR training: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def evaluate_model(db: Session, stock_symbol: str) -> bool:
    """
    Evaluate ensemble model with comprehensive metrics.

    Args:
        db: Database session
        stock_symbol: Stock symbol to evaluate

    Returns:
        True if successful
    """
    paths = get_stock_file_paths(stock_symbol)

    if not os.path.exists(paths["model"]) or not os.path.exists(paths["scaler"]):
        logger.error(f"[{stock_symbol}] ERROR: Model not found")
        return False

    try:
        logger.info(f"[{stock_symbol}] Evaluating ensemble model...")

        # Load models and scaler
        with open(paths["model"], "rb") as f:
            models = pickle.load(f)
        with open(paths["scaler"], "rb") as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data["scaler"]
            feature_columns = scaler_data["feature_columns"]

        # Load and prepare data
        df = load_stock_data_from_db(db, stock_symbol)
        df_features = calculate_technical_indicators(df)
        data = df_features[feature_columns].values
        normalized_data = scaler.transform(data)

        # Create feature matrix
        X, y = create_feature_matrix(normalized_data, settings.sequence_length)

        # Use last 20% as test set
        test_start = int(len(X) * 0.80)
        X_test = X[test_start:]
        y_test = y[test_start:]

        # Predict with each model
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X_test)

        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()), axis=0)

        # Inverse transform predictions
        dummy = np.zeros((len(ensemble_pred), scaler.n_features_in_))
        dummy[:, 0] = ensemble_pred
        predicted_prices = scaler.inverse_transform(dummy)[:, 0]

        dummy_real = np.zeros((len(y_test), scaler.n_features_in_))
        dummy_real[:, 0] = y_test
        real_prices = scaler.inverse_transform(dummy_real)[:, 0]

        # Calculate metrics
        rmse = np.sqrt(np.mean((predicted_prices - real_prices) ** 2))
        mae = np.mean(np.abs(predicted_prices - real_prices))
        mape = mean_absolute_percentage_error(real_prices, predicted_prices) * 100
        r2 = r2_score(real_prices, predicted_prices)

        # Direction accuracy
        real_direction = np.sign(np.diff(real_prices))
        pred_direction = np.sign(np.diff(predicted_prices))
        direction_accuracy = np.mean(real_direction == pred_direction) * 100

        logger.info(
            f"[{stock_symbol}] RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, "
            f"R²={r2:.3f}, Direction Acc={direction_accuracy:.1f}%"
        )

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        ax1.plot(real_prices, color="red", label="Real Price", linewidth=2)
        ax1.plot(
            predicted_prices,
            color="blue",
            label="Predicted Price",
            linewidth=2,
            alpha=0.8,
        )
        ax1.set_title(
            f"{stock_symbol} Stock Price - Model Evaluation",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_xlabel("Time (Days)", fontsize=12)
        ax1.set_ylabel("Price", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        textstr = (
            f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\n"
            f"R²: {r2:.3f}\nDirection Acc: {direction_accuracy:.1f}%"
        )
        ax1.text(
            0.02,
            0.98,
            textstr,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Prediction error plot
        error = predicted_prices - real_prices
        ax2.plot(error, color="purple", linewidth=1)
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax2.fill_between(range(len(error)), error, 0, alpha=0.3, color="purple")
        ax2.set_title("Prediction Error", fontsize=14)
        ax2.set_xlabel("Time (Days)", fontsize=12)
        ax2.set_ylabel("Error (Predicted - Real)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(paths["plot"], dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"[{stock_symbol}] Evaluation plot saved")
        return True

    except Exception as e:
        logger.error(f"[{stock_symbol}] ERROR evaluating: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def predict_future_prices(
    db: Session,
    stock_symbol: str,
    days_ahead: int | None = None,
    run_id: str | None = None,
) -> bool:
    """
    Predict future stock prices using ensemble.

    Args:
        db: Database session
        stock_symbol: Stock symbol to predict
        days_ahead: Number of days to predict
        run_id: Optional experiment run ID to link predictions

    Returns:
        True if successful
    """
    if days_ahead is None:
        days_ahead = settings.future_days

    paths = get_stock_file_paths(stock_symbol)

    if not os.path.exists(paths["model"]) or not os.path.exists(paths["scaler"]):
        logger.error(f"[{stock_symbol}] ERROR: Model or scaler not found")
        return False

    try:
        logger.info(f"[{stock_symbol}] Predicting {days_ahead} days ahead...")

        # Load models and scaler
        with open(paths["model"], "rb") as f:
            models = pickle.load(f)
        with open(paths["scaler"], "rb") as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data["scaler"]
            feature_columns = scaler_data["feature_columns"]

        # Load and prepare data
        df = load_stock_data_from_db(db, stock_symbol)
        df_features = calculate_technical_indicators(df)
        data = df_features[feature_columns].values
        normalized_data = scaler.transform(data)

        # Use last SEQUENCE_LENGTH days to predict future
        last_sequence = normalized_data[-settings.sequence_length :]

        # Predict iteratively
        future_predictions = []
        current_sequence = last_sequence.copy()

        for i in range(days_ahead):
            input_features = current_sequence.flatten().reshape(1, -1)

            model_predictions = []
            for name, model in models.items():
                pred = model.predict(input_features)[0]
                model_predictions.append(pred)

            next_pred_normalized = np.mean(model_predictions)
            future_predictions.append(next_pred_normalized)

            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred_normalized
            current_sequence = np.vstack([current_sequence[1:], new_row])

        # Inverse transform predictions
        dummy = np.zeros((len(future_predictions), scaler.n_features_in_))
        dummy[:, 0] = future_predictions
        predicted_prices = scaler.inverse_transform(dummy)[:, 0]

        # Create prediction dataframe
        last_date = pd.to_datetime(df["date"].iloc[-1])
        future_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]

        predictions_df = pd.DataFrame(
            {
                "date": future_dates,
                "predicted_price": predicted_prices,
            }
        )

        # Save predictions
        predictions_df.to_csv(paths["future_csv"], index=False)
        logger.info(f"[{stock_symbol}] Predictions saved to {paths['future_csv']}")

        # Insert prediction summaries into database for API
        current_price = float(df["close"].iloc[-1])
        as_of_date = last_date.date()

        # Get stock from database
        from sqlalchemy import select

        stock_stmt = select(Stock).where(Stock.ticker == stock_symbol)
        stock = db.execute(stock_stmt).scalar_one_or_none()

        if stock:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            from uuid import UUID as PyUUID

            # Save prediction summaries (aggregated % change for key horizons)
            for horizon in PREDICTION_HORIZONS:
                if horizon <= days_ahead:
                    # Get predicted price for this horizon (0-indexed, so horizon-1)
                    pred_price = float(predicted_prices[horizon - 1])
                    pct_change = ((pred_price - current_price) / current_price) * 100

                    # Upsert prediction summary
                    stmt = (
                        pg_insert(StockPredictionSummary)
                        .values(
                            stock_id=stock.id,
                            as_of_date=as_of_date,
                            horizon_days=horizon,
                            predicted_change_pct=round(pct_change, 4),
                            experiment_run_id=PyUUID(run_id) if run_id else None,
                        )
                        .on_conflict_do_update(
                            constraint="uq_pred_summary",
                            set_={
                                "predicted_change_pct": round(pct_change, 4),
                                "experiment_run_id": PyUUID(run_id) if run_id else None,
                            },
                        )
                    )
                    db.execute(stmt)

            db.commit()
            logger.info(f"[{stock_symbol}] Prediction summaries saved to database")

            # Save prediction points (per-day forecast for chart overlay)
            # We save points for the full horizon (days_ahead days)
            for i, (pred_date, pred_price) in enumerate(
                zip(future_dates, predicted_prices)
            ):
                day_offset = i + 1  # 1-indexed day offset

                stmt = (
                    pg_insert(StockPredictionPoint)
                    .values(
                        stock_id=stock.id,
                        experiment_run_id=PyUUID(run_id) if run_id else None,
                        horizon_days=day_offset,
                        prediction_date=(
                            pred_date.date()
                            if hasattr(pred_date, "date")
                            else pred_date
                        ),
                        predicted_price=round(float(pred_price), 4),
                    )
                    .on_conflict_do_update(
                        constraint="uq_pred_point",
                        set_={
                            "predicted_price": round(float(pred_price), 4),
                        },
                    )
                )
                db.execute(stmt)

            db.commit()
            logger.info(
                f"[{stock_symbol}] {len(predicted_prices)} prediction points saved to database"
            )

        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 7))

        historical_data = df.tail(90)
        ax.plot(
            range(len(historical_data)),
            historical_data["close"].values,
            color="blue",
            label="Historical Price",
            linewidth=2,
        )

        future_x = range(len(historical_data), len(historical_data) + days_ahead)
        ax.plot(
            future_x,
            predicted_prices,
            color="red",
            label=f"{days_ahead}-Day Prediction",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
        )

        ax.axvline(
            x=len(historical_data) - 1,
            color="green",
            linestyle=":",
            linewidth=2,
            label="Prediction Start",
        )

        ax.set_title(
            f"{stock_symbol} Stock Price - {days_ahead}-Day Future Prediction",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        pred_min, pred_max = predicted_prices.min(), predicted_prices.max()
        pred_last = predicted_prices[-1]
        textstr = f"Predicted Range:\nMin: {pred_min:.2f}\nMax: {pred_max:.2f}\nFinal: {pred_last:.2f}"
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(paths["future_plot"], dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"[{stock_symbol}] Future prediction plot saved")
        logger.info(
            f"[{stock_symbol}] Predicted price range: {pred_min:.2f} - {pred_max:.2f}"
        )
        return True

    except Exception as e:
        logger.error(f"[{stock_symbol}] ERROR predicting: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
