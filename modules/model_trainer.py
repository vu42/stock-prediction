"""
Model Training and Prediction Module

Ensemble machine learning models for stock prediction using Random Forest,
Gradient Boosting, SVR, and Ridge regression.

Features:
1. Multi-feature input (OHLCV + Technical Indicators)
2. Ensemble modeling (Random Forest + Gradient Boosting + SVR)
3. Feature engineering for time series
4. Cross-validation for model selection
5. Incremental training support
6. Comprehensive evaluation metrics
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Ridge

from config import (
    SEQUENCE_LENGTH,
    FUTURE_DAYS,
    BASE_OUTPUT_DIR,
    CONTINUE_TRAINING,
    PREDICTION_HORIZONS,
)
from modules.database import load_stock_data_from_db


def get_stock_file_paths(stock_symbol):
    """
    Get file paths for a stock's model, scaler, and outputs.

    Args:
        stock_symbol: Stock symbol

    Returns:
        dict: Dictionary with file paths
    """
    stock_dir = os.path.join(BASE_OUTPUT_DIR, stock_symbol)
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


def calculate_technical_indicators(df):
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

    # Additional features for scikit-learn
    # Price volatility
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


def create_feature_matrix(data, seq_length, horizon=1):
    """
    Create feature matrix for scikit-learn models by flattening sequences.

    Args:
        data: Scaled data with multiple features
        seq_length: Number of time steps to look back
        horizon: Number of days ahead to predict (default 1)

    Returns:
        tuple: (X_features, y_targets)
    """
    features, targets = [], []

    for idx in range(seq_length, len(data) - horizon + 1):
        # Flatten the sequence into a single feature vector
        sequence_features = data[idx - seq_length : idx].flatten()
        features.append(sequence_features)
        # Target is the closing price 'horizon' days ahead
        targets.append(data[idx + horizon - 1, 0])

    return np.array(features), np.array(targets)


def create_multi_horizon_feature_matrix(data, seq_length, horizons):
    """
    Create feature matrix for direct multi-horizon prediction.

    Each sample has multiple targets: one for each horizon.
    This enables training models that directly predict T+1, T+3, T+7, etc.
    without recursive error accumulation.

    Args:
        data: Scaled data with multiple features (n_samples, n_features)
        seq_length: Number of time steps to look back
        horizons: List of horizons to predict, e.g. [1, 3, 7]

    Returns:
        tuple: (X_features, y_targets_dict)
            - X_features: shape (n_samples, seq_length * n_features)
            - y_targets_dict: dict mapping horizon -> target array
    """
    max_horizon = max(horizons)
    features = []
    targets_dict = {h: [] for h in horizons}

    for idx in range(seq_length, len(data) - max_horizon + 1):
        # Flatten the sequence into a single feature vector
        sequence_features = data[idx - seq_length : idx].flatten()
        features.append(sequence_features)

        # Get target for each horizon
        for h in horizons:
            # Target is the closing price 'h' days ahead (column 0 = close)
            targets_dict[h].append(data[idx + h - 1, 0])

    X = np.array(features)
    y_dict = {h: np.array(targets_dict[h]) for h in horizons}

    return X, y_dict


def build_ensemble_model():
    """
    Build ensemble of scikit-learn models.

    Returns:
        dict: Dictionary of models
    """
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        # "gradient_boosting": GradientBoostingRegressor(
        #     n_estimators=100,
        #     learning_rate=0.1,
        #     max_depth=6,
        #     min_samples_split=5,
        #     min_samples_leaf=2,
        #     random_state=42,
        # ),
        "svr": SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1),
        "ridge": Ridge(alpha=1.0, random_state=42),
    }

    return models


def train_prediction_model_internal(stock_symbol, continue_training=True):
    """
    Train ensemble model for stock prediction.

    Args:
        stock_symbol: Stock symbol to train
        continue_training: If True, load existing model and continue training

    Returns:
        bool: True if successful
    """
    paths = get_stock_file_paths(stock_symbol)

    try:
        print(f"[{stock_symbol}] Loading data from DATABASE...")

        # Load data
        df = load_stock_data_from_db(stock_symbol)

        if df.empty:
            print(f"[{stock_symbol}] ERROR: No data in database")
            return False

        if len(df) < SEQUENCE_LENGTH + 200:
            print(f"[{stock_symbol}] WARNING: Insufficient data ({len(df)} records)")
            return False

        # Calculate technical indicators
        print(f"[{stock_symbol}] Calculating technical indicators...")
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

        # =====================================================================
        # IMPORTANT: Split data BEFORE scaling to prevent data leakage
        # The scaler should only be fit on training data, not validation/test
        # =====================================================================

        # Calculate split indices on raw data
        # Account for samples lost due to sequence length (horizon=1 for single-step)
        effective_samples = len(data) - SEQUENCE_LENGTH

        # Split ratios: 70% train, 15% validation, 15% test
        train_end_idx = SEQUENCE_LENGTH + int(effective_samples * 0.70)

        # Split raw data for scaler fitting
        data_train = data[:train_end_idx]

        print(
            f"[{stock_symbol}] Scaler will be fitted on first {train_end_idx} samples (training data only)"
        )

        # Check if we can reuse existing scaler
        scaler = None
        if continue_training and os.path.exists(paths["scaler"]):
            try:
                print(f"[{stock_symbol}] Loading existing scaler...")
                with open(paths["scaler"], "rb") as f:
                    scaler_data = pickle.load(f)
                    scaler = scaler_data["scaler"]
                    saved_features = scaler_data["feature_columns"]
                    if saved_features != feature_columns:
                        print(
                            f"[{stock_symbol}] WARNING: Feature columns changed, creating new scaler"
                        )
                        scaler = None
            except Exception as e:
                print(f"[{stock_symbol}] Could not load scaler: {e}, creating new one")
                scaler = None

        # Create new scaler if needed - FIT ONLY ON TRAINING DATA
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(data_train)  # Fit only on training data (no leakage)
            print(
                f"[{stock_symbol}] Scaler fitted on training data only (no data leakage)"
            )

        # Transform all data using the scaler fitted on training data
        normalized_data = scaler.transform(data)

        # Create feature matrix
        X, y = create_feature_matrix(normalized_data, SEQUENCE_LENGTH)

        # Split: 70% train, 15% validation, 15% test
        train_size = int(len(X) * 0.70)
        val_size = int(len(X) * 0.85)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]

        print(
            f"[{stock_symbol}] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Load existing models or build new ones
        models = None
        if continue_training and os.path.exists(paths["model"]):
            try:
                print(
                    f"[{stock_symbol}] Loading existing models for incremental training..."
                )
                with open(paths["model"], "rb") as f:
                    models = pickle.load(f)
                print(
                    f"[{stock_symbol}] Successfully loaded existing models, continuing training"
                )
            except Exception as e:
                print(f"[{stock_symbol}] Could not load existing models: {e}")
                print(f"[{stock_symbol}] Building new models from scratch")
                models = None

        # Build new models if loading failed or not continuing
        if models is None:
            print(f"[{stock_symbol}] Building new ensemble models...")
            models = build_ensemble_model()

        # Train each model
        print(f"[{stock_symbol}] Training ensemble models...")
        trained_models = {}

        for name, model in models.items():
            print(f"[{stock_symbol}] Training {name}...")

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate on validation set
            val_pred = model.predict(X_val)
            val_score = r2_score(y_val, val_pred)
            print(f"[{stock_symbol}] {name} validation R²: {val_score:.4f}")

            trained_models[name] = model

        # Evaluate ensemble on test set
        test_predictions = {}
        for name, model in trained_models.items():
            test_pred = model.predict(X_test)
            test_predictions[name] = test_pred

        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(list(test_predictions.values()), axis=0)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = np.mean(np.abs(ensemble_pred - y_test))
        mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
        r2 = r2_score(y_test, ensemble_pred)

        print(
            f"[{stock_symbol}] Ensemble Test - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}"
        )

        # Save models and scaler
        with open(paths["model"], "wb") as f:
            pickle.dump(trained_models, f)

        scaler_data = {"scaler": scaler, "feature_columns": feature_columns}
        with open(paths["scaler"], "wb") as f:
            pickle.dump(scaler_data, f)

        print(f"[{stock_symbol}] Models and scaler saved")
        return True

    except Exception as e:
        print(f"[{stock_symbol}] ERROR training: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def evaluate_model_internal(stock_symbol):
    """
    Evaluate ensemble model with comprehensive metrics.

    Returns:
        bool: True if successful
    """
    paths = get_stock_file_paths(stock_symbol)

    if not os.path.exists(paths["model"]) or not os.path.exists(paths["scaler"]):
        print(f"[{stock_symbol}] ERROR: Model not found")
        return False

    try:
        print(f"[{stock_symbol}] Evaluating ensemble model...")

        # Load models and scaler
        with open(paths["model"], "rb") as f:
            models = pickle.load(f)
        with open(paths["scaler"], "rb") as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data["scaler"]
            feature_columns = scaler_data["feature_columns"]

        # Load and prepare data
        df = load_stock_data_from_db(stock_symbol)
        df_features = calculate_technical_indicators(df)
        data = df_features[feature_columns].values
        normalized_data = scaler.transform(data)

        # Create feature matrix
        X, y = create_feature_matrix(normalized_data, SEQUENCE_LENGTH)

        # Use last 20% as test set
        test_start = int(len(X) * 0.80)
        X_test = X[test_start:]
        y_test = y[test_start:]

        # Predict with each model
        predictions = {}
        for name, model in models.items():
            pred = model.predict(X_test)
            predictions[name] = pred

        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()), axis=0)

        # Inverse transform predictions
        dummy = np.zeros((len(ensemble_pred), scaler.n_features_in_))
        dummy[:, 0] = ensemble_pred
        predicted_prices = scaler.inverse_transform(dummy)[:, 0]

        dummy_real = np.zeros((len(y_test), scaler.n_features_in_))
        dummy_real[:, 0] = y_test
        real_prices = scaler.inverse_transform(dummy_real)[:, 0]

        # Calculate comprehensive metrics
        rmse = np.sqrt(np.mean((predicted_prices - real_prices) ** 2))
        mae = np.mean(np.abs(predicted_prices - real_prices))
        mape = mean_absolute_percentage_error(real_prices, predicted_prices) * 100
        r2 = r2_score(real_prices, predicted_prices)

        # Direction accuracy
        real_direction = np.sign(np.diff(real_prices))
        pred_direction = np.sign(np.diff(predicted_prices))
        direction_accuracy = np.mean(real_direction == pred_direction) * 100

        print(
            f"[{stock_symbol}] RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R²={r2:.3f}, Direction Acc={direction_accuracy:.1f}%"
        )

        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Price comparison
        ax1.plot(real_prices, color="red", label="Real Price", linewidth=2)
        ax1.plot(
            predicted_prices,
            color="blue",
            label="Predicted Price",
            linewidth=2,
            alpha=0.8,
        )
        ax1.set_title(
            f"{stock_symbol} Stock Price - Scikit-learn Model Evaluation",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_xlabel("Time (Days)", fontsize=12)
        ax1.set_ylabel("Price", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        textstr = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.3f}\nDirection Acc: {direction_accuracy:.1f}%"
        ax1.text(
            0.02,
            0.98,
            textstr,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 2: Prediction error
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

        print(f"[{stock_symbol}] Enhanced evaluation plot saved")
        return True

    except Exception as e:
        print(f"[{stock_symbol}] ERROR evaluating: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def predict_future_prices_internal(stock_symbol, days_ahead=None):
    """
    Predict future stock prices using ensemble.

    Args:
        stock_symbol: Stock symbol to predict
        days_ahead: Number of days to predict

    Returns:
        bool: True if successful
    """
    if days_ahead is None:
        days_ahead = FUTURE_DAYS

    paths = get_stock_file_paths(stock_symbol)

    if not os.path.exists(paths["model"]) or not os.path.exists(paths["scaler"]):
        print(f"[{stock_symbol}] ERROR: Model or scaler not found")
        return False

    try:
        print(f"[{stock_symbol}] Predicting {days_ahead} days ahead...")

        # Load models and scaler
        with open(paths["model"], "rb") as f:
            models = pickle.load(f)
        with open(paths["scaler"], "rb") as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data["scaler"]
            feature_columns = scaler_data["feature_columns"]

        # Load and prepare data
        df = load_stock_data_from_db(stock_symbol)
        df_features = calculate_technical_indicators(df)
        data = df_features[feature_columns].values
        normalized_data = scaler.transform(data)

        # Use last SEQUENCE_LENGTH days to predict future
        last_sequence = normalized_data[-SEQUENCE_LENGTH:]

        # Predict iteratively
        future_predictions = []
        current_sequence = last_sequence.copy()

        for i in range(days_ahead):
            # Flatten sequence for prediction
            input_features = current_sequence.flatten().reshape(1, -1)

            # Get predictions from all models
            model_predictions = []
            for name, model in models.items():
                pred = model.predict(input_features)[0]
                model_predictions.append(pred)

            # Ensemble prediction (average)
            next_pred_normalized = np.mean(model_predictions)
            future_predictions.append(next_pred_normalized)

            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred_normalized  # Update close price

            # Shift sequence
            current_sequence = np.vstack([current_sequence[1:], new_row])

        # Inverse transform predictions
        dummy = np.zeros((len(future_predictions), scaler.n_features_in_))
        dummy[:, 0] = future_predictions
        predicted_prices = scaler.inverse_transform(dummy)[:, 0]

        # Create prediction dataframe
        last_date = pd.to_datetime(df["date"].iloc[-1])
        future_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]

        predictions_df = pd.DataFrame(
            {"date": future_dates, "predicted_price": predicted_prices}
        )

        # Save predictions
        predictions_df.to_csv(paths["future_csv"], index=False)
        print(f"[{stock_symbol}] Predictions saved to {paths['future_csv']}")

        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot historical prices (last 90 days)
        historical_data = df.tail(90)
        ax.plot(
            range(len(historical_data)),
            historical_data["close"].values,
            color="blue",
            label="Historical Price",
            linewidth=2,
        )

        # Plot predicted prices
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

        # Add vertical line at prediction start
        ax.axvline(
            x=len(historical_data) - 1,
            color="green",
            linestyle=":",
            linewidth=2,
            label="Prediction Start",
        )

        ax.set_title(
            f"{stock_symbol} Stock Price - {days_ahead}-Day Future Prediction (Scikit-learn)",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add prediction range text
        pred_min = predicted_prices.min()
        pred_max = predicted_prices.max()
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

        print(f"[{stock_symbol}] Future prediction plot saved")
        print(
            f"[{stock_symbol}] Predicted price range: {pred_min:.2f} - {pred_max:.2f}"
        )
        return True

    except Exception as e:
        print(f"[{stock_symbol}] ERROR predicting: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# Multi-Horizon Direct Prediction Functions
# ============================================================================


def get_multi_horizon_file_paths(stock_symbol):
    """
    Get file paths for multi-horizon model outputs.

    Args:
        stock_symbol: Stock symbol

    Returns:
        dict: Dictionary with file paths for multi-horizon models
    """
    stock_dir = os.path.join(BASE_OUTPUT_DIR, stock_symbol)
    os.makedirs(stock_dir, exist_ok=True)

    return {
        "dir": stock_dir,
        "model": os.path.join(stock_dir, f"{stock_symbol}_multi_horizon_model.pkl"),
        "scaler": os.path.join(
            stock_dir, f"{stock_symbol}_scaler.pkl"
        ),  # Shared scaler
        "plot": os.path.join(stock_dir, f"{stock_symbol}_multi_horizon_evaluation.png"),
        "future_plot": os.path.join(
            stock_dir, f"{stock_symbol}_multi_horizon_future.png"
        ),
        "future_csv": os.path.join(
            stock_dir, f"{stock_symbol}_multi_horizon_predictions.csv"
        ),
    }


def train_multi_horizon_model_internal(
    stock_symbol, horizons=None, continue_training=True
):
    """
    Train direct multi-horizon ensemble models for stock prediction.

    Each horizon (e.g., 1d, 3d, 7d, 15d, 30d) gets its own dedicated ensemble,
    trained directly on that target. This avoids recursive error accumulation.

    Args:
        stock_symbol: Stock symbol to train
        horizons: List of horizons to predict, e.g. [1, 3, 7]
        continue_training: If True, load existing model and continue training

    Returns:
        bool: True if successful
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS

    paths = get_multi_horizon_file_paths(stock_symbol)
    standard_paths = get_stock_file_paths(stock_symbol)

    try:
        print(
            f"[{stock_symbol}] Loading data from DATABASE for multi-horizon training..."
        )
        print(f"[{stock_symbol}] Horizons: {horizons}")

        # Load data
        df = load_stock_data_from_db(stock_symbol)

        if df.empty:
            print(f"[{stock_symbol}] ERROR: No data in database")
            return False

        max_horizon = max(horizons)
        min_required = SEQUENCE_LENGTH + max_horizon + 200
        if len(df) < min_required:
            print(
                f"[{stock_symbol}] WARNING: Insufficient data ({len(df)} records, need {min_required})"
            )
            return False

        # Calculate technical indicators
        print(f"[{stock_symbol}] Calculating technical indicators...")
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

        # =====================================================================
        # IMPORTANT: Split data BEFORE scaling to prevent data leakage
        # The scaler should only be fit on training data, not validation/test
        # =====================================================================

        # Calculate split indices on raw data
        # Account for samples lost due to sequence length and max horizon
        max_horizon = max(horizons)
        effective_samples = len(data) - SEQUENCE_LENGTH - max_horizon + 1

        # Split ratios: 70% train, 15% validation, 15% test
        train_end_idx = SEQUENCE_LENGTH + int(effective_samples * 0.70)
        val_end_idx = SEQUENCE_LENGTH + int(effective_samples * 0.85)

        # Split raw data (before scaling)
        data_train = data[:train_end_idx]
        data_val = data[:val_end_idx]  # Include train for context
        data_all = data  # Full data for final transform

        print(
            f"[{stock_symbol}] Data split - Train end: {train_end_idx}, Val end: {val_end_idx}, Total: {len(data)}"
        )

        # Check if we can reuse existing scaler (shared with single-horizon)
        scaler = None
        if continue_training and os.path.exists(standard_paths["scaler"]):
            try:
                print(f"[{stock_symbol}] Loading existing scaler...")
                with open(standard_paths["scaler"], "rb") as f:
                    scaler_data = pickle.load(f)
                    scaler = scaler_data["scaler"]
                    saved_features = scaler_data["feature_columns"]
                    if saved_features != feature_columns:
                        print(
                            f"[{stock_symbol}] WARNING: Feature columns changed, creating new scaler"
                        )
                        scaler = None
            except Exception as e:
                print(f"[{stock_symbol}] Could not load scaler: {e}, creating new one")
                scaler = None

        # Create new scaler if needed - FIT ONLY ON TRAINING DATA
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(data_train)  # Fit only on training data (no leakage)
            print(
                f"[{stock_symbol}] Scaler fitted on training data only (no data leakage)"
            )

        # Transform all data using the scaler fitted on training data
        normalized_data = scaler.transform(data_all)

        # Create multi-horizon feature matrix from normalized data
        X, y_dict = create_multi_horizon_feature_matrix(
            normalized_data, SEQUENCE_LENGTH, horizons
        )

        # Split: 70% train, 15% validation, 15% test
        train_size = int(len(X) * 0.70)
        val_size = int(len(X) * 0.85)

        X_train, X_val, X_test = X[:train_size], X[train_size:val_size], X[val_size:]

        print(
            f"[{stock_symbol}] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Load existing models or build new ones
        horizon_models = None
        if continue_training and os.path.exists(paths["model"]):
            try:
                print(f"[{stock_symbol}] Loading existing multi-horizon models...")
                with open(paths["model"], "rb") as f:
                    horizon_models = pickle.load(f)
                print(f"[{stock_symbol}] Successfully loaded existing models")
            except Exception as e:
                print(f"[{stock_symbol}] Could not load existing models: {e}")
                horizon_models = None

        # Build new models if loading failed or not continuing
        if horizon_models is None:
            horizon_models = {}

        # Train ensemble for each horizon
        for horizon in horizons:
            print(f"\n[{stock_symbol}] === Training for {horizon}-day horizon ===")

            y_train = y_dict[horizon][:train_size]
            y_val = y_dict[horizon][train_size:val_size]
            y_test = y_dict[horizon][val_size:]

            # Build or reuse ensemble for this horizon
            if horizon not in horizon_models:
                print(
                    f"[{stock_symbol}] Building new ensemble for {horizon}-day horizon..."
                )
                horizon_models[horizon] = build_ensemble_model()

            trained_models = {}
            for name, model in horizon_models[horizon].items():
                print(f"[{stock_symbol}] Training {name} for {horizon}d...")

                # Train the model
                model.fit(X_train, y_train)

                # Evaluate on validation set
                val_pred = model.predict(X_val)
                val_score = r2_score(y_val, val_pred)
                print(
                    f"[{stock_symbol}] {name} ({horizon}d) validation R²: {val_score:.4f}"
                )

                trained_models[name] = model

            # Evaluate ensemble on test set
            test_predictions = []
            for name, model in trained_models.items():
                test_pred = model.predict(X_test)
                test_predictions.append(test_pred)

            ensemble_pred = np.mean(test_predictions, axis=0)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            mae = np.mean(np.abs(ensemble_pred - y_test))
            mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
            r2 = r2_score(y_test, ensemble_pred)

            print(
                f"[{stock_symbol}] {horizon}d Ensemble Test - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}"
            )

            horizon_models[horizon] = trained_models

        # Save models and scaler
        with open(paths["model"], "wb") as f:
            pickle.dump(horizon_models, f)

        scaler_data = {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "horizons": horizons,
        }
        with open(standard_paths["scaler"], "wb") as f:
            pickle.dump(scaler_data, f)

        print(f"\n[{stock_symbol}] Multi-horizon models saved to {paths['model']}")
        return True

    except Exception as e:
        print(f"[{stock_symbol}] ERROR training multi-horizon: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def predict_multi_horizon_internal(stock_symbol, horizons=None):
    """
    Predict future stock prices using direct multi-horizon models.

    Each horizon prediction is made directly from the current data,
    without recursive chaining, avoiding error accumulation.

    Args:
        stock_symbol: Stock symbol to predict
        horizons: List of horizons to predict (defaults to PREDICTION_HORIZONS)

    Returns:
        bool: True if successful
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS

    paths = get_multi_horizon_file_paths(stock_symbol)
    standard_paths = get_stock_file_paths(stock_symbol)

    if not os.path.exists(paths["model"]) or not os.path.exists(
        standard_paths["scaler"]
    ):
        print(f"[{stock_symbol}] ERROR: Multi-horizon model or scaler not found")
        return False

    try:
        print(f"[{stock_symbol}] Predicting with multi-horizon model...")
        print(f"[{stock_symbol}] Horizons: {horizons}")

        # Load models and scaler
        with open(paths["model"], "rb") as f:
            horizon_models = pickle.load(f)
        with open(standard_paths["scaler"], "rb") as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data["scaler"]
            feature_columns = scaler_data["feature_columns"]

        # Load and prepare data
        df = load_stock_data_from_db(stock_symbol)
        df_features = calculate_technical_indicators(df)
        data = df_features[feature_columns].values
        normalized_data = scaler.transform(data)

        # Use last SEQUENCE_LENGTH days as input
        last_sequence = normalized_data[-SEQUENCE_LENGTH:]
        input_features = last_sequence.flatten().reshape(1, -1)

        # Predict for each horizon
        predictions = {}
        last_date = pd.to_datetime(df["date"].iloc[-1])
        current_price = df["close"].iloc[-1]

        print(f"[{stock_symbol}] Current price: {current_price:.2f}")
        print(f"[{stock_symbol}] Last date: {last_date.strftime('%Y-%m-%d')}")

        for horizon in horizons:
            if horizon not in horizon_models:
                print(
                    f"[{stock_symbol}] WARNING: No model for {horizon}-day horizon, skipping"
                )
                continue

            # Get predictions from all models in the ensemble
            model_predictions = []
            for name, model in horizon_models[horizon].items():
                pred = model.predict(input_features)[0]
                model_predictions.append(pred)

            # Ensemble prediction (average)
            ensemble_pred_normalized = np.mean(model_predictions)

            # Inverse transform to get actual price
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, 0] = ensemble_pred_normalized
            predicted_price = scaler.inverse_transform(dummy)[0, 0]

            future_date = last_date + timedelta(days=horizon)
            price_change = predicted_price - current_price
            pct_change = (price_change / current_price) * 100

            predictions[horizon] = {
                "date": future_date,
                "predicted_price": predicted_price,
                "price_change": price_change,
                "pct_change": pct_change,
            }

            print(
                f"[{stock_symbol}] {horizon}d: {predicted_price:.2f} ({pct_change:+.2f}%)"
            )

        # Create prediction dataframe
        pred_rows = []
        for horizon in sorted(predictions.keys()):
            pred = predictions[horizon]
            pred_rows.append(
                {
                    "horizon_days": horizon,
                    "date": pred["date"],
                    "predicted_price": pred["predicted_price"],
                    "price_change": pred["price_change"],
                    "pct_change": pred["pct_change"],
                }
            )

        predictions_df = pd.DataFrame(pred_rows)
        predictions_df.to_csv(paths["future_csv"], index=False)
        print(f"[{stock_symbol}] Predictions saved to {paths['future_csv']}")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Historical + predicted prices
        historical_data = df.tail(90)
        ax1.plot(
            range(len(historical_data)),
            historical_data["close"].values,
            color="blue",
            label="Historical Price",
            linewidth=2,
        )

        # Plot predicted points
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(predictions)))
        for (horizon, pred), color in zip(sorted(predictions.items()), colors):
            x_pos = len(historical_data) - 1 + horizon
            ax1.scatter(
                x_pos,
                pred["predicted_price"],
                color=color,
                s=100,
                zorder=5,
                label=f"{horizon}d: {pred['predicted_price']:.2f}",
            )
            ax1.annotate(
                f"{horizon}d",
                (x_pos, pred["predicted_price"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        # Connect predictions with dashed line
        pred_x = [len(historical_data) - 1 + h for h in sorted(predictions.keys())]
        pred_y = [predictions[h]["predicted_price"] for h in sorted(predictions.keys())]
        ax1.plot(pred_x, pred_y, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

        # Add vertical line at prediction start
        ax1.axvline(
            x=len(historical_data) - 1,
            color="green",
            linestyle=":",
            linewidth=2,
            label="Today",
        )

        ax1.set_title(
            f"{stock_symbol} Multi-Horizon Direct Prediction",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_xlabel("Time", fontsize=12)
        ax1.set_ylabel("Price", fontsize=12)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Predicted % changes as bar chart
        horizons_sorted = sorted(predictions.keys())
        pct_changes = [predictions[h]["pct_change"] for h in horizons_sorted]
        bar_colors = ["green" if p >= 0 else "red" for p in pct_changes]

        bars = ax2.bar(
            [f"{h}d" for h in horizons_sorted],
            pct_changes,
            color=bar_colors,
            alpha=0.7,
            edgecolor="black",
        )

        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.set_title("Predicted Price Change (%)", fontsize=14)
        ax2.set_xlabel("Horizon", fontsize=12)
        ax2.set_ylabel("Change (%)", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, pct in zip(bars, pct_changes):
            height = bar.get_height()
            ax2.annotate(
                f"{pct:+.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(paths["future_plot"], dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[{stock_symbol}] Multi-horizon prediction plot saved")
        return True

    except Exception as e:
        print(f"[{stock_symbol}] ERROR predicting multi-horizon: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def evaluate_multi_horizon_model_internal(stock_symbol, horizons=None):
    """
    Evaluate multi-horizon model with comprehensive metrics per horizon.

    Args:
        stock_symbol: Stock symbol to evaluate
        horizons: List of horizons to evaluate

    Returns:
        bool: True if successful
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS

    paths = get_multi_horizon_file_paths(stock_symbol)
    standard_paths = get_stock_file_paths(stock_symbol)

    if not os.path.exists(paths["model"]) or not os.path.exists(
        standard_paths["scaler"]
    ):
        print(f"[{stock_symbol}] ERROR: Multi-horizon model not found")
        return False

    try:
        print(f"[{stock_symbol}] Evaluating multi-horizon model...")

        # Load models and scaler
        with open(paths["model"], "rb") as f:
            horizon_models = pickle.load(f)
        with open(standard_paths["scaler"], "rb") as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data["scaler"]
            feature_columns = scaler_data["feature_columns"]

        # Load and prepare data
        df = load_stock_data_from_db(stock_symbol)
        df_features = calculate_technical_indicators(df)
        data = df_features[feature_columns].values
        normalized_data = scaler.transform(data)

        # Create multi-horizon feature matrix
        X, y_dict = create_multi_horizon_feature_matrix(
            normalized_data, SEQUENCE_LENGTH, horizons
        )

        # Use last 20% as test set
        test_start = int(len(X) * 0.80)
        X_test = X[test_start:]

        # Evaluate each horizon
        metrics = {}
        fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 4 * len(horizons)))
        if len(horizons) == 1:
            axes = [axes]

        for idx, horizon in enumerate(horizons):
            if horizon not in horizon_models:
                print(f"[{stock_symbol}] WARNING: No model for {horizon}d, skipping")
                continue

            y_test = y_dict[horizon][test_start:]

            # Predict with ensemble
            predictions = []
            for name, model in horizon_models[horizon].items():
                pred = model.predict(X_test)
                predictions.append(pred)

            ensemble_pred = np.mean(predictions, axis=0)

            # Inverse transform
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

            metrics[horizon] = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "r2": r2,
                "direction_acc": direction_accuracy,
            }

            print(
                f"[{stock_symbol}] {horizon}d - RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R²={r2:.3f}, Dir Acc={direction_accuracy:.1f}%"
            )

            # Plot
            ax = axes[idx]
            ax.plot(real_prices, color="red", label="Real Price", linewidth=2)
            ax.plot(
                predicted_prices,
                color="blue",
                label="Predicted Price",
                linewidth=2,
                alpha=0.8,
            )
            ax.set_title(
                f"{stock_symbol} - {horizon}-Day Horizon Evaluation",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Time (Days)", fontsize=10)
            ax.set_ylabel("Price", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            textstr = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.3f}\nDir Acc: {direction_accuracy:.1f}%"
            ax.text(
                0.02,
                0.98,
                textstr,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        plt.savefig(paths["plot"], dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[{stock_symbol}] Multi-horizon evaluation plot saved")
        return True

    except Exception as e:
        print(f"[{stock_symbol}] ERROR evaluating multi-horizon: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# Wrapper Functions (compatible with existing DAGs)
# ============================================================================


def train_prediction_model(stock_symbol, continue_training=None):
    """
    Train model for stock prediction.
    Wrapper function for compatibility with existing DAGs.

    Args:
        stock_symbol: Stock symbol to train
        continue_training: If True, load existing model and continue training.
                          If False, train from scratch.
                          If None, use CONTINUE_TRAINING from config (default)

    Returns:
        bool: True if successful
    """
    if continue_training is None:
        continue_training = CONTINUE_TRAINING
    return train_prediction_model_internal(
        stock_symbol, continue_training=continue_training
    )


def evaluate_model(stock_symbol):
    """
    Evaluate trained model.
    Wrapper function for compatibility with existing DAGs.

    Args:
        stock_symbol: Stock symbol to evaluate

    Returns:
        bool: True if successful
    """
    return evaluate_model_internal(stock_symbol)


def predict_future_prices(stock_symbol, days_ahead=None):
    """
    Predict future stock prices using scikit-learn.
    Wrapper function for compatibility with existing DAGs.

    Args:
        stock_symbol: Stock symbol to predict
        days_ahead: Number of days to predict (defaults to FUTURE_DAYS from config)

    Returns:
        bool: True if successful
    """
    if days_ahead is None:
        days_ahead = FUTURE_DAYS
    return predict_future_prices_internal(stock_symbol, days_ahead)


# ============================================================================
# Multi-Horizon Wrapper Functions
# ============================================================================


def train_multi_horizon_model(stock_symbol, horizons=None, continue_training=None):
    """
    Train multi-horizon direct prediction models.
    Wrapper function for compatibility with existing DAGs.

    Args:
        stock_symbol: Stock symbol to train
        horizons: List of horizons [1, 3, 7] (defaults to PREDICTION_HORIZONS)
        continue_training: If True, load existing model and continue training

    Returns:
        bool: True if successful
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS
    if continue_training is None:
        continue_training = CONTINUE_TRAINING
    return train_multi_horizon_model_internal(
        stock_symbol, horizons=horizons, continue_training=continue_training
    )


def evaluate_multi_horizon_model(stock_symbol, horizons=None):
    """
    Evaluate multi-horizon model.
    Wrapper function for compatibility with existing DAGs.

    Args:
        stock_symbol: Stock symbol to evaluate
        horizons: List of horizons to evaluate

    Returns:
        bool: True if successful
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS
    return evaluate_multi_horizon_model_internal(stock_symbol, horizons=horizons)


def predict_multi_horizon(stock_symbol, horizons=None):
    """
    Predict future prices using multi-horizon direct models.
    Wrapper function for compatibility with existing DAGs.

    Args:
        stock_symbol: Stock symbol to predict
        horizons: List of horizons to predict

    Returns:
        bool: True if successful
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS
    return predict_multi_horizon_internal(stock_symbol, horizons=horizons)
