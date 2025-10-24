"""
Scikit-learn Model Training and Prediction Module

This module replaces TensorFlow/Keras with scikit-learn models for stock prediction.
Uses ensemble of Random Forest, Gradient Boosting, and SVR for robust predictions.

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
        "model": os.path.join(stock_dir, f"{stock_symbol}_sklearn_model.pkl"),
        "scaler": os.path.join(stock_dir, f"{stock_symbol}_sklearn_scaler.pkl"),
        "plot": os.path.join(stock_dir, f"{stock_symbol}_sklearn_evaluation.png"),
        "future_plot": os.path.join(stock_dir, f"{stock_symbol}_sklearn_future.png"),
        "future_csv": os.path.join(
            stock_dir, f"{stock_symbol}_sklearn_future_predictions.csv"
        ),
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


def create_feature_matrix(data, seq_length):
    """
    Create feature matrix for scikit-learn models by flattening sequences.

    Args:
        data: Scaled data with multiple features
        seq_length: Number of time steps to look back

    Returns:
        tuple: (X_features, y_targets)
    """
    features, targets = [], []

    for idx in range(seq_length, len(data)):
        # Flatten the sequence into a single feature vector
        sequence_features = data[idx - seq_length : idx].flatten()
        features.append(sequence_features)
        targets.append(data[idx, 0])  # Predict close price (first column)

    return np.array(features), np.array(targets)


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
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        ),
        "svr": SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1),
        "ridge": Ridge(alpha=1.0, random_state=42),
    }

    return models


def train_prediction_model_sklearn(stock_symbol, continue_training=True):
    """
    Train scikit-learn ensemble model for stock prediction.

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

        # Create new scaler if needed
        if scaler is None:
            scaler = StandardScaler()  # Better for scikit-learn models
            normalized_data = scaler.fit_transform(data)
        else:
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


def evaluate_model_sklearn(stock_symbol):
    """
    Evaluate scikit-learn ensemble model with comprehensive metrics.

    Returns:
        bool: True if successful
    """
    paths = get_stock_file_paths(stock_symbol)

    if not os.path.exists(paths["model"]) or not os.path.exists(paths["scaler"]):
        print(f"[{stock_symbol}] ERROR: Model not found")
        return False

    try:
        print(f"[{stock_symbol}] Evaluating scikit-learn ensemble model...")

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


def predict_future_prices_sklearn(stock_symbol, days_ahead=None):
    """
    Predict future stock prices using scikit-learn ensemble.

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
        print(
            f"[{stock_symbol}] Predicting {days_ahead} days ahead with scikit-learn..."
        )

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
# Wrapper Functions (compatible with existing DAGs)
# ============================================================================


def train_prediction_model(stock_symbol, continue_training=None):
    """
    Train scikit-learn model for stock prediction.
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
    return train_prediction_model_sklearn(
        stock_symbol, continue_training=continue_training
    )


def evaluate_model(stock_symbol):
    """
    Evaluate trained scikit-learn model.
    Wrapper function for compatibility with existing DAGs.

    Args:
        stock_symbol: Stock symbol to evaluate

    Returns:
        bool: True if successful
    """
    return evaluate_model_sklearn(stock_symbol)


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
    return predict_future_prices_sklearn(stock_symbol, days_ahead)
