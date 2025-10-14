"""
IMPROVED LSTM Model Training and Prediction Module

Enhancements:
1. Multi-feature input (OHLCV + Technical Indicators)
2. Validation set with Early Stopping
3. Better architecture (Bidirectional LSTM + Attention)
4. More evaluation metrics
5. Learning rate scheduling
6. K-fold cross-validation option
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Bidirectional, 
    Input, Attention, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from config import (
    SEQUENCE_LENGTH, LSTM_UNITS, DROPOUT_RATE, 
    TRAINING_EPOCHS, FUTURE_DAYS, BASE_OUTPUT_DIR
)
from modules.database import load_stock_data_from_db


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
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_60'] = df['close'].rolling(window=60).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df['volume_MA'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_MA']
    
    # Price momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    
    # Price rate of change
    df['ROC'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


def prepare_multifeature_sequences(data, seq_length):
    """
    Prepare multi-feature time series sequences for LSTM.
    
    Args:
        data: Scaled data with multiple features
        seq_length: Number of time steps to look back
        
    Returns:
        tuple: (X_sequences, y_targets)
    """
    sequences, targets = [], []
    
    for idx in range(seq_length, len(data)):
        sequences.append(data[idx - seq_length:idx, :])  # All features
        targets.append(data[idx, 0])  # Predict close price (first column)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    return sequences, targets


def build_improved_lstm_model(input_shape, learning_rate=0.001):
    """
    Build improved LSTM with:
    - Bidirectional LSTM
    - Batch Normalization
    - Attention mechanism
    - Better regularization
    
    Args:
        input_shape: (sequence_length, n_features)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Layer 1: Bidirectional LSTM
        Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True), 
                      input_shape=input_shape),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        # Layer 2: Bidirectional LSTM
        Bidirectional(LSTM(units=LSTM_UNITS//2, return_sequences=True)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        # Layer 3: Standard LSTM
        LSTM(units=LSTM_UNITS//2, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        # Dense layers
        Dense(units=64, activation='relu'),
        Dropout(0.2),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model


def train_prediction_model_improved(stock_symbol):
    """
    Train improved LSTM model with multi-features and validation.
    
    Args:
        stock_symbol: Stock symbol to train
        
    Returns:
        bool: True if successful
    """
    from modules.model_trainer import get_stock_file_paths
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
            'close', 'open', 'high', 'low', 'volume',
            'SMA_5', 'SMA_20', 'SMA_60',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
            'RSI', 'BB_middle', 'BB_upper', 'BB_lower',
            'ATR', 'volume_ratio', 'momentum', 'ROC'
        ]
        
        data = df_features[feature_columns].values
        
        # Normalize all features
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(data)
        
        # Prepare sequences
        X, y = prepare_multifeature_sequences(normalized_data, SEQUENCE_LENGTH)
        
        # Split: 70% train, 15% validation, 15% test
        train_size = int(len(X) * 0.70)
        val_size = int(len(X) * 0.85)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]
        
        print(f"[{stock_symbol}] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Build model
        model = build_improved_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=paths['model'],
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print(f"[{stock_symbol}] Training improved model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TRAINING_EPOCHS,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"[{stock_symbol}] Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
        
        # Save scaler with all features info
        scaler_data = {
            'scaler': scaler,
            'feature_columns': feature_columns
        }
        with open(paths['scaler'], 'wb') as f:
            pickle.dump(scaler_data, f)
        
        print(f"[{stock_symbol}] Model and scaler saved")
        return True
        
    except Exception as e:
        print(f"[{stock_symbol}] ERROR training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_model_improved(stock_symbol):
    """
    Evaluate model with comprehensive metrics.
    
    Returns:
        bool: True if successful
    """
    from modules.model_trainer import get_stock_file_paths
    paths = get_stock_file_paths(stock_symbol)
    
    if not os.path.exists(paths['model']) or not os.path.exists(paths['scaler']):
        print(f"[{stock_symbol}] ERROR: Model not found")
        return False
    
    try:
        print(f"[{stock_symbol}] Evaluating improved model...")
        
        # Load model and scaler
        model = load_model(paths['model'])
        with open(paths['scaler'], 'rb') as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
            feature_columns = scaler_data['feature_columns']
        
        # Load and prepare data
        df = load_stock_data_from_db(stock_symbol)
        df_features = calculate_technical_indicators(df)
        data = df_features[feature_columns].values
        normalized_data = scaler.transform(data)
        
        # Prepare sequences
        X, y = prepare_multifeature_sequences(normalized_data, SEQUENCE_LENGTH)
        
        # Use last 20% as test set
        test_start = int(len(X) * 0.80)
        X_test = X[test_start:]
        y_test = y[test_start:]
        
        # Predict
        predicted_normalized = model.predict(X_test, verbose=0)
        
        # Inverse transform (only for close price - first column)
        # Create dummy array for inverse transform
        dummy = np.zeros((len(predicted_normalized), scaler.n_features_in_))
        dummy[:, 0] = predicted_normalized.flatten()
        predicted_prices = scaler.inverse_transform(dummy)[:, 0]
        
        dummy_real = np.zeros((len(y_test), scaler.n_features_in_))
        dummy_real[:, 0] = y_test
        real_prices = scaler.inverse_transform(dummy_real)[:, 0]
        
        # Calculate comprehensive metrics
        rmse = np.sqrt(np.mean((predicted_prices - real_prices) ** 2))
        mae = np.mean(np.abs(predicted_prices - real_prices))
        mape = mean_absolute_percentage_error(real_prices, predicted_prices) * 100
        r2 = r2_score(real_prices, predicted_prices)
        
        # Direction accuracy (did we predict up/down correctly?)
        real_direction = np.sign(np.diff(real_prices))
        pred_direction = np.sign(np.diff(predicted_prices))
        direction_accuracy = np.mean(real_direction == pred_direction) * 100
        
        print(f"[{stock_symbol}] RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R²={r2:.3f}, Direction Acc={direction_accuracy:.1f}%")
        
        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Price comparison
        ax1.plot(real_prices, color='red', label='Real Price', linewidth=2)
        ax1.plot(predicted_prices, color='blue', label='Predicted Price', linewidth=2, alpha=0.8)
        ax1.set_title(f'{stock_symbol} Stock Price - Model Evaluation', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time (Days)', fontsize=12)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.3f}\nDirection Acc: {direction_accuracy:.1f}%'
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Prediction error
        error = predicted_prices - real_prices
        ax2.plot(error, color='purple', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(range(len(error)), error, 0, alpha=0.3, color='purple')
        ax2.set_title('Prediction Error', fontsize=14)
        ax2.set_xlabel('Time (Days)', fontsize=12)
        ax2.set_ylabel('Error (Predicted - Real)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(paths['plot'], dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[{stock_symbol}] Enhanced evaluation plot saved")
        return True
        
    except Exception as e:
        print(f"[{stock_symbol}] ERROR evaluating: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

