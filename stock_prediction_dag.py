"""
Stock Price Prediction DAG using LSTM Neural Network
Author: TuPH
"""
import os
import ssl
import json
import base64
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

# Configuration
STOCK_SYMBOL = "DIG"
DATA_START_DATE = "2000-01-01"
OUTPUT_DIR = "/Users/aphan/Learning/stock_data"
CSV_FILE_PATH = os.path.join(OUTPUT_DIR, "stock_price.csv")
MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, "stockmodel.h5")
API_BASE_URL = "https://api-finfo.vndirect.com.vn/v4/stock_prices"
SEQUENCE_LENGTH = 60
LSTM_UNITS = 50
DROPOUT_RATE = 0.2

# SSL configuration
ssl._create_default_https_context = ssl._create_unverified_context


def fetch_stock_data(**context):
    """
    Fetch historical stock price data from VNDirect API.
    
    Args:
        context: Airflow context containing execution date
        
    Returns:
        bool: True if data fetch successful
    """
    end_date = context["to_date"]
    
    # Build API request URL
    query_params = f"sort=date&q=code:{STOCK_SYMBOL}~date:gte:{DATA_START_DATE}~date:lte:{end_date}&size=9990&page=1"
    api_url = f"{API_BASE_URL}?{query_params}"
    
    print(f"Fetching data from: {api_url}")
    
    # Configure HTTP request with headers
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    request = Request(api_url, headers=headers)
    
    # Fetch and parse API response
    response_data = urlopen(request, timeout=10).read()
    parsed_data = json.loads(response_data)['data']
    
    # Convert to DataFrame
    df = pd.DataFrame(parsed_data)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save to CSV
    df.to_csv(CSV_FILE_PATH, index=False)
    print(f"Successfully saved {len(df)} records to {CSV_FILE_PATH}")
    
    return True



def prepare_sequences(data, seq_length):
    """
    Prepare time series sequences for LSTM training.
    
    Args:
        data: Scaled price data
        seq_length: Number of time steps to look back
        
    Returns:
        tuple: (X_sequences, y_targets)
    """
    sequences, targets = [], []
    
    for idx in range(seq_length, len(data)):
        sequences.append(data[idx - seq_length:idx, 0])
        targets.append(data[idx, 0])
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Reshape for LSTM input: (samples, time_steps, features)
    sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    
    return sequences, targets


def build_lstm_model(input_shape):
    """
    Build a 4-layer LSTM neural network for stock price prediction.
    
    Args:
        input_shape: Shape of input sequences (time_steps, features)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        LSTM(units=LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(units=LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(units=LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_prediction_model():
    """
    Train LSTM model on historical stock price data.
    
    Returns:
        bool: True if training successful
    """
    print("Loading stock price data...")
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Extract closing prices (column index 5)
    close_prices = df.iloc[:, 5:6].values
    
    # Normalize data to [0, 1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_prices = scaler.fit_transform(close_prices)
    
    # Prepare training sequences
    X_train, y_train = prepare_sequences(normalized_prices, SEQUENCE_LENGTH)
    
    print(f"Training on {len(X_train)} sequences...")
    
    # Build and train model
    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
    
    # Save trained model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(MODEL_FILE_PATH)
    print(f"Model saved to {MODEL_FILE_PATH}")
    
    return True


def send_email_notification():
    """
    Send email notification with stock price data attachment.
    
    Returns:
        bool: True if email sent successfully
    """
    sender_email = 'ainoodle.tech@gmail.com'
    recipient_email = 'tuph.alex@gmail.com'
    
    # Compose email message
    email_body = (
        '<p>Hi,</p>'
        f'<p>Your daily stock price data for {STOCK_SYMBOL} is attached.</p>'
        '<p>Best regards</p>'
    )
    
    message = Mail(
        from_email=sender_email,
        to_emails=recipient_email,
        subject=f'Stock Price Data - {STOCK_SYMBOL}',
        html_content=email_body
    )
    
    # Attach CSV file
    try:
        with open(CSV_FILE_PATH, 'rb') as file:
            file_data = file.read()
        
        encoded_content = base64.b64encode(file_data).decode()
        
        attachment = Attachment(
            FileContent(encoded_content),
            FileName('stock_data.csv'),
            FileType('text/csv'),
            Disposition('attachment')
        )
        message.attachment = attachment
        
        # Send email via SendGrid
        client = SendGridAPIClient("Send Grid Token here")
        response = client.send(message)
        
        print(f"Email sent successfully at {datetime.now()}")
        return True
        
    except Exception as error:
        print(f"Error sending email: {str(error)}")
        return False

# DAG default arguments
default_args = {
    'owner': 'tuph',
    'email': ['tuph.alex@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
with DAG(
    dag_id='stock_price_prediction_lstm',
    default_args=default_args,
    description='ML pipeline for stock price prediction using LSTM',
    schedule_interval=timedelta(days=1),
    start_date=datetime.today() - timedelta(days=1),
    catchup=False,
    tags=['machine-learning', 'stock-prediction', 'lstm'],
) as dag:
    
    # Task 1: Fetch stock price data from API
    fetch_data_task = PythonOperator(
        task_id='fetch_stock_data',
        python_callable=fetch_stock_data,
        op_kwargs={"to_date": "{{ ds }}"},
    )
    
    # Task 2: Train LSTM prediction model
    train_task = PythonOperator(
        task_id='train_lstm_model',
        python_callable=train_prediction_model,
    )
    
    # Task 3: Send email notification
    notify_task = PythonOperator(
        task_id='send_notification',
        python_callable=send_email_notification,
    )
    
    # Define task dependencies
    fetch_data_task >> train_task >> notify_task