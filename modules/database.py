"""
Database operations for Stock Prediction System
Handles PostgreSQL interactions for incremental crawling
"""

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from config import DB_CONFIG


def get_db_connection():
    """Create PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG)


# def get_db_engine():
#     """Create SQLAlchemy engine for pandas."""
#     conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
#     return create_engine(conn_string)


def init_database():
    """
    Initialize database tables if they don't exist.
    Creates stock_prices, crawl_metadata, predictions, and model_runs tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Create stock_prices table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_prices (
                id SERIAL PRIMARY KEY,
                stock_symbol VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(15, 2),
                high DECIMAL(15, 2),
                low DECIMAL(15, 2),
                close DECIMAL(15, 2) NOT NULL,
                volume BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stock_symbol, date)
            );
        """
        )

        # Create index for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stock_date 
            ON stock_prices(stock_symbol, date DESC);
        """
        )

        # Create metadata table for tracking crawl status
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_metadata (
                stock_symbol VARCHAR(10) PRIMARY KEY,
                last_crawl_date DATE,
                last_data_date DATE,
                total_records INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )

        # Create predictions table for storing model predictions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                stock_symbol VARCHAR(10) NOT NULL,
                prediction_date DATE NOT NULL,
                predicted_price DECIMAL(15, 2) NOT NULL,
                model_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_prediction UNIQUE(stock_symbol, prediction_date, created_at)
            );
        """
        )

        # Create index for predictions
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date 
            ON predictions(stock_symbol, prediction_date DESC);
        """
        )

        # Create model_runs table for tracking model performance
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
                id SERIAL PRIMARY KEY,
                stock_symbol VARCHAR(10) NOT NULL,
                model_type VARCHAR(50) DEFAULT 'ensemble',
                train_date DATE NOT NULL,
                rmse DECIMAL(10, 4),
                mae DECIMAL(10, 4),
                mape DECIMAL(10, 4),
                r2_score DECIMAL(10, 4),
                direction_accuracy DECIMAL(5, 2),
                training_samples INTEGER,
                test_samples INTEGER,
                feature_count INTEGER,
                model_path VARCHAR(255),
                scaler_path VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )

        # Create index for model_runs
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_runs_symbol_date 
            ON model_runs(stock_symbol, train_date DESC);
        """
        )

        conn.commit()
        print(
            "Database initialized successfully (stock_prices, crawl_metadata, predictions, model_runs)"
        )

    except Exception as e:
        conn.rollback()
        print(f"ERROR initializing database: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_last_data_date(stock_symbol):
    """
    Get the last date available for a stock in database.
    Used for incremental crawling.

    Returns:
        str: Last date in 'YYYY-MM-DD' format, or None if no data exists
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT MAX(date) FROM stock_prices WHERE stock_symbol = %s
        """,
            (stock_symbol,),
        )

        result = cursor.fetchone()
        return result[0].strftime("%Y-%m-%d") if result and result[0] else None

    finally:
        cursor.close()
        conn.close()


def insert_stock_data(df, stock_symbol):
    """
    Insert stock data into database with UPSERT (update on conflict).
    Also updates crawl_metadata table.

    Args:
        df: DataFrame with stock data (columns: date, open, high, low, close, volume)
        stock_symbol: Stock symbol

    Returns:
        int: Number of records inserted/updated
    """
    if df.empty:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Prepare records for batch insert
        records = []
        for _, row in df.iterrows():
            records.append(
                (
                    stock_symbol,
                    pd.to_datetime(row.get("date") or row.get("Date")).date(),
                    float(row.get("open") or row.get("Open", 0)),
                    float(row.get("high") or row.get("High", 0)),
                    float(row.get("low") or row.get("Low", 0)),
                    float(row.get("close") or row.get("Close", 0)),
                    int(row.get("volume") or row.get("Volume", 0)),
                )
            )

        # Batch insert with ON CONFLICT UPDATE
        execute_values(
            cursor,
            """
            INSERT INTO stock_prices (stock_symbol, date, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (stock_symbol, date) 
            DO UPDATE SET 
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                updated_at = CURRENT_TIMESTAMP
        """,
            records,
        )

        conn.commit()

        # Update crawl metadata
        last_date = pd.to_datetime(
            df["date" if "date" in df.columns else "Date"].max()
        ).date()
        cursor.execute(
            """
            INSERT INTO crawl_metadata (stock_symbol, last_crawl_date, last_data_date, total_records)
            VALUES (%s, CURRENT_DATE, %s, (SELECT COUNT(*) FROM stock_prices WHERE stock_symbol = %s))
            ON CONFLICT (stock_symbol) 
            DO UPDATE SET 
                last_crawl_date = CURRENT_DATE,
                last_data_date = EXCLUDED.last_data_date,
                total_records = EXCLUDED.total_records,
                last_updated = CURRENT_TIMESTAMP
        """,
            (stock_symbol, last_date, stock_symbol),
        )

        conn.commit()
        return len(records)

    except Exception as e:
        conn.rollback()
        print(f"ERROR inserting data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


def load_stock_data_from_db(stock_symbol):
    """
    Load all stock data for a symbol from PostgreSQL.

    Args:
        stock_symbol: Stock symbol to load

    Returns:
        DataFrame: Stock price data ordered by date
    """
    # Use the existing psycopg2 connection function
    conn = get_db_connection()
    try:
        query = """
            SELECT date, open, high, low, close, volume 
            FROM stock_prices 
            WHERE stock_symbol = %s 
            ORDER BY date ASC
        """
        # Pass the connection object directly to pd.read_sql
        return pd.read_sql(query, conn, params=[stock_symbol])
    finally:
        # It's good practice to close the connection
        conn.close()


def get_database_stats():
    """
    Get statistics about data in database.

    Returns:
        dict: Statistics including stock counts, date ranges, etc.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get stats per stock
        cursor.execute(
            """
            SELECT 
                stock_symbol,
                COUNT(*) as record_count,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM stock_prices
            GROUP BY stock_symbol
            ORDER BY stock_symbol
        """
        )

        stats = []
        for row in cursor.fetchall():
            stats.append(
                {
                    "symbol": row[0],
                    "records": row[1],
                    "first_date": row[2].strftime("%Y-%m-%d") if row[2] else None,
                    "last_date": row[3].strftime("%Y-%m-%d") if row[3] else None,
                }
            )

        return stats

    finally:
        cursor.close()
        conn.close()


def insert_predictions(stock_symbol, predictions_df, model_version="ensemble"):
    """
    Insert predictions into the predictions table.

    Args:
        stock_symbol: Stock symbol
        predictions_df: DataFrame with columns ['date', 'predicted_price']
        model_version: Version/type of model used for prediction

    Returns:
        int: Number of predictions inserted
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Prepare data for bulk insert
        records = [
            (stock_symbol, row["date"], float(row["predicted_price"]), model_version)
            for _, row in predictions_df.iterrows()
        ]

        # Bulk insert using execute_values (efficient for large datasets)
        query = """
            INSERT INTO predictions (stock_symbol, prediction_date, predicted_price, model_version)
            VALUES %s
            ON CONFLICT (stock_symbol, prediction_date, created_at) DO NOTHING
        """
        execute_values(cursor, query, records)

        conn.commit()
        inserted_count = cursor.rowcount
        print(f"[{stock_symbol}] Inserted {inserted_count} predictions into database")

        return inserted_count

    except Exception as e:
        conn.rollback()
        print(f"[{stock_symbol}] ERROR inserting predictions: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_latest_predictions(stock_symbol, days=30):
    """
    Get the latest predictions for a stock.

    Args:
        stock_symbol: Stock symbol
        days: Number of future days to retrieve

    Returns:
        DataFrame: Latest predictions
    """
    conn = get_db_connection()

    try:
        query = """
            SELECT prediction_date, predicted_price, model_version, created_at
            FROM predictions
            WHERE stock_symbol = %s
            AND created_at = (
                SELECT MAX(created_at) 
                FROM predictions 
                WHERE stock_symbol = %s
            )
            ORDER BY prediction_date ASC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(stock_symbol, stock_symbol, days))
        return df

    finally:
        conn.close()


def log_model_run(
    stock_symbol,
    train_date,
    metrics,
    model_path,
    scaler_path,
    training_samples,
    test_samples,
    feature_count,
    model_type="ensemble",
):
    """
    Log a model training run to the database.

    Args:
        stock_symbol: Stock symbol
        train_date: Date when model was trained (YYYY-MM-DD)
        metrics: Dict with keys: rmse, mae, mape, r2_score, direction_accuracy
        model_path: Path to saved model file
        scaler_path: Path to saved scaler file
        training_samples: Number of training samples
        test_samples: Number of test samples
        feature_count: Number of features used
        model_type: Type of model (default: ensemble)

    Returns:
        int: ID of inserted record
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        query = """
            INSERT INTO model_runs (
                stock_symbol, model_type, train_date,
                rmse, mae, mape, r2_score, direction_accuracy,
                training_samples, test_samples, feature_count,
                model_path, scaler_path
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s
            )
            RETURNING id
        """

        cursor.execute(
            query,
            (
                stock_symbol,
                model_type,
                train_date,
                metrics.get("rmse"),
                metrics.get("mae"),
                metrics.get("mape"),
                metrics.get("r2_score"),
                metrics.get("direction_accuracy"),
                training_samples,
                test_samples,
                feature_count,
                model_path,
                scaler_path,
            ),
        )

        run_id = cursor.fetchone()[0]
        conn.commit()

        print(
            f"[{stock_symbol}] Logged model run #{run_id} - RMSE: {metrics.get('rmse', 0):.4f}"
        )

        return run_id

    except Exception as e:
        conn.rollback()
        print(f"[{stock_symbol}] ERROR logging model run: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_model_performance_history(stock_symbol, limit=10):
    """
    Get model performance history for a stock.

    Args:
        stock_symbol: Stock symbol
        limit: Number of recent runs to retrieve

    Returns:
        DataFrame: Model performance history
    """
    conn = get_db_connection()

    try:
        query = """
            SELECT 
                train_date, model_type,
                rmse, mae, mape, r2_score, direction_accuracy,
                training_samples, test_samples, feature_count,
                created_at
            FROM model_runs
            WHERE stock_symbol = %s
            ORDER BY train_date DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(stock_symbol, limit))
        return df

    finally:
        conn.close()
