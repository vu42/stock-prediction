"""
Database operations for Stock Prediction System
Handles PostgreSQL interactions for incremental crawling
"""
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import pandas as pd
from config import DB_CONFIG


def get_db_connection():
    """Create PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_db_engine():
    """Create SQLAlchemy engine for pandas."""
    conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(conn_string)


def init_database():
    """
    Initialize database tables if they don't exist.
    Creates stock_prices and crawl_metadata tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create stock_prices table
        cursor.execute("""
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
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_date 
            ON stock_prices(stock_symbol, date DESC);
        """)
        
        # Create metadata table for tracking crawl status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crawl_metadata (
                stock_symbol VARCHAR(10) PRIMARY KEY,
                last_crawl_date DATE,
                last_data_date DATE,
                total_records INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        print("Database initialized successfully")
        
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
        cursor.execute("""
            SELECT MAX(date) FROM stock_prices WHERE stock_symbol = %s
        """, (stock_symbol,))
        
        result = cursor.fetchone()
        return result[0].strftime('%Y-%m-%d') if result and result[0] else None
        
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
            records.append((
                stock_symbol,
                pd.to_datetime(row.get('date') or row.get('Date')).date(),
                float(row.get('open') or row.get('Open', 0)),
                float(row.get('high') or row.get('High', 0)),
                float(row.get('low') or row.get('Low', 0)),
                float(row.get('close') or row.get('Close', 0)),
                int(row.get('volume') or row.get('Volume', 0))
            ))
        
        # Batch insert with ON CONFLICT UPDATE
        execute_values(cursor, """
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
        """, records)
        
        conn.commit()
        
        # Update crawl metadata
        last_date = pd.to_datetime(df['date' if 'date' in df.columns else 'Date'].max()).date()
        cursor.execute("""
            INSERT INTO crawl_metadata (stock_symbol, last_crawl_date, last_data_date, total_records)
            VALUES (%s, CURRENT_DATE, %s, (SELECT COUNT(*) FROM stock_prices WHERE stock_symbol = %s))
            ON CONFLICT (stock_symbol) 
            DO UPDATE SET 
                last_crawl_date = CURRENT_DATE,
                last_data_date = EXCLUDED.last_data_date,
                total_records = EXCLUDED.total_records,
                last_updated = CURRENT_TIMESTAMP
        """, (stock_symbol, last_date, stock_symbol))
        
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
    engine = get_db_engine()
    query = """
        SELECT date, open, high, low, close, volume 
        FROM stock_prices 
        WHERE stock_symbol = %s 
        ORDER BY date ASC
    """
    return pd.read_sql(query, engine, params=[stock_symbol])


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
        cursor.execute("""
            SELECT 
                stock_symbol,
                COUNT(*) as record_count,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM stock_prices
            GROUP BY stock_symbol
            ORDER BY stock_symbol
        """)
        
        stats = []
        for row in cursor.fetchall():
            stats.append({
                'symbol': row[0],
                'records': row[1],
                'first_date': row[2].strftime('%Y-%m-%d') if row[2] else None,
                'last_date': row[3].strftime('%Y-%m-%d') if row[3] else None
            })
        
        return stats
        
    finally:
        cursor.close()
        conn.close()

