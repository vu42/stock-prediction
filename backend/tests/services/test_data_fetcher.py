"""
Tests for data fetcher service.
Migrated from original tests/test_data_fetcher.py to use new structure.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from app.db.models import CrawlMetadata, Stock, StockPrice
from app.services.data_fetcher import (
    fetch_all_stocks,
    fetch_stock_data,
    get_last_data_date,
    insert_stock_data,
    load_stock_data_from_db,
)


@pytest.fixture
def test_stock(db: Session) -> Stock:
    """Create a test stock."""
    stock = Stock(
        ticker="VCB",
        name="Vietcombank",
        is_active=True,
    )
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return stock


@pytest.fixture
def mock_api_response():
    """Sample API response matching VNDirect API format."""
    return {
        "data": [
            {
                "date": "2024-01-01",
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "nmvolume": 1000000,
                "code": "VCB",
            },
            {
                "date": "2024-01-02",
                "open": 103.0,
                "high": 107.0,
                "low": 102.0,
                "close": 106.0,
                "nmvolume": 1200000,
                "code": "VCB",
            },
        ]
    }


class TestGetLastDataDate:
    """Test cases for get_last_data_date function."""

    def test_no_data_returns_none(self, db: Session):
        """Test when no data exists."""
        result = get_last_data_date(db, "VCB")
        assert result is None

    def test_with_stock_prices(self, db: Session, test_stock: Stock):
        """Test with existing stock prices."""
        # Add price data
        price = StockPrice(
            stock_id=test_stock.id,
            price_date=datetime(2024, 1, 15).date(),
            close_price=100.0,
        )
        db.add(price)
        db.commit()
        
        result = get_last_data_date(db, "VCB")
        assert result == "2024-01-15"

    def test_with_crawl_metadata(self, db: Session):
        """Test with crawl metadata only."""
        meta = CrawlMetadata(
            stock_symbol="VCB",
            last_data_date=datetime(2024, 1, 10).date(),
        )
        db.add(meta)
        db.commit()
        
        result = get_last_data_date(db, "VCB")
        assert result == "2024-01-10"


class TestInsertStockData:
    """Test cases for insert_stock_data function."""

    def test_insert_new_data(self, db: Session):
        """Test inserting new stock data."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "open": [100.0, 103.0],
            "high": [105.0, 107.0],
            "low": [98.0, 102.0],
            "close": [103.0, 106.0],
            "volume": [1000000, 1200000],
        })
        
        count = insert_stock_data(db, df, "VCB")
        
        assert count == 2

    def test_upsert_existing_data(self, db: Session, test_stock: Stock):
        """Test updating existing data."""
        # First insert
        df1 = pd.DataFrame({
            "date": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [98.0],
            "close": [103.0],
            "volume": [1000000],
        })
        insert_stock_data(db, df1, "VCB")
        
        # Update with new close price
        df2 = pd.DataFrame({
            "date": ["2024-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [98.0],
            "close": [110.0],  # Changed
            "volume": [1000000],
        })
        insert_stock_data(db, df2, "VCB")
        
        # Verify update
        price = db.query(StockPrice).filter(
            StockPrice.stock_id == test_stock.id
        ).first()
        assert float(price.close_price) == 110.0


class TestFetchStockData:
    """Test cases for fetch_stock_data function."""

    @patch("app.services.data_fetcher.urlopen")
    def test_initial_crawl(self, mock_urlopen, db: Session, mock_api_response):
        """Test initial crawl when no data exists."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode()
        mock_urlopen.return_value = mock_response
        
        result = fetch_stock_data(db, "VCB", to_date="2024-01-03")
        
        assert result is True
        assert mock_urlopen.called

    @patch("app.services.data_fetcher.urlopen")
    def test_no_new_data(self, mock_urlopen, db: Session):
        """Test when API returns empty data."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"data": []}).encode()
        mock_urlopen.return_value = mock_response
        
        result = fetch_stock_data(db, "VCB", to_date="2024-01-03")
        
        assert result is True  # Not an error

    @patch("app.services.data_fetcher.urlopen")
    def test_api_error(self, mock_urlopen, db: Session):
        """Test handling of API errors."""
        mock_urlopen.side_effect = TimeoutError("Connection timeout")
        
        result = fetch_stock_data(db, "VCB", to_date="2024-01-03")
        
        assert result is False


class TestFetchAllStocks:
    """Test cases for fetch_all_stocks function."""

    @patch("app.services.data_fetcher.fetch_stock_data")
    def test_fetch_all_success(self, mock_fetch, db: Session):
        """Test fetching multiple stocks successfully."""
        mock_fetch.return_value = True
        
        results = fetch_all_stocks(db, ["VCB", "FPT", "VNM"])
        
        assert len(results["success"]) == 3
        assert len(results["failed"]) == 0

    @patch("app.services.data_fetcher.fetch_stock_data")
    def test_fetch_all_partial_failure(self, mock_fetch, db: Session):
        """Test with some failures."""
        def side_effect(db, stock, **kwargs):
            return stock != "FPT"
        
        mock_fetch.side_effect = side_effect
        
        results = fetch_all_stocks(db, ["VCB", "FPT", "VNM"])
        
        assert len(results["success"]) == 2
        assert len(results["failed"]) == 1
        assert "FPT" in results["failed"]


class TestLoadStockDataFromDb:
    """Test cases for load_stock_data_from_db function."""

    def test_load_empty(self, db: Session):
        """Test loading when no data exists."""
        df = load_stock_data_from_db(db, "VCB")
        assert df.empty

    def test_load_with_data(self, db: Session, test_stock: Stock):
        """Test loading existing data."""
        # Add some prices
        for i in range(5):
            price = StockPrice(
                stock_id=test_stock.id,
                price_date=datetime(2024, 1, i + 1).date(),
                open_price=100 + i,
                high_price=105 + i,
                low_price=98 + i,
                close_price=103 + i,
                volume=1000000,
            )
            db.add(price)
        db.commit()
        
        df = load_stock_data_from_db(db, "VCB")
        
        assert len(df) == 5
        assert "close" in df.columns
        assert "date" in df.columns

