"""
Tests for stock API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import Stock


@pytest.fixture
def test_stock(db: Session) -> Stock:
    """Create a test stock."""
    stock = Stock(
        ticker="VCB",
        name="Vietcombank",
        sector="Banking",
        exchange="HOSE",
        description="Vietnam Joint Stock Commercial Bank for Foreign Trade",
        is_active=True,
    )
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return stock


class TestListStocks:
    """Test cases for listing stocks."""

    def test_list_stocks_empty(self, client: TestClient, db: Session):
        """Test listing stocks when empty."""
        response = client.get("/api/v1/stocks")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_list_stocks_with_data(self, client: TestClient, test_stock: Stock):
        """Test listing stocks with data."""
        response = client.get("/api/v1/stocks")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "VCB"
        assert data[0]["name"] == "Vietcombank"


class TestGetStock:
    """Test cases for getting a single stock."""

    def test_get_stock_success(self, client: TestClient, test_stock: Stock):
        """Test getting a stock by ticker."""
        response = client.get("/api/v1/stocks/VCB")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "VCB"
        assert data["sector"] == "Banking"

    def test_get_stock_not_found(self, client: TestClient):
        """Test getting a non-existent stock."""
        response = client.get("/api/v1/stocks/NONEXISTENT")
        
        assert response.status_code == 404


class TestCreateStock:
    """Test cases for creating stocks."""

    def test_create_stock_success(self, client: TestClient, db: Session):
        """Test creating a new stock."""
        response = client.post(
            "/api/v1/stocks",
            json={
                "ticker": "FPT",
                "name": "FPT Corporation",
                "sector": "Technology",
                "exchange": "HOSE",
            },
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["ticker"] == "FPT"
        assert data["name"] == "FPT Corporation"


class TestUpdateStock:
    """Test cases for updating stocks."""

    def test_update_stock_success(self, client: TestClient, test_stock: Stock):
        """Test updating an existing stock."""
        response = client.patch(
            "/api/v1/stocks/VCB",
            json={"description": "Updated description"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "VCB"


class TestTopPicks:
    """Test cases for top picks endpoint."""

    def test_top_picks_empty(self, client: TestClient, db: Session):
        """Test top picks when no predictions exist."""
        response = client.get("/api/v1/stocks/top-picks")
        
        assert response.status_code == 200
        assert response.json() == []


class TestMarketTable:
    """Test cases for market table endpoint."""

    def test_market_table_empty(self, client: TestClient, db: Session):
        """Test market table when empty."""
        response = client.get("/api/v1/stocks/market-table")
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["meta"]["total"] == 0

