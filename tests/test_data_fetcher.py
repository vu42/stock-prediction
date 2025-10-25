"""
Test suite for data_fetcher.py module
Tests incremental crawling, API integration, and error handling
"""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import pytest

# Import the module to test
from modules.data_fetcher import fetch_stock_data, fetch_all_stocks


class TestFetchStockData:
    """Test cases for fetch_stock_data function"""

    @pytest.fixture
    def mock_api_response(self):
        """Sample API response matching VNDirect API format"""
        return {
            "data": [
                {
                    "date": "2024-01-01",
                    "open": 100.0,
                    "high": 105.0,
                    "low": 98.0,
                    "close": 103.0,
                    "volume": 1000000,
                    "code": "VCB",
                },
                {
                    "date": "2024-01-02",
                    "open": 103.0,
                    "high": 107.0,
                    "low": 102.0,
                    "close": 106.0,
                    "volume": 1200000,
                    "code": "VCB",
                },
                {
                    "date": "2024-01-03",
                    "open": 106.0,
                    "high": 108.0,
                    "low": 105.0,
                    "close": 107.0,
                    "volume": 1100000,
                    "code": "VCB",
                },
            ]
        }

    @pytest.fixture
    def context_with_date(self):
        """Airflow context with execution date"""
        return {"to_date": "2024-01-03"}

    @patch("modules.data_fetcher.insert_stock_data")
    @patch("modules.data_fetcher.get_last_data_date")
    @patch("modules.data_fetcher.urlopen")
    def test_initial_crawl_no_existing_data(
        self,
        mock_urlopen,
        mock_get_last_date,
        mock_insert,
        mock_api_response,
        context_with_date,
    ):
        """
        Test initial crawl when no data exists in database.
        Should fetch from DATA_START_DATE to today.
        """
        # Setup mocks
        mock_get_last_date.return_value = None  # No existing data
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode()
        mock_urlopen.return_value = mock_response
        mock_insert.return_value = 3

        # Execute
        result = fetch_stock_data("VCB", **context_with_date)

        # Assertions
        assert result is True
        mock_get_last_date.assert_called_once_with("VCB")

        # Check API call was made with correct parameters
        assert mock_urlopen.called
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]

        # Verify URL contains initial start date
        assert "2000-01-01" in request_obj.full_url
        assert "2024-01-03" in request_obj.full_url
        assert "VCB" in request_obj.full_url

        # Verify data was inserted
        mock_insert.assert_called_once()
        df_arg = mock_insert.call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 3
        assert "VCB" == mock_insert.call_args[0][1]

    @patch("modules.data_fetcher.insert_stock_data")
    @patch("modules.data_fetcher.get_last_data_date")
    @patch("modules.data_fetcher.urlopen")
    def test_incremental_crawl_with_existing_data(
        self,
        mock_urlopen,
        mock_get_last_date,
        mock_insert,
        mock_api_response,
        context_with_date,
    ):
        """
        Test incremental crawl when data exists.
        Should only fetch from (last_date + 1) onwards.
        """
        # Setup mocks - last date is 2023-12-31
        mock_get_last_date.return_value = "2023-12-31"
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode()
        mock_urlopen.return_value = mock_response
        mock_insert.return_value = 3

        # Execute
        result = fetch_stock_data("VCB", **context_with_date)

        # Assertions
        assert result is True

        # Check API call fetches from day after last_date
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert "2024-01-01" in request_obj.full_url  # Day after 2023-12-31
        assert "2024-01-03" in request_obj.full_url

    @patch("modules.data_fetcher.get_last_data_date")
    def test_already_up_to_date(self, mock_get_last_date, context_with_date):
        """
        Test when data is already up-to-date.
        Should return True without making API call.
        """
        # Last date is same as or after end_date
        mock_get_last_date.return_value = "2024-01-03"

        # Execute
        result = fetch_stock_data("VCB", **context_with_date)

        # Assertions
        assert result is True
        mock_get_last_date.assert_called_once_with("VCB")

    @patch("modules.data_fetcher.insert_stock_data")
    @patch("modules.data_fetcher.get_last_data_date")
    @patch("modules.data_fetcher.urlopen")
    def test_no_new_data_from_api(self, mock_urlopen, mock_get_last_date, mock_insert):
        """
        Test when API returns empty data (e.g., weekend/holiday).
        Should return True without error.
        """
        # Setup mocks
        mock_get_last_date.return_value = "2024-01-01"
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"data": []}).encode()
        mock_urlopen.return_value = mock_response

        # Execute
        context = {"to_date": "2024-01-06"}
        result = fetch_stock_data("VCB", **context)

        # Assertions
        assert result is True
        mock_insert.assert_not_called()  # No data to insert

    @patch("modules.data_fetcher.get_last_data_date")
    @patch("modules.data_fetcher.urlopen")
    def test_api_timeout_error(self, mock_urlopen, mock_get_last_date):
        """
        Test handling of API timeout.
        Should catch exception and return False.
        """
        # Setup mocks
        mock_get_last_date.return_value = None
        mock_urlopen.side_effect = TimeoutError("Connection timeout")

        # Execute
        context = {"to_date": "2024-01-03"}
        result = fetch_stock_data("VCB", **context)

        # Assertions
        assert result is False

    @patch("modules.data_fetcher.get_last_data_date")
    @patch("modules.data_fetcher.urlopen")
    def test_api_invalid_json(self, mock_urlopen, mock_get_last_date):
        """
        Test handling of invalid JSON response from API.
        Should catch exception and return False.
        """
        # Setup mocks
        mock_get_last_date.return_value = None
        mock_response = MagicMock()
        mock_response.read.return_value = b"Invalid JSON{{"
        mock_urlopen.return_value = mock_response

        # Execute
        context = {"to_date": "2024-01-03"}
        result = fetch_stock_data("VCB", **context)

        # Assertions
        assert result is False

    @patch("modules.data_fetcher.insert_stock_data")
    @patch("modules.data_fetcher.get_last_data_date")
    @patch("modules.data_fetcher.urlopen")
    def test_database_insert_failure(
        self, mock_urlopen, mock_get_last_date, mock_insert, mock_api_response
    ):
        """
        Test handling of database insert failure.
        Should catch exception and return False.
        """
        # Setup mocks
        mock_get_last_date.return_value = None
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode()
        mock_urlopen.return_value = mock_response
        mock_insert.side_effect = Exception("Database connection failed")

        # Execute
        context = {"to_date": "2024-01-03"}
        result = fetch_stock_data("VCB", **context)

        # Assertions
        assert result is False

    @patch("modules.data_fetcher.insert_stock_data")
    @patch("modules.data_fetcher.get_last_data_date")
    @patch("modules.data_fetcher.urlopen")
    def test_column_normalization(
        self,
        mock_urlopen,
        mock_get_last_date,
        mock_insert,
        context_with_date,
    ):
        """
        Test that column names are normalized to lowercase.
        """
        # Setup mocks with mixed-case columns
        api_response = {
            "data": [
                {
                    "Date": "2024-01-01",
                    "Open": 100.0,
                    "High": 105.0,
                    "Low": 98.0,
                    "Close": 103.0,
                    "Volume": 1000000,
                }
            ]
        }
        mock_get_last_date.return_value = None
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(api_response).encode()
        mock_urlopen.return_value = mock_response
        mock_insert.return_value = 1

        # Execute
        result = fetch_stock_data("VCB", **context_with_date)

        # Assertions
        assert result is True
        df_arg = mock_insert.call_args[0][0]
        assert all(col.islower() for col in df_arg.columns)

    def test_context_without_to_date(self):
        """
        Test that function uses current date when to_date not in context.
        """
        with patch(
            "modules.data_fetcher.get_last_data_date"
        ) as mock_get_last_date, patch(
            "modules.data_fetcher.urlopen"
        ) as mock_urlopen, patch(
            "modules.data_fetcher.insert_stock_data"
        ) as mock_insert:

            mock_get_last_date.return_value = None
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({"data": []}).encode()
            mock_urlopen.return_value = mock_response

            # Execute without to_date in context
            result = fetch_stock_data("VCB")

            # Should complete successfully
            assert result is True


class TestFetchAllStocks:
    """Test cases for fetch_all_stocks function"""

    @patch("modules.data_fetcher.fetch_stock_data")
    def test_fetch_all_stocks_success(self, mock_fetch):
        """
        Test fetching multiple stocks successfully.
        """
        # Setup mock - all succeed
        mock_fetch.return_value = True

        # Execute
        stocks = ["VCB", "FPT", "VNM"]
        context = {"to_date": "2024-01-03"}
        results = fetch_all_stocks(stocks, **context)

        # Assertions
        assert len(results["success"]) == 3
        assert len(results["failed"]) == 0
        assert results["success"] == ["VCB", "FPT", "VNM"]
        assert mock_fetch.call_count == 3

    @patch("modules.data_fetcher.fetch_stock_data")
    def test_fetch_all_stocks_partial_failure(self, mock_fetch):
        """
        Test fetching multiple stocks with some failures.
        """

        # Setup mock - FPT fails
        def side_effect(stock, **kwargs):
            return stock != "FPT"

        mock_fetch.side_effect = side_effect

        # Execute
        stocks = ["VCB", "FPT", "VNM"]
        context = {"to_date": "2024-01-03"}
        results = fetch_all_stocks(stocks, **context)

        # Assertions
        assert len(results["success"]) == 2
        assert len(results["failed"]) == 1
        assert "VCB" in results["success"]
        assert "VNM" in results["success"]
        assert "FPT" in results["failed"]

    @patch("modules.data_fetcher.fetch_stock_data")
    def test_fetch_all_stocks_all_fail(self, mock_fetch):
        """
        Test when all stocks fail to fetch.
        """
        # Setup mock - all fail
        mock_fetch.return_value = False

        # Execute
        stocks = ["VCB", "FPT", "VNM"]
        context = {"to_date": "2024-01-03"}
        results = fetch_all_stocks(stocks, **context)

        # Assertions
        assert len(results["success"]) == 0
        assert len(results["failed"]) == 3
        assert results["failed"] == ["VCB", "FPT", "VNM"]

    @patch("modules.data_fetcher.fetch_stock_data")
    def test_fetch_all_stocks_empty_list(self, mock_fetch):
        """
        Test with empty stock list.
        """
        # Execute
        results = fetch_all_stocks([], to_date="2024-01-03")

        # Assertions
        assert len(results["success"]) == 0
        assert len(results["failed"]) == 0
        mock_fetch.assert_not_called()


class TestIntegrationScenarios:
    """Integration-style tests for realistic scenarios"""

    @patch("modules.data_fetcher.insert_stock_data")
    @patch("modules.data_fetcher.get_last_data_date")
    @patch("modules.data_fetcher.urlopen")
    def test_weekly_update_scenario(
        self, mock_urlopen, mock_get_last_date, mock_insert
    ):
        """
        Simulate weekly update: last data was Friday, now Monday.
        Should fetch weekend + Monday (but API returns only Monday).
        """
        # Setup: Last data was Friday
        mock_get_last_date.return_value = "2024-01-05"  # Friday

        # API returns only Monday data (weekend is empty)
        api_response = {
            "data": [
                {
                    "date": "2024-01-08",  # Monday
                    "open": 100.0,
                    "high": 105.0,
                    "low": 98.0,
                    "close": 103.0,
                    "volume": 1000000,
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(api_response).encode()
        mock_urlopen.return_value = mock_response
        mock_insert.return_value = 1

        # Execute - running on Monday
        context = {"to_date": "2024-01-08"}
        result = fetch_stock_data("VCB", **context)

        # Assertions
        assert result is True

        # Should request from Saturday onwards
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert "2024-01-06" in request_obj.full_url  # Saturday (Friday + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
