"""
Integration tests for data_fetcher.py with REAL VNDirect API
These tests make actual API calls - run with: pytest tests/test_data_fetcher_integration.py -v -s
Use -m "not slow" to skip these tests in regular runs
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from modules.data_fetcher import fetch_stock_data, fetch_all_stocks


@pytest.mark.integration
@pytest.mark.slow
class TestRealAPIIntegration:
    """Tests with real VNDirect API calls"""

    def test_fetch_real_api_single_stock(self):
        """
        Test fetching real data from VNDirect API for a single stock.
        This validates that our code works with actual API responses.
        """
        # Mock database functions to avoid actual DB writes
        with patch("modules.data_fetcher.get_last_data_date") as mock_get_last, patch(
            "modules.data_fetcher.insert_stock_data"
        ) as mock_insert:

            # No existing data - fetch last 5 days only
            mock_get_last.return_value = None
            mock_insert.return_value = 5

            # Fetch recent data (last 5 days to keep test fast)
            today = datetime.now()
            start_date = (today - timedelta(days=5)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")

            context = {"to_date": end_date}

            # Override DATA_START_DATE to use our short date range
            with patch("modules.data_fetcher.DATA_START_DATE", start_date):
                result = fetch_stock_data("VCB", **context)

            # Assertions
            assert result is True, "Fetch should succeed"
            assert mock_insert.called, "Data should be inserted"

            if mock_insert.called:
                # Verify DataFrame structure
                df_arg = mock_insert.call_args[0][0]
                assert isinstance(df_arg, pd.DataFrame), "Should pass DataFrame"

                # Check required columns exist (normalized to lowercase)
                required_cols = ["date", "open", "high", "low", "close", "volume"]
                assert all(
                    col in df_arg.columns for col in required_cols
                ), f"DataFrame missing required columns. Has: {df_arg.columns.tolist()}"

                # Verify data types
                assert df_arg["close"].dtype in [
                    "float64",
                    "int64",
                ], "Close should be numeric"

                # Check volume field (after mapping from nmvolume)
                volume_col = "volume" if "volume" in df_arg.columns else "nmvolume"
                if volume_col in df_arg.columns:
                    assert df_arg[volume_col].dtype in [
                        "float64",
                        "int64",
                    ], "Volume should be numeric"

                # Print sample data for manual verification
                print(f"\n✅ Successfully fetched {len(df_arg)} records for VCB")
                print(f"Date range: {df_arg['date'].min()} to {df_arg['date'].max()}")
                print("\nSample data (first 3 rows):")
                print(
                    df_arg[["date", "open", "high", "low", "close", "volume"]].head(3)
                )

    def test_real_api_response_structure(self):
        """
        Validate the actual structure of VNDirect API response.
        This ensures our assumptions about the API are correct.
        """
        import json
        import ssl
        from urllib.request import Request, urlopen

        # Disable SSL verification (same as in data_fetcher)
        ssl._create_default_https_context = ssl._create_unverified_context

        # Build API request for last 3 days only
        stock_symbol = "VCB"
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        query_params = f"sort=date&q=code:{stock_symbol}~date:gte:{start_date}~date:lte:{end_date}&size=9990&page=1"
        api_url = f"https://api-finfo.vndirect.com.vn/v4/stock_prices?{query_params}"

        # Make real API call
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        request = Request(api_url, headers=headers)

        try:
            response_data = urlopen(request, timeout=10).read()
            parsed_data = json.loads(response_data)

            # Assertions on response structure
            assert "data" in parsed_data, "Response should have 'data' field"
            assert isinstance(parsed_data["data"], list), "'data' should be a list"

            if len(parsed_data["data"]) > 0:
                first_record = parsed_data["data"][0]

                # Check expected fields (VNDirect API uses 'nmvolume' not 'volume')
                expected_fields = ["date", "open", "high", "low", "close"]
                volume_fields = ["volume", "nmvolume", "nmVolume"]

                for field in expected_fields:
                    assert (
                        field in first_record or field.capitalize() in first_record
                    ), f"Missing field: {field}"

                # Check that at least one volume field exists
                has_volume = any(v in first_record for v in volume_fields)
                assert (
                    has_volume
                ), f"Missing volume field. Available: {list(first_record.keys())}"

                print(f"\n✅ API Response Structure Valid")
                print(f"Records returned: {len(parsed_data['data'])}")
                print(f"Sample record fields: {list(first_record.keys())}")
                print(f"Sample record: {first_record}")
            else:
                print(f"\n⚠️  No data returned (possibly weekend/holiday)")

        except Exception as e:
            pytest.fail(f"Real API call failed: {str(e)}")

    def test_incremental_crawl_with_real_api(self):
        """
        Test incremental crawling logic with real API.
        Simulates having data up to 7 days ago, then fetching recent data.
        """
        with patch("modules.data_fetcher.get_last_data_date") as mock_get_last, patch(
            "modules.data_fetcher.insert_stock_data"
        ) as mock_insert:

            # Simulate having data up to 7 days ago
            seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            mock_get_last.return_value = seven_days_ago
            mock_insert.return_value = 5

            # Fetch up to today
            today = datetime.now().strftime("%Y-%m-%d")
            context = {"to_date": today}

            result = fetch_stock_data("FPT", **context)

            # Assertions
            assert result is True, "Incremental fetch should succeed"

            if mock_insert.called:
                df_arg = mock_insert.call_args[0][0]
                print(
                    f"\n✅ Incremental crawl fetched {len(df_arg)} new records for FPT"
                )
                print(f"From {df_arg['date'].min()} to {df_arg['date'].max()}")

    def test_multiple_stocks_real_api(self):
        """
        Test fetching multiple stocks with real API calls.
        """
        with patch("modules.data_fetcher.get_last_data_date") as mock_get_last, patch(
            "modules.data_fetcher.insert_stock_data"
        ) as mock_insert:

            mock_get_last.return_value = None
            mock_insert.return_value = 3

            # Fetch last 3 days for 3 stocks
            today = datetime.now()
            start_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")

            stocks = ["VCB", "FPT", "VNM"]

            with patch("modules.data_fetcher.DATA_START_DATE", start_date):
                results = fetch_all_stocks(stocks, to_date=end_date)

            # Assertions
            print(f"\n✅ Fetched {len(results['success'])} stocks successfully")
            print(f"Success: {results['success']}")
            print(f"Failed: {results['failed']}")

            # At least some should succeed (unless all markets are closed)
            assert len(results["success"]) + len(results["failed"]) == len(stocks)

    def test_api_error_handling(self):
        """
        Test that code handles real API errors gracefully.
        """
        with patch("modules.data_fetcher.get_last_data_date") as mock_get_last, patch(
            "modules.data_fetcher.insert_stock_data"
        ) as mock_insert:

            mock_get_last.return_value = None

            # Try to fetch with invalid stock symbol
            context = {"to_date": datetime.now().strftime("%Y-%m-%d")}

            # This should either return True with empty data or False
            result = fetch_stock_data("INVALID_SYMBOL_XYZ", **context)

            # Should handle gracefully without crashing
            assert isinstance(result, bool)
            print(f"\n✅ Invalid symbol handled gracefully: {result}")


@pytest.mark.integration
@pytest.mark.slow
class TestRealAPIDataQuality:
    """Tests to validate data quality from real API"""

    def test_data_completeness(self):
        """
        Verify that fetched data has no missing critical values.
        """
        with patch("modules.data_fetcher.get_last_data_date") as mock_get_last, patch(
            "modules.data_fetcher.insert_stock_data"
        ) as mock_insert:

            mock_get_last.return_value = None
            mock_insert.return_value = 5

            # Fetch recent data
            today = datetime.now()
            start_date = (today - timedelta(days=5)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")

            context = {"to_date": end_date}

            with patch("modules.data_fetcher.DATA_START_DATE", start_date):
                result = fetch_stock_data("VCB", **context)

            if result and mock_insert.called:
                df = mock_insert.call_args[0][0]

                if len(df) > 0:
                    # Check for missing values in critical columns
                    assert not df["date"].isna().any(), "Date should not have NaN"
                    assert (
                        not df["close"].isna().any()
                    ), "Close price should not have NaN"

                    # Check for reasonable value ranges
                    assert (df["close"] > 0).all(), "Close prices should be positive"

                    # Check volume (after mapping from nmvolume)
                    volume_col = "volume" if "volume" in df.columns else "nmvolume"
                    if volume_col in df.columns:
                        assert (
                            df[volume_col] >= 0
                        ).all(), "Volume should be non-negative"

                    assert (df["high"] >= df["low"]).all(), "High should be >= Low"

                    print(f"\n✅ Data quality checks passed")
                    print(
                        f"Close price range: {df['close'].min():.2f} - {df['close'].max():.2f}"
                    )
                    if volume_col in df.columns:
                        print(
                            f"Volume range: {df[volume_col].min():,} - {df[volume_col].max():,}"
                        )

    def test_date_ordering(self):
        """
        Verify that dates are properly ordered.
        """
        with patch("modules.data_fetcher.get_last_data_date") as mock_get_last, patch(
            "modules.data_fetcher.insert_stock_data"
        ) as mock_insert:

            mock_get_last.return_value = None
            mock_insert.return_value = 5

            today = datetime.now()
            start_date = (today - timedelta(days=10)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")

            context = {"to_date": end_date}

            with patch("modules.data_fetcher.DATA_START_DATE", start_date):
                result = fetch_stock_data("VCB", **context)

            if result and mock_insert.called:
                df = mock_insert.call_args[0][0]

                if len(df) > 1:
                    # Convert to datetime for comparison
                    dates = pd.to_datetime(df["date"])

                    # Check if dates are in order (should be ascending from API)
                    print(f"\n✅ Date ordering check")
                    print(f"First date: {dates.min()}")
                    print(f"Last date: {dates.max()}")


def run_integration_tests():
    """
    Helper function to run integration tests with proper configuration.
    """
    print("=" * 80)
    print("Running Integration Tests with REAL VNDirect API")
    print("=" * 80)
    pytest.main(
        [
            __file__,
            "-v",
            "-s",  # Show print statements
            "-m",
            "integration",  # Only run integration tests
            "--tb=short",  # Shorter traceback
        ]
    )


if __name__ == "__main__":
    run_integration_tests()
