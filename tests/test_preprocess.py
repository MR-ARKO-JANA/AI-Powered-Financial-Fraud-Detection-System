"""
test_preprocess.py — Tests for data cleaning and preprocessing pipeline.
"""

import pytest
import pandas as pd
from src.preprocess import clean_data, _resolve_ip_location, _sanitize_string


class TestCleanData:
    """Tests for the clean_data function."""

    def test_empty_dataframe(self):
        """clean_data should return empty DataFrame for empty input."""
        df = pd.DataFrame()
        result = clean_data(df)
        assert result.empty

    def test_amount_currency_cleaning(self):
        """Should strip $ and , from amount strings and convert to float."""
        df = pd.DataFrame({"amount": ["$1,234.56", "$99.99", "500"]})
        result = clean_data(df)
        assert result["amount"].iloc[0] == 1234.56
        assert result["amount"].iloc[1] == 99.99
        assert result["amount"].iloc[2] == 500.0

    def test_amount_nan_handling(self):
        """Should replace NaN amounts with 0.0."""
        df = pd.DataFrame({"amount": ["abc", None, ""]})
        result = clean_data(df)
        assert all(result["amount"] == 0.0)

    def test_merchant_alias_normalization(self):
        """Should normalize merchant aliases (AMZN*MKTP -> AMAZON)."""
        df = pd.DataFrame({"merchant_id": ["AMZN*MKTP", "flipkart*india", "STARBUCKS"]})
        result = clean_data(df)
        assert result["merchant_id"].iloc[0] == "AMAZON"
        assert result["merchant_id"].iloc[1] == "FLIPKART"
        assert result["merchant_id"].iloc[2] == "STARBUCKS"

    def test_merchant_id_uppercase(self):
        """Should uppercase all merchant IDs."""
        df = pd.DataFrame({"merchant_id": ["amazon", "Netflix"]})
        result = clean_data(df)
        assert all(result["merchant_id"] == result["merchant_id"].str.upper())

    def test_merchant_city_title_case(self):
        """Should title-case merchant cities."""
        df = pd.DataFrame({"merchant_city": ["new york", "LONDON", "tokyo"]})
        result = clean_data(df)
        assert result["merchant_city"].iloc[0] == "New York"
        assert result["merchant_city"].iloc[1] == "London"
        assert result["merchant_city"].iloc[2] == "Tokyo"

    def test_use_chip_capitalize(self):
        """Should capitalize chip usage values."""
        df = pd.DataFrame({"use_chip": ["online", "SWIPE", "chip"]})
        result = clean_data(df)
        assert result["use_chip"].iloc[0] == "Online"
        assert result["use_chip"].iloc[1] == "Swipe"
        assert result["use_chip"].iloc[2] == "Chip"

    def test_null_merchant_id_default(self):
        """Should fill null merchant_id with UNKNOWN."""
        df = pd.DataFrame({"merchant_id": [None, "STARBUCKS"]})
        result = clean_data(df)
        assert result["merchant_id"].iloc[0] == "UNKNOWN"

    def test_deduplication(self):
        """Should remove duplicate rows."""
        df = pd.DataFrame({"amount": [100, 100, 200], "merchant_id": ["A", "A", "B"]})
        result = clean_data(df)
        assert len(result) == 2

    def test_date_parsing(self):
        """Should parse date strings."""
        df = pd.DataFrame({"date": ["2026-03-22T10:30:00Z"]})
        result = clean_data(df)
        assert "2026" in str(result["date"].iloc[0])

    def test_ip_location_column_added(self):
        """Should add ip_location column when ip_address exists."""
        df = pd.DataFrame({"ip_address": ["192.168.1.1"]})
        result = clean_data(df)
        assert "ip_location" in result.columns
        assert result["ip_location"].iloc[0] == "Local Corporate Network"

    def test_zip_formatting(self):
        """Should zero-pad zip codes to 5 digits."""
        df = pd.DataFrame({"zip": ["123", "00000", "12345.0"]})
        result = clean_data(df)
        assert result["zip"].iloc[0] == "00123"
        assert result["zip"].iloc[1] == "00000"
        assert result["zip"].iloc[2] == "12345"


class TestIPGeolocation:
    """Tests for IP geolocation resolution."""

    def test_private_ip_returns_local(self):
        """Private IPs should return Local Corporate Network."""
        assert _resolve_ip_location("192.168.1.1") == "Local Corporate Network"
        assert _resolve_ip_location("10.0.0.1") == "Local Corporate Network"
        assert _resolve_ip_location("127.0.0.1") == "Local Corporate Network"

    def test_unknown_ip_returns_unknown(self):
        """Unknown/empty IPs should return Unknown Location."""
        assert _resolve_ip_location("") == "Unknown Location"
        assert _resolve_ip_location("unknown") == "Unknown Location"
        assert _resolve_ip_location("nan") == "Unknown Location"
        assert _resolve_ip_location("0.0.0.0") == "Unknown Location"


class TestSanitizeString:
    """Tests for input sanitization."""

    def test_normal_string_passes(self):
        """Normal alphanumeric strings should pass through."""
        assert _sanitize_string("Hello World 123") == "Hello World 123"

    def test_strips_dangerous_characters(self):
        """Should strip SQL injection and XSS characters."""
        result = _sanitize_string("DROP TABLE; <script>alert('xss')</script>")
        assert "<script>" not in result
        assert ";" not in result

    def test_length_limit(self):
        """Should truncate strings beyond max_length."""
        long_string = "A" * 1000
        result = _sanitize_string(long_string, max_length=100)
        assert len(result) == 100
