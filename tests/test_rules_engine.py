"""
test_rules_engine.py — Tests for the configurable fraud rules engine.
"""

import pytest
import pandas as pd
from src.rules_engine import apply_rules, load_config, build_user_norms, _safe_get


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_nonexistent_returns_defaults(self):
        """Should return default config when file not found."""
        config = load_config("/nonexistent/path.json")
        assert "thresholds" in config
        assert "lists" in config
        assert "behavioral_rules" in config

    def test_default_thresholds_present(self):
        """Default config should have required threshold keys."""
        config = load_config("/nonexistent/path.json")
        assert "high_amount" in config["thresholds"]
        assert "high_velocity_count" in config["thresholds"]
        assert "impossible_travel_speed_kmh" in config["thresholds"]


class TestSafeGet:
    """Tests for the _safe_get utility."""

    def test_existing_value(self):
        """Should return existing value."""
        row = pd.Series({"amount": 100.0})
        assert _safe_get(row, "amount") == 100.0

    def test_missing_key_returns_default(self):
        """Should return default for missing keys."""
        row = pd.Series({"amount": 100.0})
        assert _safe_get(row, "missing", "default") == "default"

    def test_nan_returns_default(self):
        """Should return default for NaN values."""
        row = pd.Series({"amount": float("nan")})
        assert _safe_get(row, "amount", 0.0) == 0.0


class TestBuildUserNorms:
    """Tests for building behavioral norms."""

    def test_empty_dataframe(self):
        """Should return empty norms for empty DataFrame."""
        norms = build_user_norms(pd.DataFrame())
        assert norms == {}

    def test_none_dataframe(self):
        """Should return empty norms for None input."""
        norms = build_user_norms(None)
        assert norms == {}

    def test_builds_norms_correctly(self):
        """Should compute averages, methods, and locations per client."""
        df = pd.DataFrame({
            "client_id": ["USER-1", "USER-1", "USER-2"],
            "amount": [100.0, 200.0, 500.0],
            "use_chip": ["Chip", "Online", "Swipe"],
            "merchant_city": ["New York", "London", "Tokyo"],
        })
        norms = build_user_norms(df)

        assert "USER-1" in norms
        assert "USER-2" in norms
        assert norms["USER-1"]["avg_amount"] == 150.0
        assert norms["USER-1"]["count_amount"] == 2
        assert "chip" in norms["USER-1"]["known_methods"]
        assert "online" in norms["USER-1"]["known_methods"]

    def test_missing_columns_handled(self):
        """Should handle DataFrames missing expected columns."""
        df = pd.DataFrame({
            "client_id": ["USER-1"],
            "amount": [100.0],
        })
        norms = build_user_norms(df)
        assert "USER-1" in norms


class TestApplyRules:
    """Tests for the rules engine."""

    def test_empty_dataframe(self, test_config):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame()
        result = apply_rules(df, test_config)
        assert "suspicious_indicators_checklist" in result.columns
        assert "rule_risk_score" in result.columns

    def test_high_amount_flagged(self, test_config):
        """Transactions above threshold should be flagged."""
        df = pd.DataFrame([{
            "client_id": "USER-1",
            "amount": 15000.0,
            "merchant_id": "AMAZON",
            "use_chip": "Chip",
            "merchant_city": "New York",
        }])
        result = apply_rules(df, test_config)
        checklist = result["suspicious_indicators_checklist"].iloc[0]
        assert any("High Transaction Amount" in ind for ind in checklist)

    def test_low_amount_not_flagged(self, test_config):
        """Normal transactions should not be flagged for high amount."""
        df = pd.DataFrame([{
            "client_id": "USER-1",
            "amount": 50.0,
            "merchant_id": "AMAZON",
            "use_chip": "Chip",
            "merchant_city": "New York",
        }])
        result = apply_rules(df, test_config)
        checklist = result["suspicious_indicators_checklist"].iloc[0]
        assert not any("High Transaction Amount" in ind for ind in checklist)

    def test_high_velocity_flagged(self, test_config):
        """High velocity (>5 tx/hr) should be flagged."""
        df = pd.DataFrame([{
            "client_id": "USER-1",
            "amount": 100.0,
            "velocity_1h": 10,
            "merchant_id": "AMAZON",
        }])
        result = apply_rules(df, test_config)
        checklist = result["suspicious_indicators_checklist"].iloc[0]
        assert any("High Transaction Velocity" in ind for ind in checklist)

    def test_impossible_travel_speed_flagged(self, test_config):
        """Impossible travel speed should be flagged."""
        df = pd.DataFrame([{
            "client_id": "USER-1",
            "amount": 100.0,
            "travel_speed_kmh": 2000.0,
            "merchant_id": "AMAZON",
        }])
        result = apply_rules(df, test_config)
        checklist = result["suspicious_indicators_checklist"].iloc[0]
        assert any("Impossible Travel Speed" in ind for ind in checklist)

    def test_blacklisted_merchant_flagged(self, test_config):
        """Blacklisted merchants should be flagged."""
        df = pd.DataFrame([{
            "client_id": "USER-1",
            "amount": 100.0,
            "merchant_id": "DARKWEB*PAY",
        }])
        result = apply_rules(df, test_config)
        checklist = result["suspicious_indicators_checklist"].iloc[0]
        assert any("Blacklisted Merchant" in ind for ind in checklist)

    def test_risk_score_capped_at_100(self, test_config):
        """Risk score should never exceed 100."""
        df = pd.DataFrame([{
            "client_id": "USER-1",
            "amount": 15000.0,
            "velocity_1h": 15,
            "travel_speed_kmh": 3000.0,
            "merchant_id": "DARKWEB*PAY",
            "use_chip": "emulator",
            "merchant_city": "Unknown",
        }])
        result = apply_rules(df, test_config)
        assert result["rule_risk_score"].iloc[0] <= 100

    def test_safe_transaction_low_score(self, test_config):
        """Safe transactions should get low risk scores."""
        norms = {
            "USER-1": {
                "known_methods": {"chip"},
                "known_locations": {"new york"},
                "sum_amount": 500.0,
                "count_amount": 10,
                "avg_amount": 50.0,
            }
        }
        df = pd.DataFrame([{
            "client_id": "USER-1",
            "amount": 30.0,
            "velocity_1h": 1,
            "travel_speed_kmh": 0.0,
            "merchant_id": "AMAZON",
            "use_chip": "Chip",
            "merchant_city": "New York",
        }])
        result = apply_rules(df, test_config, precomputed_norms=norms)
        assert result["rule_risk_score"].iloc[0] == 0
