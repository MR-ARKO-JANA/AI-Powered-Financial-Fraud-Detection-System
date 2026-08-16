"""
test_alert_system.py — Tests for the alert trigger and feedback system.
"""

import pytest
from src.alert_system import (
    init_alerts_db,
    process_and_trigger_alert,
    fetch_active_alerts,
    submit_analyst_feedback,
)


class TestInitAlertsDB:
    """Tests for alert database initialization."""

    def test_init_creates_tables(self):
        """Should create model_alerts and training_feedback_logs tables."""
        init_alerts_db()
        # Should not raise - tables exist
        alerts = fetch_active_alerts("PENDING")
        assert isinstance(alerts, list)

    def test_init_idempotent(self):
        """Calling init multiple times should not fail."""
        init_alerts_db()
        init_alerts_db()
        init_alerts_db()


class TestProcessAndTriggerAlert:
    """Tests for alert triggering."""

    def test_critical_alert_triggered(self, sample_clean_data):
        """Risk score >= 80 should trigger CRITICAL alert."""
        verdict = {"fraud_type": "Account Takeover", "risk_score": 90.0, "reason": "High risk"}
        checklist = ["Blacklisted Merchant"]

        result = process_and_trigger_alert(sample_clean_data, verdict, checklist)
        assert result is not None
        assert result["severity"] == "CRITICAL"
        assert result["status"] == "PENDING"

    def test_warning_alert_triggered(self, sample_clean_data):
        """Risk score 60-79 should trigger WARNING alert."""
        verdict = {"fraud_type": "Suspicious", "risk_score": 65.0, "reason": "Moderate risk"}
        checklist = []

        result = process_and_trigger_alert(sample_clean_data, verdict, checklist)
        assert result is not None
        assert result["severity"] == "WARNING"

    def test_blacklisted_merchant_triggers_warning(self, sample_clean_data):
        """Blacklisted merchant in checklist should trigger WARNING even with low score."""
        verdict = {"fraud_type": "Unknown", "risk_score": 30.0, "reason": "Low score"}
        checklist = ["Blacklisted Merchant (DARKWEB*PAY)"]

        result = process_and_trigger_alert(sample_clean_data, verdict, checklist)
        assert result is not None
        assert result["severity"] == "WARNING"

    def test_safe_transaction_no_alert(self, sample_safe_data):
        """Safe transactions should not trigger alerts."""
        verdict = {"fraud_type": "Safe/Legitimate", "risk_score": 10.0, "reason": "Normal"}
        checklist = ["New Device: MISSING"]

        result = process_and_trigger_alert(sample_safe_data, verdict, checklist)
        assert result is None

    def test_alert_payload_fields(self, sample_clean_data):
        """Alert payload should contain all required fields."""
        verdict = {"fraud_type": "Account Takeover", "risk_score": 85.0, "reason": "High risk"}
        result = process_and_trigger_alert(sample_clean_data, verdict, [])

        assert "alert_id" in result
        assert "transaction_id" in result
        assert "client_id" in result
        assert "amount" in result
        assert "fraud_type" in result
        assert "risk_score" in result
        assert "severity" in result

    def test_alert_stored_in_db(self, sample_clean_data):
        """Triggered alert should appear in database."""
        verdict = {"fraud_type": "Card Skimming", "risk_score": 82.0, "reason": "Test"}
        process_and_trigger_alert(sample_clean_data, verdict, [])

        alerts = fetch_active_alerts("PENDING")
        assert len(alerts) >= 1


class TestFetchActiveAlerts:
    """Tests for fetching alerts."""

    def test_fetch_empty(self):
        """Should return empty list when no alerts exist."""
        alerts = fetch_active_alerts("PENDING")
        assert alerts == []

    def test_fetch_by_status(self, sample_clean_data):
        """Should filter by status."""
        verdict = {"fraud_type": "Fraud", "risk_score": 90.0, "reason": "test"}
        process_and_trigger_alert(sample_clean_data, verdict, [])

        pending = fetch_active_alerts("PENDING")
        resolved = fetch_active_alerts("CONFIRMED_FRAUD")
        assert len(pending) >= 1
        assert len(resolved) == 0


class TestSubmitFeedback:
    """Tests for analyst feedback submission."""

    def test_confirm_fraud(self, sample_clean_data):
        """Should mark alert as CONFIRMED_FRAUD."""
        verdict = {"fraud_type": "Fraud", "risk_score": 90.0, "reason": "test"}
        alert = process_and_trigger_alert(sample_clean_data, verdict, [])

        result = submit_analyst_feedback(alert["alert_id"], confirmed_is_fraud=True)
        assert result["status"] == "success"
        assert result["resolved_as"] == "CONFIRMED_FRAUD"

    def test_mark_false_positive(self, sample_clean_data):
        """Should mark alert as FALSE_POSITIVE."""
        verdict = {"fraud_type": "Fraud", "risk_score": 85.0, "reason": "test"}
        alert = process_and_trigger_alert(sample_clean_data, verdict, [])

        result = submit_analyst_feedback(alert["alert_id"], confirmed_is_fraud=False)
        assert result["status"] == "success"
        assert result["resolved_as"] == "FALSE_POSITIVE"

    def test_feedback_with_transaction_data(self, sample_clean_data):
        """Should log training feedback when transaction data provided."""
        verdict = {"fraud_type": "Fraud", "risk_score": 90.0, "reason": "test"}
        alert = process_and_trigger_alert(sample_clean_data, verdict, [])

        result = submit_analyst_feedback(
            alert["alert_id"],
            confirmed_is_fraud=True,
            clean_data=sample_clean_data
        )
        assert result["status"] == "success"

    def test_feedback_nonexistent_alert(self):
        """Should succeed even for non-existent alert ID (no rows updated)."""
        result = submit_analyst_feedback("NONEXISTENT-ALERT", confirmed_is_fraud=True)
        assert result["status"] == "success"
