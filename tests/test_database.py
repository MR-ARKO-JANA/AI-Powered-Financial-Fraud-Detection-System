"""
test_database.py — Tests for SQLite database operations.
"""

import pytest
import pandas as pd
from src.database import init_db, log_transaction, fetch_recent_logs, fetch_all_logs, clear_logs, get_db_status


class TestInitDB:
    """Tests for database initialization."""

    def test_init_creates_schema(self):
        """init_db should create the transaction_logs table."""
        init_db()
        status = get_db_status()
        assert status["status"] == "healthy"
        assert status["exists"] is True

    def test_init_idempotent(self):
        """Calling init_db multiple times should not fail."""
        init_db()
        init_db()
        init_db()
        status = get_db_status()
        assert status["status"] == "healthy"


class TestLogTransaction:
    """Tests for logging transactions."""

    def test_log_single_transaction(self, sample_clean_data, sample_ai_verdict_fraud):
        """Should successfully log a transaction."""
        checklist = ["Blacklisted Merchant (DARKWEB*PAY)"]
        log_transaction(sample_clean_data, sample_ai_verdict_fraud, checklist)

        df = fetch_recent_logs(limit=10)
        assert len(df) == 1
        assert df.iloc[0]["transaction_id"] == "TXN-TEST-001"
        assert df.iloc[0]["composite_risk_score"] == 85.0

    def test_log_multiple_transactions(self, sample_clean_data, sample_safe_data, sample_ai_verdict_fraud, sample_ai_verdict_safe):
        """Should log multiple transactions."""
        log_transaction(sample_clean_data, sample_ai_verdict_fraud, [])
        log_transaction(sample_safe_data, sample_ai_verdict_safe, [])

        df = fetch_recent_logs(limit=10)
        assert len(df) == 2

    def test_log_with_empty_checklist(self, sample_clean_data, sample_ai_verdict_fraud):
        """Should handle None and empty checklists."""
        log_transaction(sample_clean_data, sample_ai_verdict_fraud, None)
        log_transaction(sample_clean_data, sample_ai_verdict_fraud, [])

        df = fetch_recent_logs(limit=10)
        assert len(df) == 2

    def test_log_preserves_data_types(self, sample_clean_data, sample_ai_verdict_fraud):
        """Should preserve numeric data types correctly."""
        log_transaction(sample_clean_data, sample_ai_verdict_fraud, [])

        df = fetch_recent_logs(limit=1)
        row = df.iloc[0]
        assert isinstance(row["amount"], float)
        assert isinstance(row["composite_risk_score"], float)


class TestFetchLogs:
    """Tests for fetching logs."""

    def test_fetch_empty_database(self):
        """Should return empty DataFrame for empty database."""
        df = fetch_recent_logs(limit=10)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_fetch_respects_limit(self, sample_clean_data, sample_ai_verdict_fraud):
        """Should respect the limit parameter."""
        for i in range(5):
            data = {**sample_clean_data, "transaction_id": f"TXN-{i}"}
            log_transaction(data, sample_ai_verdict_fraud, [])

        df = fetch_recent_logs(limit=3)
        assert len(df) == 3

    def test_fetch_all_logs(self, sample_clean_data, sample_ai_verdict_fraud):
        """fetch_all_logs should return all entries."""
        for i in range(3):
            data = {**sample_clean_data, "transaction_id": f"TXN-{i}"}
            log_transaction(data, sample_ai_verdict_fraud, [])

        df = fetch_all_logs()
        assert len(df) == 3


class TestClearLogs:
    """Tests for clearing logs."""

    def test_clear_empties_database(self, sample_clean_data, sample_ai_verdict_fraud):
        """clear_logs should remove all entries."""
        log_transaction(sample_clean_data, sample_ai_verdict_fraud, [])
        assert len(fetch_recent_logs(limit=10)) == 1

        clear_logs()
        assert len(fetch_recent_logs(limit=10)) == 0

    def test_clear_on_empty_database(self):
        """Should not fail when clearing empty database."""
        clear_logs()  # Should not raise


class TestGetDBStatus:
    """Tests for database health check."""

    def test_healthy_status(self):
        """Should return healthy status."""
        status = get_db_status()
        assert status["status"] == "healthy"
        assert "total_rows" in status

    def test_row_count_accurate(self, sample_clean_data, sample_ai_verdict_fraud):
        """Row count should be accurate."""
        log_transaction(sample_clean_data, sample_ai_verdict_fraud, [])
        status = get_db_status()
        assert status["total_rows"] == 1
