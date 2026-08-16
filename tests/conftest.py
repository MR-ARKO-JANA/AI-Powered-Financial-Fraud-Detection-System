"""
conftest.py — Shared pytest fixtures for FinGuard AI test suite.
"""

import os
import sys
import pytest
import sqlite3
import tempfile
import shutil

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    """
    Redirect all SQLite operations to a temporary database for test isolation.
    Automatically cleans up after each test.
    """
    temp_db = str(tmp_path / "test_transactions.db")

    # Patch database module
    monkeypatch.setattr("src.database.DB_PATH", temp_db)
    monkeypatch.setattr("src.alert_system.DB_PATH", temp_db)

    # Initialize fresh schema
    from src.database import init_db
    from src.alert_system import init_alerts_db
    init_db()
    init_alerts_db()

    yield temp_db


@pytest.fixture
def sample_clean_data():
    """Sample cleaned transaction dictionary for testing."""
    return {
        "transaction_id": "TXN-TEST-001",
        "client_id": "USER-100",
        "amount": 3500.0,
        "date": "2026-03-22T10:30:00Z",
        "merchant_id": "DARKWEB*PAY",
        "use_chip": "Online",
        "merchant_city": "Unknown",
        "velocity_1h": 8,
        "travel_speed_kmh": 1400.0,
        "ip_address": "185.220.101.5",
        "ip_location": "External IP",
    }


@pytest.fixture
def sample_safe_data():
    """Sample safe/legitimate transaction for testing."""
    return {
        "transaction_id": "TXN-SAFE-001",
        "client_id": "USER-101",
        "amount": 25.50,
        "date": "2026-03-22T14:00:00Z",
        "merchant_id": "AMAZON",
        "use_chip": "Chip",
        "merchant_city": "New York",
        "velocity_1h": 1,
        "travel_speed_kmh": 0.0,
        "ip_address": "192.168.1.1",
        "ip_location": "Local Corporate Network",
    }


@pytest.fixture
def sample_ai_verdict_fraud():
    """Sample AI verdict for a fraudulent transaction."""
    return {
        "fraud_type": "Account Takeover",
        "risk_score": 85.0,
        "reason": "High risk flagged by ML model with suspicious behavioral indicators."
    }


@pytest.fixture
def sample_ai_verdict_safe():
    """Sample AI verdict for a safe transaction."""
    return {
        "fraud_type": "Safe/Legitimate",
        "risk_score": 12.0,
        "reason": "Low risk with standard user behavior."
    }


@pytest.fixture
def sample_checklist_suspicious():
    """Suspicious indicators checklist."""
    return [
        "Blacklisted Merchant (DARKWEB*PAY)",
        "High Transaction Velocity (8 tx/hr)",
        "Impossible Travel Speed (1400.0 km/h)",
        "New Device: PARTIAL (No Prior History)",
        "Location Change: PARTIAL (Missing Data)",
    ]


@pytest.fixture
def sample_checklist_clean():
    """Clean indicators checklist."""
    return [
        "New Device: MISSING",
        "Location Change: MISSING",
    ]


@pytest.fixture
def test_config():
    """Test fraud typology configuration."""
    return {
        "thresholds": {
            "high_amount": 10000.0,
            "high_velocity_count": 5,
            "impossible_travel_speed_kmh": 1000.0
        },
        "lists": {
            "blacklisted_merchants": ["DARKWEB*PAY", "SHADY*ELECTRONICS"],
            "suspicious_devices": ["emulator", "rooted_android"]
        },
        "behavioral_rules": {
            "check_new_device": True,
            "check_location_change": True,
            "amount_anomaly_multiplier": 3.0
        }
    }


@pytest.fixture
def api_client():
    """FastAPI TestClient with authentication headers."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    return client
