"""
test_api.py — Integration tests for the FastAPI REST API endpoints.
"""

import pytest
from unittest.mock import patch


class TestHealthEndpoint:
    """Tests for /health endpoint (public, no auth required)."""

    def test_health_returns_200(self, api_client):
        """Health endpoint should return 200."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, api_client):
        """Health response should contain required fields."""
        response = api_client.get("/health")
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests for / root endpoint."""

    def test_root_returns_200(self, api_client):
        """Root should return 200."""
        response = api_client.get("/")
        assert response.status_code == 200


class TestAnalyzeEndpoint:
    """Tests for /analyze and /v1/analyze endpoints."""

    def test_analyze_valid_transaction(self, api_client):
        """Should successfully analyze a valid transaction."""
        payload = {
            "transaction_id": "TXN-API-001",
            "client_id": "USER-105",
            "amount": 3500.0,
            "date": "2026-03-22T10:30:00Z",
            "merchant_id": "DARKWEB*PAY",
            "use_chip": "Online",
            "merchant_city": "Unknown",
            "velocity_1h": 8,
            "travel_speed_kmh": 1400.0,
            "ip_address": "185.220.101.5"
        }
        response = api_client.post("/analyze", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "Output_A_Summary" in data
        assert "Output_B_Verdict" in data
        assert "Output_C_Checklist" in data

    def test_analyze_v1_route(self, api_client):
        """Should work via /v1/analyze route."""
        payload = {
            "transaction_id": "TXN-V1-001",
            "client_id": "USER-100",
            "amount": 100.0,
            "date": "2026-03-22T10:00:00Z",
            "merchant_id": "AMAZON",
        }
        response = api_client.post("/v1/analyze", json=payload)
        assert response.status_code == 200

    def test_analyze_missing_required_fields(self, api_client):
        """Should return 422 for missing required fields."""
        payload = {"amount": 100.0}  # Missing transaction_id, client_id, date, merchant_id
        response = api_client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_negative_amount_rejected(self, api_client):
        """Should reject negative amounts."""
        payload = {
            "transaction_id": "TXN-NEG",
            "client_id": "USER-100",
            "amount": -500.0,
            "date": "2026-03-22T10:00:00Z",
            "merchant_id": "AMAZON",
        }
        response = api_client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_analyze_returns_risk_score(self, api_client):
        """Response should contain a numeric risk score."""
        payload = {
            "transaction_id": "TXN-SCORE",
            "client_id": "USER-100",
            "amount": 50.0,
            "date": "2026-03-22T10:00:00Z",
            "merchant_id": "AMAZON",
        }
        response = api_client.post("/analyze", json=payload)
        data = response.json()
        verdict = data["Output_B_Verdict"]
        assert isinstance(verdict["risk_score"], (int, float))

    def test_request_id_header(self, api_client):
        """Response should include X-Request-ID header."""
        payload = {
            "transaction_id": "TXN-RID",
            "client_id": "USER-100",
            "amount": 50.0,
            "date": "2026-03-22T10:00:00Z",
            "merchant_id": "AMAZON",
        }
        response = api_client.post("/analyze", json=payload)
        assert "x-request-id" in response.headers


class TestAnalyzeBatchEndpoint:
    """Tests for /analyze_batch endpoint."""

    def test_batch_empty_list(self, api_client):
        """Should handle empty batch."""
        response = api_client.post("/analyze_batch", json=[])
        assert response.status_code == 200
        assert response.json()["batch_results"] == []

    def test_batch_multiple_transactions(self, api_client):
        """Should process multiple transactions."""
        tx_list = [
            {
                "transaction_id": f"TXN-BATCH-{i}",
                "client_id": "USER-100",
                "amount": 100.0 + i,
                "date": "2026-03-22T10:00:00Z",
                "merchant_id": "AMAZON",
            }
            for i in range(3)
        ]
        response = api_client.post("/analyze_batch", json=tx_list)
        assert response.status_code == 200
        assert len(response.json()["batch_results"]) == 3

    def test_batch_size_limit(self, api_client):
        """Should reject batches exceeding MAX_BATCH_SIZE."""
        tx_list = [
            {
                "transaction_id": f"TXN-OVER-{i}",
                "client_id": "USER-100",
                "amount": 10.0,
                "date": "2026-03-22T10:00:00Z",
                "merchant_id": "AMAZON",
            }
            for i in range(101)
        ]
        response = api_client.post("/analyze_batch", json=tx_list)
        assert response.status_code == 400


class TestAlertsEndpoint:
    """Tests for /alerts endpoint."""

    def test_get_alerts_default(self, api_client):
        """Should return alerts list."""
        response = api_client.get("/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "alerts_count" in data

    def test_get_alerts_with_status_filter(self, api_client):
        """Should filter by status parameter."""
        response = api_client.get("/alerts?status=CONFIRMED_FRAUD")
        assert response.status_code == 200


class TestAnalyticsEndpoint:
    """Tests for /analytics endpoint."""

    def test_analytics_empty_db(self, api_client):
        """Should return zero metrics on empty DB."""
        response = api_client.get("/analytics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_transactions"] == 0

    def test_analytics_after_transactions(self, api_client):
        """Should reflect logged transactions."""
        # First, analyze a transaction to populate DB
        payload = {
            "transaction_id": "TXN-ANALYTICS",
            "client_id": "USER-100",
            "amount": 5000.0,
            "date": "2026-03-22T10:00:00Z",
            "merchant_id": "DARKWEB*PAY",
        }
        api_client.post("/analyze", json=payload)

        response = api_client.get("/analytics")
        data = response.json()
        assert data["total_transactions"] >= 1


class TestClearLogsEndpoint:
    """Tests for /logs/clear endpoint."""

    def test_clear_logs(self, api_client):
        """Should clear all logs."""
        response = api_client.post("/logs/clear")
        assert response.status_code == 200
        assert "cleared" in response.json()["message"].lower() or "clear" in response.json()["message"].lower()


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_data(self, api_client):
        """Should return operational metrics."""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "finguard-ai"
        assert "version" in data
