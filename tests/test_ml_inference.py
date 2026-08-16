"""
test_ml_inference.py — Tests for ML inference and AI risk scoring.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from src.ml_inference import get_ai_risk_score, _clean_json_response


class TestCleanJsonResponse:
    """Tests for JSON response cleaning from Gemini API."""

    def test_clean_json(self):
        """Should parse clean JSON."""
        result = _clean_json_response('{"fraud_type": "Safe", "risk_score": 10.0, "reason": "test"}')
        assert result["fraud_type"] == "Safe"

    def test_json_with_code_fence(self):
        """Should strip markdown code fences."""
        raw = '```json\n{"fraud_type": "Safe", "risk_score": 10.0, "reason": "test"}\n```'
        result = _clean_json_response(raw)
        assert result["fraud_type"] == "Safe"

    def test_json_with_generic_fence(self):
        """Should strip generic code fences."""
        raw = '```\n{"fraud_type": "Fraud", "risk_score": 80.0, "reason": "test"}\n```'
        result = _clean_json_response(raw)
        assert result["fraud_type"] == "Fraud"

    def test_invalid_json_raises(self):
        """Should raise on invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            _clean_json_response("not valid json at all")


class TestGetAIRiskScore:
    """Tests for the AI risk scoring function."""

    def test_returns_dict(self, sample_clean_data, sample_checklist_suspicious):
        """Should always return a dict with required keys."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            result = get_ai_risk_score(sample_clean_data, sample_checklist_suspicious)
            assert isinstance(result, dict)
            assert "fraud_type" in result
            assert "risk_score" in result
            assert "reason" in result

    def test_risk_score_is_float(self, sample_clean_data, sample_checklist_suspicious):
        """Risk score should be a float."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            result = get_ai_risk_score(sample_clean_data, sample_checklist_suspicious)
            assert isinstance(result["risk_score"], float)

    def test_high_risk_classification(self, sample_clean_data, sample_checklist_suspicious):
        """Suspicious transactions should get high-risk classification."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            result = get_ai_risk_score(sample_clean_data, sample_checklist_suspicious)
            assert result["risk_score"] > 30.0  # Should be flagged

    def test_safe_transaction_low_risk(self, sample_safe_data, sample_checklist_clean):
        """Safe transactions should get low risk scores."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            result = get_ai_risk_score(sample_safe_data, sample_checklist_clean)
            assert result["risk_score"] < 50.0

    def test_offline_mode_works(self, sample_clean_data, sample_checklist_suspicious):
        """Should work without Gemini API key (offline fallback)."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            result = get_ai_risk_score(sample_clean_data, sample_checklist_suspicious)
            assert result["fraud_type"] in [
                "Account Takeover", "Card Skimming",
                "Safe/Legitimate", "Unknown Suspicious Activity"
            ]

    def test_empty_checklist(self, sample_clean_data):
        """Should handle empty checklist gracefully."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            result = get_ai_risk_score(sample_clean_data, [])
            assert isinstance(result, dict)
            assert "risk_score" in result

    def test_gemini_failure_falls_back(self, sample_clean_data, sample_checklist_suspicious):
        """Should fall back to local model when Gemini API call fails."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}, clear=False):
            # Gemini will fail with a fake key, triggering fallback
            result = get_ai_risk_score(sample_clean_data, sample_checklist_suspicious)
            assert isinstance(result, dict)
            assert "risk_score" in result
            assert "fraud_type" in result
