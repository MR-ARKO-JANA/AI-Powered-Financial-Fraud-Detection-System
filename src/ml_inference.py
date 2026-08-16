import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from dotenv import load_dotenv
from src.feature_engineering import prepare_feature_dataframe, get_trained_pipeline
from src.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load Local Fine-Tuned Transformer & RF Ensemble Pipeline
ml_pipeline = get_trained_pipeline()


def _clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Clean markdown code fences from AI response string and parse JSON."""
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        elif clean.endswith("```json"):
            clean = clean.rsplit("```json", 1)[0]
    clean = clean.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def get_ai_risk_score(clean_data: Dict[str, Any], checklist: List[str]) -> Dict[str, Any]:
    """
    Combines rule checklist, fine-tuned Transformer & Random Forest Ensemble model prediction,
    and Gemini GenAI reasoning.
    """
    # 1. Rules Risk Score Calculation
    suspicious_count = 0
    for item in checklist:
        if any(kw in item for kw in ["PRESENT", "PARTIAL", "High", "Impossible", "Blacklisted", "Suspicious", "Anomaly", "Error"]):
            suspicious_count += 1

    total_indicators = len(checklist) if len(checklist) > 0 else 1
    rules_risk_score = (suspicious_count / total_indicators) * 100.0

    # 2. Local Fine-Tuned Transformer & RF Ensemble Model Inference
    ml_fraud_probability = 0.0
    active_pipeline = ml_pipeline or get_trained_pipeline()

    if active_pipeline is not None:
        try:
            df_feat = prepare_feature_dataframe(clean_data)
            # Predict probability using full preprocessed column transformer pipeline
            probs = active_pipeline.predict_proba(df_feat)[0]
            ml_fraud_probability = float(probs[1]) * 100.0
        except Exception as e:
            # Fallback to feature matrix if scalar input
            try:
                amt = float(clean_data.get("amount", 0.0))
                vel = int(clean_data.get("velocity_1h", 1))
                speed = float(clean_data.get("travel_speed_kmh", 0.0))
                hour = 12
                probs = active_pipeline.predict_proba([[amt, vel, speed, hour]])[0]
                ml_fraud_probability = float(probs[1]) * 100.0
            except Exception as err:
                logger.warning("ML inference fallback failed: %s | %s", e, err, exc_info=True)

    # Composite Risk Score Calculation
    composite_base_score = round((rules_risk_score * 0.35) + (ml_fraud_probability * 0.65), 2)

    # Deterministic Classification Fallback
    if composite_base_score >= 70:
        default_type = "Account Takeover" if any("Device" in c or "Speed" in c for c in checklist) else "Card Skimming"
        default_reason = f"Fine-tuned Transformer & RF Ensemble flagged high risk ({composite_base_score:.1f}/100) across preprocessed behavioral features."
    elif composite_base_score >= 40:
        default_type = "Unknown Suspicious Activity"
        default_reason = f"Moderate risk ({composite_base_score:.1f}/100) requires verification of transaction details."
    else:
        default_type = "Safe/Legitimate"
        default_reason = f"Low risk score ({composite_base_score:.1f}/100) with standard user behavior."

    # 3. Call Google Gemini GenAI if API Key is configured
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            prompt = f"""
You are an expert financial fraud investigation AI.
Review this financial transaction payload:
{json.dumps(clean_data, indent=2)}

Flagged rule indicators:
{json.dumps(checklist, indent=2)}

Model Risk Metrics:
- Rules Engine Risk Score: {rules_risk_score:.1f}/100
- Fine-Tuned Transformer & RF Ensemble Probability: {ml_fraud_probability:.1f}%
- Composite Risk Score: {composite_base_score:.1f}/100

Classify this transaction into EXACTLY ONE of the following categories:
- Account Takeover
- Card Skimming
- Safe/Legitimate
- Unknown Suspicious Activity

Return ONLY a valid JSON object matching this schema:
{{
  "fraud_type": "Classification Category",
  "risk_score": {composite_base_score:.1f},
  "reason": "Clear concise analytical sentence explaining the decision."
}}
"""

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if response and response.text:
                ai_verdict = _clean_json_response(response.text)
                ai_verdict["risk_score"] = float(ai_verdict.get("risk_score", composite_base_score))
                logger.info("Gemini AI verdict received for transaction.")
                return ai_verdict

        except Exception as err:
            logger.warning("Gemini AI call failed (using local hybrid fallback): %s", err)

    # Offline / Fallback Return
    return {
        "fraud_type": default_type,
        "risk_score": float(composite_base_score),
        "reason": default_reason
    }
