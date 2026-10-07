"""
alert_system.py — Production-Grade Alert Trigger & Model Refinement Engine for FinGuard AI.

1. Triggers real-time alerts for high-risk, anomalous, or ambiguous transactions.
2. Manages an active alert queue in SQLite with context-managed connections.
3. Provides an Active Learning Feedback Loop to incrementally refine and retrain
   the Transformer & Random Forest Hybrid Ensemble model.
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

from src.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "transactions_log.db")


@contextmanager
def get_connection():
    """Context-managed SQLite connection with guaranteed cleanup."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_alerts_db():
    """Initialize model_alerts and feedback tables if not present."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE,
                transaction_id TEXT,
                client_id TEXT,
                amount REAL,
                fraud_type TEXT,
                risk_score REAL,
                severity TEXT,
                reason TEXT,
                status TEXT DEFAULT 'PENDING',
                analyst_feedback TEXT DEFAULT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_feedback_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                client_id TEXT,
                amount REAL,
                velocity_1h INTEGER,
                travel_speed_kmh REAL,
                hour_of_day INTEGER,
                merchant_id TEXT,
                use_chip TEXT,
                merchant_city TEXT,
                ip_location TEXT,
                confirmed_is_fraud INTEGER,
                logged_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        logger.info("Alerts database schema initialized.")


def process_and_trigger_alert(clean_data: Dict[str, Any], ai_verdict: Dict[str, Any], checklist: List[str]) -> Optional[Dict[str, Any]]:
    """
    Evaluates transaction verdict against alert thresholds.
    Triggers an alert if risk_score >= 65.0 or suspicious rules are present.
    """

    risk_score = float(ai_verdict.get("risk_score", 0.0))
    fraud_type = str(ai_verdict.get("fraud_type", "UNKNOWN"))

    # Alert Severity Rules
    if risk_score >= 80.0:
        severity = "CRITICAL"
    elif risk_score >= 60.0:
        severity = "WARNING"
    elif any("Blacklisted" in c or "Speed" in c for c in checklist):
        severity = "WARNING"
    else:
        return None  # Safe transaction, no alert required

    alert_id = f"ALT-{int(datetime.now(timezone.utc).timestamp())}-{str(clean_data.get('transaction_id', 'TXN'))[:8]}"

    alert_payload = {
        "alert_id": alert_id,
        "transaction_id": str(clean_data.get("transaction_id", "")),
        "client_id": str(clean_data.get("client_id", "")),
        "amount": float(clean_data.get("amount", 0.0)),
        "fraud_type": fraud_type,
        "risk_score": risk_score,
        "severity": severity,
        "reason": str(ai_verdict.get("reason", "")),
        "status": "PENDING"
    }

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO model_alerts (
                    alert_id, transaction_id, client_id, amount,
                    fraud_type, risk_score, severity, reason, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_payload["alert_id"],
                alert_payload["transaction_id"],
                alert_payload["client_id"],
                alert_payload["amount"],
                alert_payload["fraud_type"],
                alert_payload["risk_score"],
                alert_payload["severity"],
                alert_payload["reason"],
                alert_payload["status"]
            ))
            conn.commit()
            logger.info("Alert %s triggered: severity=%s, risk=%.1f", alert_id, severity, risk_score)
            return alert_payload

    except Exception as e:
        logger.error("Failed to record model alert: %s", e, exc_info=True)
        return alert_payload


def fetch_active_alerts(status: str = "PENDING") -> List[Dict[str, Any]]:
    """Fetch pending or resolved model alerts."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_alerts WHERE status = ? ORDER BY id DESC", (status,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error("Failed to fetch alerts: %s", e, exc_info=True)
        return []


def submit_analyst_feedback(alert_id: str, confirmed_is_fraud: bool, clean_data: Optional[Dict[str, Any]] = None):
    """
    Submits human analyst feedback (Confirmed Fraud vs False Positive).
    Updates alert status and logs feedback sample for model refinement.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            status_text = "CONFIRMED_FRAUD" if confirmed_is_fraud else "FALSE_POSITIVE"
            cursor.execute(
                "UPDATE model_alerts SET status = ?, analyst_feedback = ? WHERE alert_id = ?",
                (status_text, f"Feedback submitted at {datetime.now(timezone.utc).isoformat()}", alert_id)
            )

            if clean_data:
                hour_of_day = 12
                date_str = clean_data.get("date", "")
                if date_str:
                    try:
                        from dateutil import parser as dp
                        hour_of_day = dp.parse(str(date_str)).hour
                    except Exception:
                        pass
                cursor.execute('''
                    INSERT INTO training_feedback_logs (
                        transaction_id, client_id, amount, velocity_1h, travel_speed_kmh,
                        hour_of_day, merchant_id, use_chip, merchant_city, ip_location, confirmed_is_fraud
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(clean_data.get("transaction_id", "")),
                    str(clean_data.get("client_id", "")),
                    float(clean_data.get("amount", 0.0)),
                    int(clean_data.get("velocity_1h", 1)),
                    float(clean_data.get("travel_speed_kmh", 0.0)),
                    hour_of_day,
                    str(clean_data.get("merchant_id", "UNKNOWN")),
                    str(clean_data.get("use_chip", "Online")),
                    str(clean_data.get("merchant_city", "Unknown")),
                    str(clean_data.get("ip_location", "Unknown Location")),
                    1 if confirmed_is_fraud else 0
                ))

            conn.commit()
            logger.info("Alert %s resolved as: %s", alert_id, status_text)
            return {"status": "success", "alert_id": alert_id, "resolved_as": status_text}

    except Exception as e:
        logger.error("Failed to submit analyst feedback for %s: %s", alert_id, e, exc_info=True)
        return {"status": "error", "message": str(e)}


def refine_and_retrain_model() -> Dict[str, Any]:
    """
    Refines the Transformer & Random Forest Model Pipeline by retraining on both
    the base preprocessed dataset and new verified analyst feedback.
    """
    try:
        from train_transformer_model import train_fine_tuned_transformer_rf_ensemble

        logger.info("Model Refinement Engine: Triggering incremental retraining...")
        train_fine_tuned_transformer_rf_ensemble()

        return {
            "status": "success",
            "message": "Transformer & Random Forest Model successfully refined and re-deployed!",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as err:
        logger.error("Model refinement failed: %s", err, exc_info=True)
        return {"status": "error", "message": f"Refinement failed: {err}"}
