"""
rules_engine.py — Configurable, rule-based fraud-indicator engine.

Applies static thresholds, list matching, and behavioral analysis on
transaction DataFrames to generate a per-row checklist of suspicious
indicators and a composite risk score (0-100).
"""

import json
import logging
import os
import pandas as pd
from typing import Any, Dict, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "fraud_typology.json")


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load fraud-typology configuration from JSON file with defaults."""
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info("Loaded config from %s", config_path)
        return config
    
    # Fallback config if file not found
    return {
        "thresholds": {
            "high_amount": 10000.0,
            "high_velocity_count": 5,
            "impossible_travel_speed_kmh": 1000.0
        },
        "lists": {
            "blacklisted_merchants": ["DARKWEBSTORE", "DARKWEB*PAY", "SHADYELECTRONICS", "SHADY*ELECTRONICS"],
            "suspicious_devices": ["emulator", "rooted_android", "unknown_browser"]
        },
        "behavioral_rules": {
            "check_new_device": True,
            "check_location_change": True,
            "amount_anomaly_multiplier": 3.0
        }
    }


def _safe_get(row: pd.Series, col: str, default: Any = None) -> Any:
    """Return row[col] if available and not NaN."""
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


def build_user_norms(historical_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Precompute behavioral norms for all clients for fast live scoring."""
    user_norms: Dict[str, Dict[str, Any]] = {}
    if historical_df is not None and not historical_df.empty:
        required_cols = ["client_id", "amount", "use_chip", "merchant_city"]
        for col in required_cols:
            if col not in historical_df.columns:
                historical_df[col] = "unknown" if col != "amount" else 0.0

        df = historical_df[["client_id", "amount", "use_chip", "merchant_city"]].copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["use_chip"] = df["use_chip"].fillna("unknown").astype(str).str.lower()
        df["merchant_city"] = df["merchant_city"].fillna("unknown").astype(str).str.lower()

        agg_res = df.groupby("client_id")["amount"].agg(["sum", "count", "mean"])
        methods_map = df.groupby("client_id")["use_chip"].apply(set).to_dict()
        locations_map = df.groupby("client_id")["merchant_city"].apply(set).to_dict()

        for cid, stats in agg_res.iterrows():
            cid_str = str(cid)
            user_norms[cid_str] = {
                "known_methods": set(methods_map.get(cid, set())),
                "known_locations": set(locations_map.get(cid, set())),
                "sum_amount": float(stats["sum"]),
                "count_amount": int(stats["count"]),
                "avg_amount": float(stats["mean"]),
            }
    return user_norms


def apply_rules(
    df: pd.DataFrame,
    config: Dict[str, Any],
    historical_df: Optional[pd.DataFrame] = None,
    precomputed_norms: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Apply rules engine and append checklist + risk score."""
    if df.empty:
        df["suspicious_indicators_checklist"] = []
        df["rule_risk_score"] = 0
        return df

    df = df.copy()

    thresholds = config.get("thresholds", {})
    lists = config.get("lists", {})
    behavioral = config.get("behavioral_rules", {})

    user_norms: Dict[str, Dict[str, Any]]
    if precomputed_norms is not None:
        user_norms = precomputed_norms
    else:
        user_norms = build_user_norms(historical_df)

    has_date = "date" in df.columns
    if has_date:
        df = df.sort_values(by="date")

    def _check_row(row: pd.Series) -> List[str]:
        indicators: List[str] = []
        raw_client_id = _safe_get(row, "client_id", "UNKNOWN")
        client_id_str = str(raw_client_id)

        if client_id_str not in user_norms:
            user_norms[client_id_str] = {
                "known_methods": set(),
                "known_locations": set(),
                "sum_amount": 0.0,
                "count_amount": 0,
                "avg_amount": 0.0,
            }

        norms: Dict[str, Any] = user_norms[client_id_str]

        # 1. Static Thresholds
        amt = float(_safe_get(row, "amount", 0.0) or 0.0)
        high_amt_threshold = float(thresholds.get("high_amount", 10000.0))
        if amt > high_amt_threshold:
            indicators.append(f"High Transaction Amount (${amt:,.2f} > ${high_amt_threshold:,.2f})")

        velocity = _safe_get(row, "velocity_1h")
        if velocity is not None:
            try:
                vel_num = float(velocity)
                high_vel_thresh = float(thresholds.get("high_velocity_count", 5))
                if vel_num > high_vel_thresh:
                    indicators.append(f"High Transaction Velocity ({int(vel_num)} tx/hr)")
            except (ValueError, TypeError):
                pass

        travel_speed = _safe_get(row, "travel_speed_kmh")
        if travel_speed is not None:
            try:
                speed_num = float(travel_speed)
                max_speed_thresh = float(thresholds.get("impossible_travel_speed_kmh", 1000.0))
                if speed_num > max_speed_thresh:
                    indicators.append(f"Impossible Travel Speed ({speed_num:.1f} km/h)")
            except (ValueError, TypeError):
                pass

        # 2. List Matching
        merch = str(_safe_get(row, "merchant_id", "")).upper()
        blacklisted = [str(m).upper() for m in lists.get("blacklisted_merchants", [])]
        if merch and any(b in merch for b in blacklisted):
            indicators.append(f"Blacklisted Merchant ({merch})")

        method_val = str(_safe_get(row, "use_chip", "")).lower()
        susp_devices = [str(d).lower() for d in lists.get("suspicious_devices", [])]
        if method_val and method_val in susp_devices:
            indicators.append(f"Suspicious Device/Method ({method_val})")

        # 3. Behavioral Rules
        if behavioral.get("check_new_device", True):
            if not method_val or method_val in ("nan", "unknown", ""):
                indicators.append("New Device: PARTIAL (Missing Data)")
            elif not norms["known_methods"]:
                indicators.append("New Device: PARTIAL (No Prior History)")
            elif method_val not in norms["known_methods"]:
                indicators.append("New Device: PRESENT")
            else:
                indicators.append("New Device: MISSING")

        loc_val = str(_safe_get(row, "merchant_city", "")).lower()
        if behavioral.get("check_location_change", True):
            if not loc_val or loc_val in ("nan", "unknown", ""):
                indicators.append("Location Change: PARTIAL (Missing Data)")
            elif not norms["known_locations"]:
                indicators.append("Location Change: PARTIAL (No Prior History)")
            elif loc_val not in norms["known_locations"]:
                indicators.append("Location Change: PRESENT")
            else:
                indicators.append("Location Change: MISSING")

        amt_multiplier = float(behavioral.get("amount_anomaly_multiplier", 3.0))
        if amt > 0 and norms["avg_amount"] > 0:
            if amt > norms["avg_amount"] * amt_multiplier:
                indicators.append(f"Amount Anomaly: PRESENT (${amt:.2f} vs Avg ${norms['avg_amount']:.2f})")

        err = str(_safe_get(row, "errors", "")).strip()
        if err and err.lower() not in ("", "nan", "none"):
            indicators.append(f"Transaction Error: {err}")

        # Update running norms for client
        if method_val and method_val not in ("nan", "unknown", ""):
            norms["known_methods"].add(method_val)
        if loc_val and loc_val not in ("nan", "unknown", ""):
            norms["known_locations"].add(loc_val)
        if amt > 0:
            norms["sum_amount"] += amt
            norms["count_amount"] += 1
            norms["avg_amount"] = norms["sum_amount"] / norms["count_amount"]

        return indicators

    df["suspicious_indicators_checklist"] = df.apply(_check_row, axis=1)

    def _calc_risk(indicators: List[str]) -> int:
        score = 0
        for ind in indicators:
            if any(kw in ind for kw in ("PRESENT", "High", "Impossible", "Blacklisted", "Suspicious", "Error", "Anomaly")):
                score += 25
            elif "PARTIAL" in ind:
                score += 10
        return min(score, 100)

    df["rule_risk_score"] = df["suspicious_indicators_checklist"].apply(_calc_risk)

    if has_date:
        df = df.sort_index()

    logger.info("Rules applied — %d rows scored.", len(df))
    return df
