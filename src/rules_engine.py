"""
rules_engine.py — Configurable, rule-based fraud-indicator engine.

Applies static thresholds, list matching, and behavioral analysis on
transaction DataFrames to generate a per-row checklist of suspicious
indicators and a composite risk score (0-100).
"""

import json
import logging
import pandas as pd
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Config loader ───────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/fraud_typology.json") -> Dict[str, Any]:
    """Load the fraud-typology configuration from a JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info("Loaded config from %s", config_path)
    return config


# ── Helper: safe column accessor ───────────────────────────────────────────

def _safe_get(row: pd.Series, col: str, default=None):
    """Return row[col] if the column exists and is not NaN, else *default*."""
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


# ── Main engine ─────────────────────────────────────────────────────────────

def build_user_norms(historical_df: pd.DataFrame) -> Dict[str, Any]:
    """Precompute behavioural norms for all clients to avoid costly recalculation during live streaming."""
    user_norms: Dict[str, Any] = {}
    if historical_df is not None and not historical_df.empty:
        # 1. Clean data once
        df = historical_df[["client_id", "amount", "use_chip", "merchant_city"]].copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["use_chip"] = df["use_chip"].fillna("unknown").astype(str).str.lower()
        df["merchant_city"] = df["merchant_city"].fillna("unknown").astype(str).str.lower()

        # 2. Fast numeric aggregation
        agg_res = df.groupby("client_id")["amount"].agg(["sum", "count", "mean"])

        # 3. Fast set building (vectorized groupby unique is fast)
        # We manually build the dictionaries to avoid the O(N^2) overhead of multiple applies
        methods_map = df.groupby("client_id")["use_chip"].apply(set).to_dict()
        locations_map = df.groupby("client_id")["merchant_city"].apply(set).to_dict()

        # 4. Final assembly (fast)
        for cid, stats in agg_res.iterrows():
            cid_str = str(cid)
            user_norms[cid_str] = {
                "known_methods": methods_map.get(cid, set()),
                "known_locations": locations_map.get(cid, set()),
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
    """
    Apply fraud-typology rules and return the DataFrame enriched with:
      • suspicious_indicators_checklist  (list[str])
      • rule_risk_score                  (int, 0-100)

    Parameters
    ----------
    df : pd.DataFrame
        The batch of transactions to score.
    config : dict
        Fraud-typology config (thresholds, lists, behavioral_rules).
    historical_df : pd.DataFrame, optional
        Past transactions used to build per-user behavioural norms on the fly.
    precomputed_norms : dict, optional
        Pass precomputed norms dictionary (from build_user_norms) directly for extreme performance.
    """
    df = df.copy()

    thresholds = config.get("thresholds", {})
    lists = config.get("lists", {})
    behavioral = config.get("behavioral_rules", {})

    # ── 1. Load or Build per-user norms ────────────────────────
    if precomputed_norms is not None:
        user_norms = precomputed_norms
    else:
        user_norms = build_user_norms(historical_df)

    # ── 2. Sort chronologically (if date column exists) ─────────────────────
    has_date = "date" in df.columns
    if has_date:
        df = df.sort_values(by="date")

    # ── 3. Row-level rule evaluation ────────────────────────────────────────
    def _check_row(row: pd.Series) -> list:
        indicators: list[str] = []
        client_id = _safe_get(row, "client_id")

        # Initialise norms for first-time users
        if str(client_id) not in user_norms:
            user_norms[str(client_id)] = {
                "known_methods": set(),
                "known_locations": set(),
                "sum_amount": 0.0,
                "count_amount": 0,
                "avg_amount": 0.0,
            }
        
        # We explicitly cast to Any here to un-confuse strict static type-checkers on dictionary values
        # Pyre thinks this might be a set interacting with floats without this.
        norms: Dict[str, Any] = user_norms[str(client_id)]

        # ── 3a. Static thresholds ───────────────────────────────────────────
        amt = _safe_get(row, "amount", 0.0)
        if isinstance(amt, (int, float)):
            if amt > thresholds.get("high_amount", 10_000.0):
                indicators.append("High Transaction Amount")

        velocity = _safe_get(row, "velocity_1h")
        if velocity is not None and isinstance(velocity, (int, float)):
            if velocity > thresholds.get("high_velocity_count", 5):
                indicators.append("High Transaction Velocity")

        travel_speed = _safe_get(row, "travel_speed_kmh")
        if travel_speed is not None and isinstance(travel_speed, (int, float)):
            if travel_speed > thresholds.get("impossible_travel_speed_kmh", 1000.0):
                indicators.append("Impossible Travel Speed")

        # ── 3b. List matching ───────────────────────────────────────────────
        merch = str(_safe_get(row, "merchant_id", "")).lower()
        if merch and merch in [m.lower() for m in lists.get("blacklisted_merchants", [])]:
            indicators.append("Blacklisted Merchant")

        method_val = str(_safe_get(row, "use_chip", "")).lower()
        if method_val and method_val in [d.lower() for d in lists.get("suspicious_devices", [])]:
            indicators.append("Suspicious Device")

        # ── 3c. Behavioural rules (vs. historical norms) ────────────────────
        # New payment method / device check
        if behavioral.get("check_new_device", True):
            if not method_val or method_val in ("nan", ""):
                indicators.append("New Device: PARTIAL (Missing Data)")
            elif not norms["known_methods"]:
                indicators.append("New Device: PARTIAL (No Prior History)")
            elif method_val not in norms["known_methods"]:
                indicators.append("New Device: PRESENT")
            else:
                indicators.append("New Device: MISSING")

        # Location change check
        loc_val = str(_safe_get(row, "merchant_city", "")).lower()
        if behavioral.get("check_location_change", True):
            if not loc_val or loc_val in ("nan", ""):
                indicators.append("Location Change: PARTIAL (Missing Data)")
            elif not norms["known_locations"]:
                indicators.append("Location Change: PARTIAL (No Prior History)")
            elif loc_val not in norms["known_locations"]:
                indicators.append("Location Change: PRESENT")
            else:
                indicators.append("Location Change: MISSING")

        # Amount anomaly check
        amt_multiplier = behavioral.get("amount_anomaly_multiplier", 3.0)
        if isinstance(amt, (int, float)) and norms["avg_amount"] > 0:
            if amt > norms["avg_amount"] * amt_multiplier:
                indicators.append("Amount Anomaly: PRESENT")

        # ── 3d. Error-field check (bonus) ───────────────────────────────────
        err = str(_safe_get(row, "errors", "")).strip()
        if err and err.lower() not in ("", "nan"):
            indicators.append(f"Transaction Error: {err}")

        # ── 3e. Update running norms for subsequent rows ────────────────────
        if method_val and method_val not in ("nan", ""):
            norms["known_methods"].add(method_val)
        if loc_val and loc_val not in ("nan", ""):
            norms["known_locations"].add(loc_val)
        if isinstance(amt, (int, float)):
            norms["sum_amount"] += amt
            norms["count_amount"] += 1
            norms["avg_amount"] = norms["sum_amount"] / norms["count_amount"]

        return indicators

    df["suspicious_indicators_checklist"] = df.apply(_check_row, axis=1)

    # ── 4. Risk score (0-100) ───────────────────────────────────────────────
    def _calc_risk(indicators: list) -> int:
        score = 0
        for ind in indicators:
            if any(
                kw in ind
                for kw in ("PRESENT", "High", "Impossible", "Blacklisted", "Suspicious", "Error")
            ):
                score += 25
            elif "PARTIAL" in ind:
                score += 10
        return min(score, 100)

    df["rule_risk_score"] = df["suspicious_indicators_checklist"].apply(_calc_risk)

    # ── 5. Restore original ordering ────────────────────────────────────────
    if has_date:
        df = df.sort_index()

    logger.info(
        "Rules applied — %d rows scored.  Risk distribution:\n%s",
        len(df),
        df["rule_risk_score"].describe().to_string(),
    )
    return df
