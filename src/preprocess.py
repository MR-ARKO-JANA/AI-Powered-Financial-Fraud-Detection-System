"""
preprocess.py — Data cleaning and preprocessing pipeline for transaction data.

Reads raw transactions CSV, cleans and normalizes fields, and outputs a
clean CSV ready for feature engineering and rules-based analysis.
"""

import os
import pandas as pd
import logging
import requests
import time
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_data.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "clean_transactions.csv")

# In-memory IP Geolocation Cache to avoid repeated HTTP calls
_IP_CACHE: Dict[str, str] = {}


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw transaction data into a DataFrame."""
    if not os.path.isfile(path):
        logger.warning(f"Raw data file not found at {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    logger.info("Loading raw data from %s ...", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def _resolve_ip_location(ip_str: str) -> str:
    """Resolve raw IP to Location string with caching & fallback."""
    ip_clean = str(ip_str).strip()
    if not ip_clean or ip_clean.lower() in ("unknown", "nan", "none", "0.0.0.0"):
        return "Unknown Location"
    if ip_clean.startswith(("192.168.", "10.", "172.16.", "127.0.")):
        return "Local Corporate Network"
    if ip_clean in _IP_CACHE:
        return _IP_CACHE[ip_clean]

    try:
        res = requests.get(f"http://ip-api.com/json/{ip_clean}", timeout=1.5)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                city = data.get("city", "Unknown")
                country = data.get("countryCode", "Unknown")
                loc_str = f"{city}, {country}".strip(", ")
                _IP_CACHE[ip_clean] = loc_str
                return loc_str
    except Exception:
        pass

    fallback_loc = "External IP"
    _IP_CACHE[ip_clean] = fallback_loc
    return fallback_loc


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data cleaning, type conversion, standardization, and normalization."""
    if df.empty:
        return df

    df = df.copy()

    # 1. Standardize Amount
    if "amount" in df.columns:
        # Convert to string, strip '$' and ',', then parse numeric safely
        clean_amt_str = (
            df["amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        parsed_amounts = pd.to_numeric(clean_amt_str, errors="coerce")
        df["amount"] = parsed_amounts.fillna(0.0)

    # 2. Merchant ID normalization & Alias merging
    if "merchant_id" in df.columns:
        df["merchant_id"] = (
            df["merchant_id"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        merchant_aliases = {
            "AMZN*MKTP": "AMAZON",
            "AMZN": "AMAZON",
            "FLIPKART*INDIA": "FLIPKART",
            "FLIPKART*": "FLIPKART",
            "DARKWEBSTORE": "DARKWEB*PAY"
        }
        df["merchant_id"] = df["merchant_id"].replace(merchant_aliases)

    # 3. Merchant City
    if "merchant_city" in df.columns:
        df["merchant_city"] = (
            df["merchant_city"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.title()
        )

    # 4. Merchant State
    if "merchant_state" in df.columns:
        df["merchant_state"] = (
            df["merchant_state"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # 5. Chip usage
    if "use_chip" in df.columns:
        df["use_chip"] = (
            df["use_chip"]
            .fillna("online")
            .astype(str)
            .str.strip()
            .str.capitalize()
        )

    # 6. Errors field
    if "errors" in df.columns:
        df["errors"] = df["errors"].fillna("").astype(str).str.strip()

    # 7. Zip code formatting
    if "zip" in df.columns:
        df["zip"] = (
            df["zip"]
            .fillna("00000")
            .astype(str)
            .str.split(".")
            .str[0]
            .str.zfill(5)
        )

    # 8. IP Geolocation mapping
    if "ip_address" in df.columns:
        df["ip_location"] = df["ip_address"].astype(str).apply(_resolve_ip_location)

    # 9. Date parsing
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").astype(str)

    # 10. Deduplication
    before = len(df)
    df = df.drop_duplicates()
    dupes = before - len(df)
    if dupes > 0:
        logger.info("Dropped %d duplicate rows.", dupes)

    return df


def save_clean_data(df: pd.DataFrame, path: str = CLEAN_DATA_PATH) -> None:
    """Save cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved clean data (%d rows) -> %s", len(df), path)


def preprocess(raw_path: str = RAW_DATA_PATH, clean_path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Full pipeline: load -> clean -> save -> return DataFrame."""
    df = load_raw_data(raw_path)
    if not df.empty:
        df = clean_data(df)
        save_clean_data(df, clean_path)
    return df


if __name__ == "__main__":
    preprocess()