"""
preprocess.py — Data cleaning and preprocessing pipeline for transaction data.

Reads raw transactions CSV, cleans and normalizes fields, and outputs a
clean CSV ready for feature engineering and rules-based analysis.
"""

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Path setup (relative to project root) ──────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_data.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "clean_transactions.csv")


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw transactions CSV and return a DataFrame."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    logger.info("Loading raw data from %s ...", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning / normalization steps and return a clean DataFrame."""
    df = df.copy()

    # ── 1. Handle missing values ────────────────────────────────────────────
    missing = df.isnull().sum()
    has_missing = missing[missing > 0]
    if not has_missing.empty:
        logger.info("Missing values before cleaning:\n%s", has_missing.to_string())
    else:
        logger.info("No missing values found.")

    # Amount: strip '$' and convert to float, fill NaN with 0
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)   # handle amounts like "$1,234.00"
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0) # type: ignore

    # merchant_id: fill unknowns, normalize casing, merge aliases
    if "merchant_id" in df.columns:
        df["merchant_id"] = df["merchant_id"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        merchant_aliases = {
            "AMZN*MKTP": "AMAZON",
            "AMZN": "AMAZON",
            "FLIPKART*INDIA": "FLIPKART",
        }
        df["merchant_id"] = df["merchant_id"].replace(merchant_aliases)

    # merchant_city: fill unknowns, normalize casing
    if "merchant_city" in df.columns:
        df["merchant_city"] = df["merchant_city"].fillna("UNKNOWN").astype(str).str.strip().str.title()

    # merchant_state: fill unknowns, normalize casing
    if "merchant_state" in df.columns:
        df["merchant_state"] = df["merchant_state"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    # use_chip: fill unknowns, normalize casing
    if "use_chip" in df.columns:
        df["use_chip"] = df["use_chip"].fillna("UNKNOWN").astype(str).str.strip().str.lower()

    # errors: fill empty strings
    if "errors" in df.columns:
        df["errors"] = df["errors"].fillna("").astype(str).str.strip()

    # zip: convert to string for consistency (avoid float zips like 58523.0)
    if "zip" in df.columns:
        df["zip"] = df["zip"].fillna("00000").astype(str).str.split(".").str[0].str.zfill(5)

    # ip_address: Convert raw IP addresses into readable locations (City/Country)
    if "ip_address" in df.columns:
        import requests
        import time
        unique_ips = df["ip_address"].dropna().unique()
        ip_map = {}
        for ip in unique_ips:
            ip_str = str(ip).strip()
            if not ip_str or ip_str.lower() in ("unknown", "nan"):
                ip_map[ip] = "Unknown"
            elif ip_str.startswith("192.168.") or ip_str.startswith("10.") or ip_str == "127.0.0.1":
                ip_map[ip] = "Local Network"
            else:
                try:
                    res = requests.get(f"http://ip-api.com/json/{ip_str}", timeout=2)
                    if res.status_code == 200 and res.json().get("status") == "success":
                        data = res.json()
                        ip_map[ip] = f"{data.get('city', 'Unknown')}, {data.get('countryCode', 'Unknown')}".strip(", ")
                    else:
                        ip_map[ip] = "Unknown Location"
                except Exception:
                    ip_map[ip] = "Unknown Location"
                time.sleep(0.3)
        df["ip_location"] = df["ip_address"].map(ip_map).fillna("Unknown") # type: ignore

    # ── 2. Date parsing ─────────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        bad_dates = df["date"].isna().sum()
        if bad_dates > 0:
            logger.warning("%d rows have unparseable dates — they will be NaT.", bad_dates)

    # ── 3. Deduplication ─────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    dupes = before - len(df)
    if dupes:
        logger.info("Dropped %d duplicate rows.", dupes)

    return df


def save_clean_data(df: pd.DataFrame, path: str = CLEAN_DATA_PATH) -> None:
    """Write the cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved clean data (%d rows) → %s", len(df), path)


def preprocess(raw_path: str = RAW_DATA_PATH, clean_path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Full pipeline: load → clean → save → return DataFrame."""
    df = load_raw_data(raw_path)
    df = clean_data(df)
    save_clean_data(df, clean_path)
    return df


# ── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocess()