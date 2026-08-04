import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "transformer_rf_pipeline.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_classifier.pkl")

def get_trained_pipeline():
    """Load the fine-tuned Transformer & Random Forest Hybrid Pipeline."""
    try:
        if os.path.isfile(PIPELINE_PATH):
            return joblib.load(PIPELINE_PATH)
        elif os.path.isfile(MODEL_PATH):
            return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load ML pipeline: {e}")
    return None

def prepare_feature_dataframe(clean_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Construct a single-row DataFrame containing all numerical and categorical
    features matching the fine-tuned model pipeline schema.
    """
    try:
        amount = float(clean_data.get("amount", 0.0))
    except (ValueError, TypeError):
        amount = 0.0

    try:
        velocity = int(clean_data.get("velocity_1h", 1))
    except (ValueError, TypeError):
        velocity = 1

    try:
        travel_speed = float(clean_data.get("travel_speed_kmh", 0.0))
    except (ValueError, TypeError):
        travel_speed = 0.0

    date_str = str(clean_data.get("date", ""))
    hour_of_day = 12
    if "T" in date_str:
        try:
            hour_of_day = int(date_str.split("T")[1].split(":")[0])
        except Exception:
            pass
    elif " " in date_str:
        try:
            hour_of_day = int(date_str.split(" ")[1].split(":")[0])
        except Exception:
            pass

    merchant = str(clean_data.get("merchant_id", "UNKNOWN")).upper()
    use_chip = str(clean_data.get("use_chip", "Online")).capitalize()
    city = str(clean_data.get("merchant_city", "Unknown")).title()
    ip_loc = str(clean_data.get("ip_location", "Unknown Location"))

    df = pd.DataFrame([{
        "amount": amount,
        "velocity_1h": velocity,
        "travel_speed_kmh": travel_speed,
        "hour_of_day": hour_of_day,
        "merchant_id": merchant,
        "use_chip": use_chip,
        "merchant_city": city,
        "ip_location": ip_loc
    }])

    return df

def prepare_ml_features(clean_data: Dict[str, Any]) -> pd.DataFrame:
    """Returns formatted feature DataFrame ready for model pipeline prediction."""
    return prepare_feature_dataframe(clean_data)
