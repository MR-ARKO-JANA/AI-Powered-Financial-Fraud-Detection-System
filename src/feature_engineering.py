import os
import joblib
import numpy as np

# Load the scaler once when the module starts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

try:
    if os.path.isfile(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = None
except Exception:
    scaler = None

def prepare_ml_features(clean_data: dict) -> np.ndarray:
    """
    Extracts numerical features from the cleaned transaction dictionary
    and scales them using the pre-fitted StandarScaler.
    
    Expected features: ['amount', 'velocity_1h', 'travel_speed_kmh', 'hour_of_day']
    """
    amount = float(clean_data.get("amount", 0.0))
    velocity = int(clean_data.get("velocity_1h", 1))
    travel_speed = float(clean_data.get("travel_speed_kmh", 0.0))
    
    # Extract hour from ISO date: "2026-03-22T10:30:00Z" -> 10
    date_str = str(clean_data.get("date", ""))
    try:
        hour_of_day = int(date_str.split("T")[1].split(":")[0])
    except Exception:
        hour_of_day = 12 # Default to noon if parsing fails
        
    features = np.array([[amount, velocity, travel_speed, hour_of_day]])
    
    if scaler is not None:
        scaled_features = scaler.transform(features)
        return scaled_features
    else:
        return features # Fallback to unscaled
