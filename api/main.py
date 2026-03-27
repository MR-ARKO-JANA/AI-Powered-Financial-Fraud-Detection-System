import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# --- NEW: Importing the real functions from your teammates ---
from src.preprocess import preprocess_data
from src.rules_engine import apply_rules, load_config
from src.ml_inference import get_ai_risk_score

# 1. Initialize the API
app = FastAPI(title="Fraud Detection Pipeline API")

# 2. Define the "Contract" for incoming data
# This tells FastAPI exactly what a "messy log" should look like when it arrives.
class RawTransaction(BaseModel):
    transaction_id: str
    client_id: str
    amount: float
    date: str
    merchant_id: str
    use_chip: Optional[str] = "unknown"
    merchant_city: Optional[str] = "unknown"
    velocity_1h: Optional[int] = 0
    travel_speed_kmh: Optional[float] = 0.0


# --- 3. INITIALIZE CONFIG ---
# Load the rules once when the API starts
CONFIG = load_config("configs/fraud_typology.json")

# 3. The Main Pipeline Route
@app.post("/analyze")
async def process_transaction(tx: RawTransaction):
    try:
        # STEP 1: Convert to DataFrame
        input_df = pd.DataFrame([tx.dict()])

        # STEP 2: Preprocessing
        cleaned_df = preprocess_data(input_df)

        # STEP 3: Rules Engine
        final_df = apply_rules(cleaned_df, CONFIG, historical_df=None)
        result_row = final_df.iloc[0]
        checklist = result_row.get("suspicious_indicators_checklist", [])

        # STEP 4: AI Inference
        clean_dict = cleaned_df.iloc[0].to_dict()
        ai_verdict = get_ai_risk_score(clean_dict, checklist)

        # STEP 5: Return Formatted Response to Streamlit
        return {
            "Output_A_Summary": clean_dict,
            "Output_B_Verdict": ai_verdict,
            "Output_C_Checklist": checklist
        }

    except Exception as e:
        return {"error": f"Pipeline Break: {str(e)}"}