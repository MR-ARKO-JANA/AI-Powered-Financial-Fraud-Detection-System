import os
import sys

# --- Ensure the project root is on the Python path ---
# This allows `src.*` imports to resolve correctly no matter how the app is started.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# --- Importing the real functions from your teammates ---
from src.preprocess import clean_data
from src.rules_engine import apply_rules, load_config, build_user_norms
from src.ml_inference import get_ai_risk_score
from src.database import init_db, log_transaction

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
    ip_address: Optional[str] = "unknown"


# --- 3. INITIALIZE CONFIG ---
# Load the rules once when the API starts
CONFIG = load_config(os.path.join(PROJECT_ROOT, "configs", "fraud_typology.json"))

# Initialize the SQLite logs database
init_db()

# Load historical behavioral context for Sidhu's logic engine
try:
    print("Pre-computing historical transaction norms. This takes a moment but enables instant live responses...")
    _data_path = os.path.join(PROJECT_ROOT, "data", "clean_transactions.csv")
    if os.path.isfile(_data_path):
        HISTORICAL_DF = pd.read_csv(_data_path)
        PRECOMPUTED_NORMS = build_user_norms(HISTORICAL_DF)
        print(f"Loaded bounds for {len(PRECOMPUTED_NORMS)} unique clients.")
        del HISTORICAL_DF # Free up RAM immediately!
    else:
        PRECOMPUTED_NORMS = {}
except Exception as err:
    print(f"Historical dataset skipped (Not Found or Failed): {err}")
    PRECOMPUTED_NORMS = {}

# 3. The Main Pipeline Route
@app.post("/analyze")
async def process_transaction(tx: RawTransaction):
    try:
        # STEP 1: Convert to DataFrame
        input_df = pd.DataFrame([tx.model_dump()])

        # STEP 2: Preprocessing
        cleaned_df = clean_data(input_df)

        # STEP 3: Rules Engine
        final_df = apply_rules(cleaned_df, CONFIG, precomputed_norms=PRECOMPUTED_NORMS)
        result_row = final_df.iloc[0]
        checklist = result_row.get("suspicious_indicators_checklist", [])

        # STEP 4: AI Inference
        clean_dict = cleaned_df.iloc[0].to_dict()
        ai_verdict = get_ai_risk_score(clean_dict, checklist)

        # STEP 4.5: Log to database!
        log_transaction(clean_dict, ai_verdict)

        # STEP 5: Return Formatted Response to Streamlit
        return {
            "Output_A_Summary": clean_dict,
            "Output_B_Verdict": ai_verdict,
            "Output_C_Checklist": checklist
        }

    except Exception as e:
        return {"error": f"Pipeline Break: {str(e)}"}


@app.post("/analyze_batch")
async def process_batch(tx_list: List[RawTransaction]):
    try:
        # STEP 1: Convert to DataFrame
        input_df = pd.DataFrame([tx.model_dump() for tx in tx_list])

        # STEP 2: Preprocessing
        cleaned_df = clean_data(input_df)

        # STEP 3: Rules Engine
        final_df = apply_rules(cleaned_df, CONFIG, precomputed_norms=PRECOMPUTED_NORMS)
        
        results = []
        for i in range(len(final_df)):
            result_row = final_df.iloc[i]
            checklist = result_row.get("suspicious_indicators_checklist", [])
            
            # STEP 4: AI Inference (run per transaction)
            clean_dict = cleaned_df.iloc[i].to_dict()
            ai_verdict = get_ai_risk_score(clean_dict, checklist)
            
            # STEP 4.5: Log to database!
            log_transaction(clean_dict, ai_verdict)
            
            results.append({
                "Output_A_Summary": clean_dict,
                "Output_B_Verdict": ai_verdict,
                "Output_C_Checklist": checklist
            })

        return {"batch_results": results}

    except Exception as e:
        return {"error": f"Batch Pipeline Break: {str(e)}"}