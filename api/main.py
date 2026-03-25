from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# 1. Initialize the API
app = FastAPI(title="Fraud Detection Pipeline API")

# 2. Define the "Contract" for incoming data
# This tells FastAPI exactly what a "messy log" should look like when it arrives.
class RawTransaction(BaseModel):
    transaction_id: str
    amount: float
    ip_address: str
    merchant: str
    timestamp: str

# --- TEAMMATE PLACEHOLDERS ---
# You will eventually replace these with imports from Arko, Sidhu, and Subham's files.

def arko_clean_data(raw_tx: RawTransaction):
    # Simulates Arko cleaning the data and getting the location
    return {
        "transaction_id": raw_tx.transaction_id,
        "date": raw_tx.timestamp.split("T")[0], # Fake date split
        "amount": raw_tx.amount,
        "sender": "Victim_User_1",
        "merchant": raw_tx.merchant.replace("*", " "), # Fake cleaning
        "location_metadata": "Delhi, IN (Derived from IP)"
    }

def sidhu_check_rules(clean_data: dict):
    # Simulates Sidhu checking the fraud_typology.json
    return [
        {"indicator": "Login from unrecognized IP/Location", "status": "PRESENT", "reason": "IP mismatch detected"},
        {"indicator": "Multiple failed login attempts", "status": "MISSING", "reason": "Normal behavior"}
    ]

def subham_run_ai(clean_data: dict, indicators: list):
    # Simulates Subham's AI returning a verdict
    return {
        "fraud_type": "Account Takeover",
        "risk_score": 85,
        "reason": "Sudden location change detected with high transaction amount."
    }
# -----------------------------

# 3. The Main Pipeline Route
@app.post("/analyze")
async def process_transaction(tx: RawTransaction):
    """
    This is the main engine. It takes a raw transaction, passes it through
    the team's functions, and formats the final Output A, B, and C.
    """
    
    # Step 1: Data Engineering (Arko)
    clean_summary = arko_clean_data(tx)
    
    # Step 2: Rule Checking (Sidhu)
    checklist = sidhu_check_rules(clean_summary)
    
    # Step 3: AI Inference (Subham)
    verdict = subham_run_ai(clean_summary, checklist)
    
    # Step 4: System Architect (Joyce) formats the final response
    return {
        "Output_A_Summary": clean_summary,
        "Output_B_Verdict": verdict,
        "Output_C_Checklist": checklist
    }