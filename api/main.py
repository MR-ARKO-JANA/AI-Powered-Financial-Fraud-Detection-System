import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.preprocess import clean_data
from src.rules_engine import apply_rules, load_config, build_user_norms
from src.ml_inference import get_ai_risk_score
from src.database import init_db, log_transaction, fetch_all_logs, clear_logs
from src.alert_system import init_alerts_db, process_and_trigger_alert, fetch_active_alerts, submit_analyst_feedback, refine_and_retrain_model

app = FastAPI(
    title="FinGuard AI — Fraud Detection & Alert System API",
    description="High-performance real-time AI, Tabular Transformer, and Active Alert Refinement REST API.",
    version="2.1.0"
)

# Enable CORS for external dashboards and integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEBSITE_DIR = os.path.join(PROJECT_ROOT, "website")
if os.path.isdir(WEBSITE_DIR):
    app.mount("/static", StaticFiles(directory=WEBSITE_DIR), name="static")


class RawTransaction(BaseModel):
    transaction_id: str
    client_id: str
    amount: float
    date: str
    merchant_id: str
    use_chip: Optional[str] = "Online"
    merchant_city: Optional[str] = "Unknown"
    velocity_1h: Optional[int] = 1
    travel_speed_kmh: Optional[float] = 0.0
    ip_address: Optional[str] = "127.0.0.1"


class FeedbackPayload(BaseModel):
    alert_id: str
    confirmed_is_fraud: bool
    transaction_data: Optional[Dict[str, Any]] = None


# Initialize configuration & DB schemas
CONFIG = load_config(os.path.join(PROJECT_ROOT, "configs", "fraud_typology.json"))
init_db()
init_alerts_db()

# Precompute historical behavioral norms
PRECOMPUTED_NORMS: Dict[str, Any] = {}
try:
    _data_path = os.path.join(PROJECT_ROOT, "data", "transactions_data.csv")
    if not os.path.isfile(_data_path):
        _data_path = os.path.join(PROJECT_ROOT, "data", "clean_transactions.csv")
    
    if os.path.isfile(_data_path):
        historical_df = pd.read_csv(_data_path)
        PRECOMPUTED_NORMS = build_user_norms(historical_df)
        print(f"✅ Historical norms loaded for {len(PRECOMPUTED_NORMS)} unique clients.")
    else:
        PRECOMPUTED_NORMS = {}
except Exception as err:
    print(f"⚠️ Historical dataset notice: {err}")
    PRECOMPUTED_NORMS = {}


@app.get("/")
def read_root():
    index_file = os.path.join(WEBSITE_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {
        "status": "online",
        "service": "FinGuard AI Fraud Detection & Alert Engine",
        "version": "2.1.0",
        "endpoints": ["/showcase", "/health", "/analyze", "/analyze_batch", "/alerts", "/model/refine"]
    }


@app.get("/showcase")
def read_showcase():
    index_file = os.path.join(WEBSITE_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Showcase website file not found.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": os.path.isfile(os.path.join(PROJECT_ROOT, "data", "transactions_log.db")),
        "norms_cached_users": len(PRECOMPUTED_NORMS),
        "pending_alerts_count": len(fetch_active_alerts("PENDING")),
        "gemini_api_active": bool(os.environ.get("GEMINI_API_KEY"))
    }


@app.post("/analyze")
async def process_transaction(tx: RawTransaction):
    try:
        input_df = pd.DataFrame([tx.model_dump()])
        cleaned_df = clean_data(input_df)

        final_df = apply_rules(cleaned_df, CONFIG, precomputed_norms=PRECOMPUTED_NORMS)
        result_row = final_df.iloc[0]
        checklist = result_row.get("suspicious_indicators_checklist", [])

        clean_dict = cleaned_df.iloc[0].to_dict()
        ai_verdict = get_ai_risk_score(clean_dict, checklist)

        # Log transaction to SQLite
        log_transaction(clean_dict, ai_verdict, checklist)

        # Evaluate and Trigger Real-Time Model Alert
        triggered_alert = process_and_trigger_alert(clean_dict, ai_verdict, checklist)

        return {
            "Output_A_Summary": clean_dict,
            "Output_B_Verdict": ai_verdict,
            "Output_C_Checklist": checklist,
            "Output_D_AlertTriggered": triggered_alert
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/analyze_batch")
async def process_batch(tx_list: List[RawTransaction]):
    if not tx_list:
        return {"batch_results": []}

    try:
        input_df = pd.DataFrame([tx.model_dump() for tx in tx_list])
        cleaned_df = clean_data(input_df)
        final_df = apply_rules(cleaned_df, CONFIG, precomputed_norms=PRECOMPUTED_NORMS)

        results = []
        for i in range(len(final_df)):
            result_row = final_df.iloc[i]
            checklist = result_row.get("suspicious_indicators_checklist", [])
            clean_dict = cleaned_df.iloc[i].to_dict()

            ai_verdict = get_ai_risk_score(clean_dict, checklist)
            log_transaction(clean_dict, ai_verdict, checklist)
            triggered_alert = process_and_trigger_alert(clean_dict, ai_verdict, checklist)

            results.append({
                "Output_A_Summary": clean_dict,
                "Output_B_Verdict": ai_verdict,
                "Output_C_Checklist": checklist,
                "Output_D_AlertTriggered": triggered_alert
            })

        return {"batch_results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")


@app.get("/alerts")
def get_alerts(status: str = "PENDING"):
    alerts = fetch_active_alerts(status)
    return {"status": "success", "alerts_count": len(alerts), "alerts": alerts}


@app.post("/alerts/resolve")
def resolve_alert(payload: FeedbackPayload):
    res = submit_analyst_feedback(payload.alert_id, payload.confirmed_is_fraud, payload.transaction_data)
    return res


@app.post("/model/refine")
def trigger_model_refinement():
    res = refine_and_retrain_model()
    return res


@app.get("/analytics")
def get_analytics():
    df = fetch_all_logs()
    if df.empty:
        return {
            "total_transactions": 0,
            "total_fraud_detected": 0,
            "fraud_rate_pct": 0.0,
            "avg_risk_score": 0.0,
            "fraud_by_type": {}
        }

    total = len(df)
    fraud_df = df[df["composite_risk_score"] >= 50]
    total_fraud = len(fraud_df)
    by_type = df["fraud_type"].value_counts().to_dict()

    return {
        "total_transactions": total,
        "total_fraud_detected": total_fraud,
        "fraud_rate_pct": round((total_fraud / total) * 100, 2),
        "avg_risk_score": round(df["composite_risk_score"].mean(), 2),
        "fraud_by_type": by_type
    }


@app.get("/analytics/codevalues")
def get_codevalue_analytics():
    csv_path = os.path.join(PROJECT_ROOT, "data", "codevalue_accuracy_report.csv")
    if os.path.isfile(csv_path):
        df_res = pd.read_csv(csv_path)
        return {"status": "success", "codevalues_count": len(df_res), "metrics": df_res.to_dict(orient="records")}
    return {"status": "error", "message": "CodeValue report not generated yet. Run evaluate_codevalue_metrics.py."}


@app.post("/logs/clear")
def clear_all_logs():
    clear_logs()
    return {"message": "Transaction logs cleared successfully."}