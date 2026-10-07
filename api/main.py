import os
import sys
import uuid
import asyncio

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from src.logging_config import setup_logging, get_logger, set_request_id, get_request_id
from src.preprocess import clean_data
from src.rules_engine import apply_rules, load_config, build_user_norms
from src.ml_inference import get_ai_risk_score
from src.database import init_db, log_transaction, fetch_all_logs, clear_logs, get_db_status
from src.alert_system import init_alerts_db, process_and_trigger_alert, fetch_active_alerts, submit_analyst_feedback, refine_and_retrain_model
from src.auth import AuthMiddleware
from src.rate_limiter import RateLimiterMiddleware

# Initialize structured logging at startup
setup_logging()
logger = get_logger(__name__)

# ─── Allowed CORS origins (configurable via environment) ───
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "FINGUARD_CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if origin.strip()
]

# ─── Request / Response Models ───

class RawTransaction(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=100, description="Unique transaction identifier")
    client_id: str = Field(..., min_length=1, max_length=100, description="Client/user identifier")
    amount: float = Field(..., ge=0.0, le=1_000_000.0, description="Transaction amount in USD")
    date: str = Field(..., min_length=1, max_length=50, description="ISO 8601 date string")
    merchant_id: str = Field(..., min_length=1, max_length=200, description="Merchant identifier")
    use_chip: Optional[str] = Field("Online", max_length=50)
    merchant_city: Optional[str] = Field("Unknown", max_length=100)
    velocity_1h: Optional[int] = Field(1, ge=0, le=1000)
    travel_speed_kmh: Optional[float] = Field(0.0, ge=0.0, le=50000.0)
    ip_address: Optional[str] = Field("127.0.0.1", max_length=45)


class FeedbackPayload(BaseModel):
    alert_id: str = Field(..., min_length=1, max_length=100)
    confirmed_is_fraud: bool
    transaction_data: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    Output_A_Summary: Dict[str, Any]
    Output_B_Verdict: Dict[str, Any]
    Output_C_Checklist: List[str]
    Output_D_AlertTriggered: Optional[Dict[str, Any]] = None


class BatchResponse(BaseModel):
    batch_results: List[AnalysisResponse]


class HealthResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    norms_cached_users: int
    pending_alerts_count: int
    gemini_api_active: bool
    version: str


# ─── Application Lifespan ───

PRECOMPUTED_NORMS: Dict[str, Any] = {}
CONFIG: Dict[str, Any] = {}

# Maximum batch size for /analyze_batch
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "100"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global PRECOMPUTED_NORMS, CONFIG

    logger.info("FinGuard AI starting up...")

    # Initialize configuration & DB schemas
    CONFIG = load_config(os.path.join(PROJECT_ROOT, "configs", "fraud_typology.json"))
    init_db()
    init_alerts_db()

    # Precompute historical behavioral norms
    try:
        _data_path = os.path.join(PROJECT_ROOT, "data", "transactions_data.csv")
        if not os.path.isfile(_data_path):
            _data_path = os.path.join(PROJECT_ROOT, "data", "clean_transactions.csv")

        if os.path.isfile(_data_path):
            historical_df = pd.read_csv(_data_path)
            PRECOMPUTED_NORMS = build_user_norms(historical_df)
            logger.info("Historical norms loaded for %d unique clients.", len(PRECOMPUTED_NORMS))
        else:
            PRECOMPUTED_NORMS = {}
            logger.warning("No historical dataset found. Starting with empty norms.")
    except Exception as err:
        logger.error("Failed to load historical norms: %s", err, exc_info=True)
        PRECOMPUTED_NORMS = {}

    logger.info("FinGuard AI ready. CORS origins: %s", ALLOWED_ORIGINS)
    yield

    # Shutdown
    logger.info("FinGuard AI shutting down gracefully.")


# ─── FastAPI Application ───

app = FastAPI(
    title="FinGuard AI - Fraud Detection & Alert System API",
    description="Production-grade real-time AI fraud detection with authentication, rate limiting, and structured logging.",
    version="2.2.0",
    lifespan=lifespan,
)

# ─── Middleware Stack (order matters: outermost first) ───

# 1. CORS — restricted to configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# 2. Rate Limiter
app.add_middleware(RateLimiterMiddleware)

# 3. Authentication & RBAC
app.add_middleware(AuthMiddleware)


# ─── Request ID Middleware ───

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Inject request correlation ID into every request/response cycle."""
    incoming_id = request.headers.get("X-Request-ID")
    request_id = set_request_id(incoming_id)
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ─── Static Files ───

WEBSITE_DIR = os.path.join(PROJECT_ROOT, "website")
if os.path.isdir(WEBSITE_DIR):
    app.mount("/static", StaticFiles(directory=WEBSITE_DIR), name="static")


# ─── Endpoints ───

@app.get("/")
def read_root():
    index_file = os.path.join(WEBSITE_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {
        "status": "online",
        "service": "FinGuard AI Fraud Detection & Alert Engine",
        "version": "2.2.0",
        "endpoints": ["/showcase", "/health", "/v1/analyze", "/v1/analyze_batch", "/v1/alerts", "/v1/model/refine"]
    }


@app.get("/showcase")
def read_showcase():
    index_file = os.path.join(WEBSITE_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Showcase website file not found.")


@app.get("/health", response_model=HealthResponse)
def health_check():
    db_status = get_db_status()
    return {
        "status": "healthy",
        "database": db_status,
        "norms_cached_users": len(PRECOMPUTED_NORMS),
        "pending_alerts_count": len(fetch_active_alerts("PENDING")),
        "gemini_api_active": bool(os.environ.get("GEMINI_API_KEY")),
        "version": "2.2.0",
    }


# ─── V1 API Routes (versioned) ───

@app.post("/v1/analyze", response_model=AnalysisResponse)
@app.post("/analyze", response_model=AnalysisResponse, include_in_schema=False)
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

        logger.info(
            "Transaction %s analyzed: risk=%.1f, type=%s",
            tx.transaction_id,
            ai_verdict.get("risk_score", 0),
            ai_verdict.get("fraud_type", "UNKNOWN"),
        )

        return {
            "Output_A_Summary": clean_dict,
            "Output_B_Verdict": ai_verdict,
            "Output_C_Checklist": checklist,
            "Output_D_AlertTriggered": triggered_alert
        }

    except Exception as e:
        logger.error("Pipeline error for %s: %s", tx.transaction_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/v1/analyze_batch", response_model=BatchResponse)
@app.post("/analyze_batch", response_model=BatchResponse, include_in_schema=False)
async def process_batch(tx_list: List[RawTransaction]):
    if not tx_list:
        return {"batch_results": []}

    if len(tx_list) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(tx_list)} exceeds maximum of {MAX_BATCH_SIZE}. Split into smaller batches."
        )

    try:
        input_df = pd.DataFrame([tx.model_dump() for tx in tx_list])
        cleaned_df = clean_data(input_df)
        final_df = apply_rules(cleaned_df, CONFIG, precomputed_norms=PRECOMPUTED_NORMS)

        semaphore = asyncio.Semaphore(10)

        async def process_one(i: int) -> dict:
            async with semaphore:
                result_row = final_df.iloc[i]
                checklist = result_row.get("suspicious_indicators_checklist", [])
                clean_dict = cleaned_df.iloc[i].to_dict()
                loop = asyncio.get_event_loop()
                ai_verdict = await loop.run_in_executor(
                    None, get_ai_risk_score, clean_dict, checklist
                )
                log_transaction(clean_dict, ai_verdict, checklist)
                triggered_alert = process_and_trigger_alert(clean_dict, ai_verdict, checklist)
                return {
                    "Output_A_Summary": clean_dict,
                    "Output_B_Verdict": ai_verdict,
                    "Output_C_Checklist": checklist,
                    "Output_D_AlertTriggered": triggered_alert
                }

        results = await asyncio.gather(*[process_one(i) for i in range(len(final_df))])
        logger.info("Batch of %d transactions processed successfully.", len(tx_list))
        return {"batch_results": list(results)}

    except Exception as e:
        logger.error("Batch processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")


@app.get("/v1/alerts")
@app.get("/alerts", include_in_schema=False)
def get_alerts(status: str = "PENDING"):
    alerts = fetch_active_alerts(status)
    return {"status": "success", "alerts_count": len(alerts), "alerts": alerts}


@app.post("/v1/alerts/resolve")
@app.post("/alerts/resolve", include_in_schema=False)
def resolve_alert(payload: FeedbackPayload):
    res = submit_analyst_feedback(payload.alert_id, payload.confirmed_is_fraud, payload.transaction_data)
    return res


@app.post("/v1/model/refine")
@app.post("/model/refine", include_in_schema=False)
def trigger_model_refinement():
    res = refine_and_retrain_model()
    return res


@app.get("/v1/analytics")
@app.get("/analytics", include_in_schema=False)
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


@app.get("/v1/analytics/codevalues")
@app.get("/analytics/codevalues", include_in_schema=False)
def get_codevalue_analytics():
    csv_path = os.path.join(PROJECT_ROOT, "data", "codevalue_accuracy_report.csv")
    if os.path.isfile(csv_path):
        df_res = pd.read_csv(csv_path)
        return {"status": "success", "codevalues_count": len(df_res), "metrics": df_res.to_dict(orient="records")}
    return {"status": "error", "message": "CodeValue report not generated yet. Run evaluate_codevalue_metrics.py."}


@app.post("/v1/logs/clear")
@app.post("/logs/clear", include_in_schema=False)
def clear_all_logs():
    clear_logs()
    return {"message": "Transaction logs cleared successfully."}


@app.get("/v1/metrics")
@app.get("/metrics", include_in_schema=False)
def get_operational_metrics():
    """Basic operational metrics endpoint for monitoring."""
    db_status = get_db_status()
    return {
        "service": "finguard-ai",
        "version": "2.2.0",
        "database_rows": db_status.get("total_rows", 0),
        "database_healthy": db_status.get("status") == "healthy",
        "norms_cached": len(PRECOMPUTED_NORMS),
        "auth_enabled": bool(os.environ.get("FINGUARD_API_KEYS")),
        "gemini_enabled": bool(os.environ.get("GEMINI_API_KEY")),
    }