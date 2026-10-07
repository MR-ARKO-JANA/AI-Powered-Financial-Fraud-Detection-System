"""
database.py -- Production-Grade Database Engine for FinGuard AI.
PostgreSQL (production) | SQLite (dev fallback)
"""

import os, json, sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from src.logging_config import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# Try PostgreSQL only if DATABASE_URL is set AND psycopg2 is available
USE_POSTGRES = False
if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
    try:
        import psycopg2
        import psycopg2.pool
        USE_POSTGRES = True
    except ImportError:
        logger.warning("psycopg2 is not installed. Falling back to SQLite.")
        USE_POSTGRES = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(PROJECT_ROOT, "data", "transactions_log.db"))
DB_PATH = SQLITE_DB_PATH

if USE_POSTGRES:
    logger.info("Database: PostgreSQL (%s)", DATABASE_URL.split("@")[-1])
else:
    logger.info("Database: SQLite (%s)", SQLITE_DB_PATH)

_pg_pool = None

def _get_pg_pool():
    global _pg_pool
    if _pg_pool is None and USE_POSTGRES:
        from psycopg2 import pool as pg_pool
        _pg_pool = pg_pool.ThreadedConnectionPool(
            minconn=2, maxconn=20, dsn=DATABASE_URL, connect_timeout=10
        )
        logger.info("PostgreSQL pool initialized (2-20 connections).")
    return _pg_pool

@contextmanager
def get_connection():
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            pool.putconn(conn)
    else:
        effective_path = globals().get("DB_PATH", SQLITE_DB_PATH)
        os.makedirs(os.path.dirname(effective_path), exist_ok=True)
        conn = sqlite3.connect(effective_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

def _ph() -> str:
    return "%s" if USE_POSTGRES else "?"

def init_db():
    pk = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    ts = "TIMESTAMPTZ" if USE_POSTGRES else "DATETIME"
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(f"""CREATE TABLE IF NOT EXISTS transaction_logs (
            id {pk}, transaction_id TEXT, client_id TEXT, amount REAL, date TEXT,
            merchant_id TEXT, merchant_city TEXT, use_chip TEXT, ip_address TEXT,
            ip_location TEXT, velocity_1h INTEGER, travel_speed_kmh REAL,
            fraud_type TEXT, composite_risk_score REAL, ai_reason TEXT,
            checklist_json TEXT, timestamp {ts} DEFAULT CURRENT_TIMESTAMP)""")
        for idx, col in [("idx_tx_id","transaction_id"),("idx_client_id","client_id"),
                         ("idx_risk_score","composite_risk_score"),("idx_timestamp","timestamp")]:
            c.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON transaction_logs ({col});")
        logger.info("DB schema initialized (%s).", "PostgreSQL" if USE_POSTGRES else "SQLite")

def log_transaction(clean_data, ai_verdict, checklist=None):
    ph = _ph()
    sql = f"""INSERT INTO transaction_logs (transaction_id, client_id, amount, date,
        merchant_id, merchant_city, use_chip, ip_address, ip_location, velocity_1h,
        travel_speed_kmh, fraud_type, composite_risk_score, ai_reason, checklist_json)
        VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})"""
    vals = (str(clean_data.get("transaction_id","")), str(clean_data.get("client_id","")),
            float(clean_data.get("amount",0.0)), str(clean_data.get("date","")),
            str(clean_data.get("merchant_id","")), str(clean_data.get("merchant_city","Unknown")),
            str(clean_data.get("use_chip","Online")), str(clean_data.get("ip_address","")),
            str(clean_data.get("ip_location","Unknown")), int(clean_data.get("velocity_1h",1)),
            float(clean_data.get("travel_speed_kmh",0.0)), str(ai_verdict.get("fraud_type","UNKNOWN")),
            float(ai_verdict.get("risk_score",0.0)), str(ai_verdict.get("reason","")),
            json.dumps(checklist or []))
    with get_connection() as conn:
        conn.cursor().execute(sql, vals)

def fetch_recent_logs(limit=500):
    init_db()
    ph = _ph()
    with get_connection() as conn:
        if USE_POSTGRES:
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(f"SELECT * FROM transaction_logs ORDER BY id DESC LIMIT {ph}", (limit,))
            rows = cur.fetchall()
            return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
        effective_path = globals().get("DB_PATH", SQLITE_DB_PATH)
        return pd.read_sql_query("SELECT * FROM transaction_logs ORDER BY id DESC LIMIT ?", conn, params=(limit,))

def fetch_all_logs():
    return fetch_recent_logs(limit=5000)

def clear_logs():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("TRUNCATE TABLE transaction_logs RESTART IDENTITY;" if USE_POSTGRES else "DELETE FROM transaction_logs")
    logger.info("Logs cleared.")

def get_db_status():
    try:
        effective_path = globals().get("DB_PATH", SQLITE_DB_PATH)
        db_exists = True if USE_POSTGRES else (os.path.isfile(effective_path) or True)
        row_count = 0
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM transaction_logs")
            row_count = c.fetchone()[0]
        return {
            "status": "healthy",
            "backend": "postgresql" if USE_POSTGRES else "sqlite",
            "path": effective_path if not USE_POSTGRES else DATABASE_URL,
            "exists": True,
            "total_rows": row_count,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "exists": False}
