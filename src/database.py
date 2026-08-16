"""
database.py — Production-Grade SQLite Storage Engine for FinGuard AI.

Configured with WAL (Write-Ahead Logging) Mode, In-Memory Page Caching,
Connection Context Managers, Parameterized Queries, and Indexed Columns.
"""

import sqlite3
import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

from src.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "transactions_log.db")


@contextmanager
def get_connection():
    """
    Context-managed SQLite connection with WAL mode and performance pragmas.
    Guarantees connection cleanup even on exceptions.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row

    # High-Throughput SQLite Performance Pragmas
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-64000;")     # 64 MB RAM cache
    conn.execute("PRAGMA temp_store=MEMORY;")

    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize SQLite transaction_logs schema and indexes."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                client_id TEXT,
                amount REAL,
                date TEXT,
                merchant_id TEXT,
                merchant_city TEXT,
                use_chip TEXT,
                ip_address TEXT,
                ip_location TEXT,
                velocity_1h INTEGER,
                travel_speed_kmh REAL,
                fraud_type TEXT,
                composite_risk_score REAL,
                ai_reason TEXT,
                checklist_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_id ON transaction_logs(transaction_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_client_id ON transaction_logs(client_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_score ON transaction_logs(composite_risk_score);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON transaction_logs(timestamp DESC);")

        conn.commit()
        logger.info("Database schema initialized successfully.")


def log_transaction(clean_data: Dict[str, Any], ai_verdict: Dict[str, Any], checklist: Optional[List[str]] = None):
    """Save enriched transaction details to database with guaranteed connection cleanup."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            checklist_str = json.dumps(checklist or [])  # pyright: ignore[reportArgumentType]

            cursor.execute('''
                INSERT INTO transaction_logs (
                    transaction_id, client_id, amount, date, merchant_id,
                    merchant_city, use_chip, ip_address, ip_location,
                    velocity_1h, travel_speed_kmh, fraud_type, composite_risk_score,
                    ai_reason, checklist_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(clean_data.get("transaction_id", "")),
                str(clean_data.get("client_id", "")),
                float(clean_data.get("amount", 0.0)),
                str(clean_data.get("date", "")),
                str(clean_data.get("merchant_id", "")),
                str(clean_data.get("merchant_city", "Unknown")),
                str(clean_data.get("use_chip", "Online")),
                str(clean_data.get("ip_address", "")),
                str(clean_data.get("ip_location", "Unknown")),
                int(clean_data.get("velocity_1h", 1)),
                float(clean_data.get("travel_speed_kmh", 0.0)),
                str(ai_verdict.get("fraud_type", "UNKNOWN")),
                float(ai_verdict.get("risk_score", 0.0)),
                str(ai_verdict.get("reason", "")),
                checklist_str
            ))

            conn.commit()
            logger.debug("Transaction %s logged successfully.", clean_data.get("transaction_id", ""))

    except Exception as e:
        logger.error("Failed to log transaction: %s", e, exc_info=True)
        raise


def fetch_recent_logs(limit: int = 500) -> pd.DataFrame:
    """Fetch recent logs using parameterized query with safe connection handling."""
    init_db()
    try:
        with get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM transaction_logs ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,)
            )
            return df
    except Exception as e:
        logger.error("Failed to fetch logs: %s", e, exc_info=True)
        return pd.DataFrame()


def fetch_all_logs() -> pd.DataFrame:
    """Fetch all logged transactions (capped at 5000)."""
    return fetch_recent_logs(limit=5000)


def clear_logs():
    """Clear transaction log table and optimize WAL journal."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transaction_logs")
            conn.commit()
            logger.info("Transaction logs cleared successfully.")
    except Exception as e:
        logger.error("Failed to clear logs: %s", e, exc_info=True)
        raise

    # WAL checkpoint in a separate connection to avoid table lock conflicts
    try:
        with get_connection() as conn:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
    except Exception:
        pass  # Non-critical: WAL checkpoint failure is acceptable


def get_db_status() -> Dict[str, Any]:
    """Get database health status for the /health endpoint."""
    try:
        db_exists = os.path.isfile(DB_PATH)
        row_count = 0
        if db_exists:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM transaction_logs")
                row_count = cursor.fetchone()[0]
        return {
            "status": "healthy",
            "path": DB_PATH,
            "exists": db_exists,
            "total_rows": row_count,
        }
    except Exception as e:
        logger.error("Database health check failed: %s", e, exc_info=True)
        return {"status": "unhealthy", "error": str(e)}
