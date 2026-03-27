import sqlite3
import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "transactions_log.db")

def init_db():
    """Create the analytics transactions table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transaction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            client_id TEXT,
            amount REAL,
            date TEXT,
            merchant_id TEXT,
            velocity_1h INTEGER,
            travel_speed_kmh REAL,
            fraud_type TEXT,
            composite_risk_score REAL,
            ai_reason TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_transaction(clean_data: dict, ai_verdict: dict):
    """Save the transaction details and AI verdict to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transaction_logs (
                transaction_id, client_id, amount, date, merchant_id, 
                velocity_1h, travel_speed_kmh, fraud_type, composite_risk_score, ai_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(clean_data.get("transaction_id", "")),
            str(clean_data.get("client_id", "")),
            float(clean_data.get("amount", 0.0)),
            str(clean_data.get("date", "")),
            str(clean_data.get("merchant_id", "")),
            int(clean_data.get("velocity_1h", 1)),
            float(clean_data.get("travel_speed_kmh", 0.0)),
            str(ai_verdict.get("fraud_type", "UNKNOWN")),
            float(ai_verdict.get("risk_score", 0.0)),
            str(ai_verdict.get("reason", ""))
        ))
        
        conn.commit()
    except Exception as e:
        print(f"Failed to log transaction to DB: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
