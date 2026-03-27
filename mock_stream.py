import requests
import time
import random
import uuid
from datetime import datetime

API_URL = "http://127.0.0.1:8000/analyze"

MERCHANTS = ["AMZN*MKTP", "FLIPKART", "UBER", "NETFLIX", "STARBUCKS", "UNKNOWN_STORE", "DARKWEBSTORE"]
CITIES = ["New York", "London", "Tokyo", "Mumbai", "Unknown", "Sydney"]

def generate_random_transaction():
    """Generates a realistic mock transaction"""
    
    # 5% chance of being artificially highly suspicious
    is_suspicious = random.random() < 0.05
    
    txn = {
        "transaction_id": f"TXN-SIM-{str(uuid.uuid4())[:8]}",
        "client_id": f"USER-{random.randint(100, 999)}",
        "amount": round(random.uniform(5.0, 5000.0) if is_suspicious else random.uniform(5.0, 300.0), 2),
        "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "merchant_id": random.choice(["DARKWEBSTORE", "SHADYELECTRONICS"]) if is_suspicious else random.choice(MERCHANTS),
        "use_chip": random.choice(["Online", "Swipe", "Chip"]),
        "merchant_city": random.choice(CITIES),
        "velocity_1h": random.randint(5, 15) if is_suspicious else random.randint(1, 3),
        "travel_speed_kmh": random.uniform(800.0, 1500.0) if is_suspicious else random.uniform(0.0, 60.0),
        "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    }
    return txn

def stream_data():
    print("🚀 Starting FinGuard AI Live Data Streamer...")
    print(f"Targeting API: {API_URL}")
    print("Press Ctrl+C to stop.\n")
    
    transactions_sent = 0
    
    try:
        while True:
            payload = generate_random_transaction()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending Transaction {payload['transaction_id']} for ${payload['amount']}")
            
            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    verdict = result.get('Output_B_Verdict', {})
                    score = verdict.get('risk_score', 0)
                    fraud_type = verdict.get('fraud_type', 'Unknown')
                    
                    if score > 50:
                        print(f"  🚨 FRAUD DETECTED: {fraud_type} (Score: {score:.1f})")
                    else:
                        print(f"  ✅ Safe Transaction (Score: {score:.1f})")
                else:
                    print(f"  ⚠️ Error from API: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print("  ❌ Connection Error! Make sure your FastAPI server is running.")
                time.sleep(5)
                
            transactions_sent += 1
            
            # Wait for 3 seconds before sending the next one
            time.sleep(3)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stream stopped. Sent {transactions_sent} total transactions.")

if __name__ == "__main__":
    stream_data()
