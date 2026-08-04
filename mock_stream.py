import requests
import time
import random
import uuid
from datetime import datetime

API_URL = "http://127.0.0.1:8000/analyze"

MERCHANTS = ["AMZN*MKTP", "FLIPKART", "UBER", "NETFLIX", "STARBUCKS", "DARKWEB*PAY", "SHADY*ELECTRONICS"]
CITIES = ["New York", "London", "Tokyo", "Mumbai", "Berlin", "Sydney"]

def generate_random_transaction():
    """Generates a realistic mock transaction."""
    is_suspicious = random.random() < 0.15
    
    txn = {
        "transaction_id": f"TXN-SIM-{str(uuid.uuid4())[:8].upper()}",
        "client_id": f"USER-{random.randint(100, 115)}",
        "amount": round(random.uniform(2500.0, 7500.0) if is_suspicious else random.uniform(5.0, 250.0), 2),
        "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "merchant_id": random.choice(["DARKWEB*PAY", "SHADY*ELECTRONICS"]) if is_suspicious else random.choice(MERCHANTS),
        "use_chip": "Online" if is_suspicious else random.choice(["Online", "Swipe", "Chip"]),
        "merchant_city": random.choice(CITIES),
        "velocity_1h": random.randint(6, 14) if is_suspicious else random.randint(1, 3),
        "travel_speed_kmh": round(random.uniform(850.0, 2200.0) if is_suspicious else random.uniform(0.0, 50.0), 1),
        "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    }
    return txn

def stream_data():
    print("==================================================")
    print("🚀 FinGuard AI — Real-Time Live Data Streamer")
    print(f"Targeting API: {API_URL}")
    print("Press Ctrl+C to stop.")
    print("==================================================\n")
    
    sent_count = 0
    fraud_count = 0
    
    try:
        while True:
            payload = generate_random_transaction()
            time_str = datetime.now().strftime('%H:%M:%S')
            print(f"[{time_str}] Sending TX {payload['transaction_id']} | Client: {payload['client_id']} | Amt: ${payload['amount']:,.2f}")
            
            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    verdict = result.get('Output_B_Verdict', {})
                    score = verdict.get('risk_score', 0)
                    fraud_type = verdict.get('fraud_type', 'Unknown')
                    
                    if score >= 50:
                        fraud_count += 1
                        print(f"  🚨 FRAUD DETECTED: {fraud_type} (Score: {score:.1f}/100)")
                        print(f"     Reason: {verdict.get('reason')}")
                    else:
                        print(f"  ✅ Safe Transaction (Score: {score:.1f}/100)")
                else:
                    print(f"  ⚠️ API returned status: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print("  ❌ Connection Error! Make sure FastAPI server is running on port 8000.")
                time.sleep(4)
                
            sent_count += 1
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stream stopped. Sent {sent_count} total transactions ({fraud_count} flagged as fraud).")

if __name__ == "__main__":
    stream_data()
