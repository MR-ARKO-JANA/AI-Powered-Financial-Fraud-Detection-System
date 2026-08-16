import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate synthetic financial transaction dataset with realistic features."""
    np.random.seed(42)
    
    # 5% target fraud rate
    is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Feature 1: Amount ($)
    amounts = np.where(
        is_fraud == 1,
        np.random.normal(3200, 1200, n_samples),
        np.random.normal(120, 80, n_samples)
    )
    amounts = np.clip(amounts, 1.0, 25000.0)
    
    # Feature 2: Velocity (transactions in last 1 hour)
    velocity = np.where(
        is_fraud == 1,
        np.random.randint(4, 16, n_samples),
        np.random.randint(1, 4, n_samples)
    )
    
    # Feature 3: Travel Speed (km/h)
    travel_speed = np.where(
        is_fraud == 1,
        np.random.uniform(750, 4500, n_samples),
        np.random.uniform(0, 110, n_samples)
    )
    
    # Feature 4: Hour of Day (0-23)
    hours = np.where(
        is_fraud == 1,
        np.random.choice([1, 2, 3, 4, 23], n_samples),
        np.random.randint(7, 23, n_samples)
    )

    df = pd.DataFrame({
        'amount': np.round(amounts, 2),
        'velocity_1h': velocity,
        'travel_speed_kmh': np.round(travel_speed, 2),
        'hour_of_day': hours,
        'is_fraud': is_fraud
    })
    
    return df

def generate_sample_transactions_csv(n_records: int = 150) -> pd.DataFrame:
    """Generate a realistic raw transaction log CSV for historical norms and batch testing."""
    np.random.seed(101)
    
    merchants = ["AMZN*MKTP", "FLIPKART*INDIA", "UBER*TRIP", "NETFLIX", "STARBUCKS", "DARKWEB*PAY", "SHADY*ELECTRONICS", "APPLE*STORE"]
    cities = ["New York", "London", "Tokyo", "Mumbai", "San Francisco", "Sydney", "Berlin", "Dubai"]
    use_chips = ["Chip", "Swipe", "Online"]
    
    records = []
    base_time = pd.Timestamp("2026-03-22 08:00:00")
    
    for i in range(n_records):
        client_id = f"USER-{np.random.randint(100, 120)}" # 20 unique clients
        is_fraud = np.random.random() < 0.08
        
        amt = round(float(np.random.uniform(2500, 9500) if is_fraud else np.random.uniform(10, 400)), 2)
        merch = np.random.choice(["DARKWEB*PAY", "SHADY*ELECTRONICS"]) if is_fraud else np.random.choice(merchants)
        city = np.random.choice(cities)
        chip = "Online" if is_fraud else np.random.choice(use_chips)
        vel = int(np.random.randint(5, 14) if is_fraud else np.random.randint(1, 3))
        speed = round(float(np.random.uniform(850, 3000) if is_fraud else np.random.uniform(0, 50)), 1)
        ip = f"{np.random.randint(1,254)}.{np.random.randint(1,254)}.{np.random.randint(1,254)}.{np.random.randint(1,254)}"
        
        timestamp = base_time + pd.Timedelta(minutes=i * 15 + np.random.randint(0, 10))
        
        records.append({
            "transaction_id": f"TXN-{1000 + i}",
            "client_id": client_id,
            "amount": f"${amt:,.2f}" if i % 3 == 0 else amt, # Introduce raw messy format for preprocessing test
            "date": timestamp.isoformat() + "Z",
            "merchant_id": merch,
            "use_chip": chip,
            "merchant_city": city,
            "velocity_1h": vel,
            "travel_speed_kmh": speed,
            "ip_address": ip
        })
        
    return pd.DataFrame(records)

def train_and_save():
    print("[1/3] Generating synthetic training dataset (10,000 samples)...")
    df = generate_synthetic_data(10000)
    
    X = df[['amount', 'velocity_1h', 'travel_speed_kmh', 'hour_of_day']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[2/3] Fitting StandardScaler & Training Random Forest Classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    print("\nModel Performance Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Fraud"]))
    
    models_dir = os.path.join(PROJECT_ROOT, "models")
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(model, os.path.join(models_dir, "fraud_classifier.pkl"))
    print(f"Models saved to: {models_dir}")
    
    # Save seed transaction data
    sample_df = generate_sample_transactions_csv(150)
    sample_path = os.path.join(data_dir, "transactions_data.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Seed transactions CSV generated ({len(sample_df)} rows) at: {sample_path}")

if __name__ == "__main__":
    train_and_save()
