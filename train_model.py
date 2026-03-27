import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def generate_training_data(n_samples=5000):
    np.random.seed(42)
    
    # Feature 1: Transaction Amount
    # Normal amounts: mean=100, std=50
    # Fraud amounts: mean=2500, std=1000
    is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    amounts = np.where(is_fraud == 1, 
                       np.random.normal(2500, 1000, n_samples),
                       np.random.normal(100, 50, n_samples))
    amounts = np.clip(amounts, 1.0, None) # Ensure positive
    
    # Feature 2: Velocity (transactions in last hour)
    # Normal velocity: 1 to 3
    # Fraud velocity: 4 to 15
    velocity = np.where(is_fraud == 1,
                        np.random.randint(4, 16, n_samples),
                        np.random.randint(1, 4, n_samples))
                        
    # Feature 3: Travel Speed (km/h)
    # Normal speed: 0 to 120 (driving/flying possible)
    # Fraud speed: 800 to 5000 (impossible travel)
    travel_speed = np.where(is_fraud == 1,
                            np.random.uniform(800, 5000, n_samples),
                            np.random.uniform(0, 120, n_samples))
                            
    # Feature 4: Time of day (0-23)
    # Normal: mostly 8 to 22
    # Fraud: often 1 to 5
    hours = np.where(is_fraud == 1,
                     np.random.choice([1, 2, 3, 4, 5, 23], n_samples),
                     np.random.randint(8, 23, n_samples))
                     
    df = pd.DataFrame({
        'amount': amounts,
        'velocity_1h': velocity,
        'travel_speed_kmh': travel_speed,
        'hour_of_day': hours,
        'is_fraud': is_fraud
    })
    
    return df

def train_and_save():
    print("Generating synthetic training dataset...")
    df = generate_training_data(10000)
    
    X = df[['amount', 'velocity_1h', 'travel_speed_kmh', 'hour_of_day']]
    y = df['is_fraud']
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(model, 'models/fraud_classifier.pkl')
    
    print("Training Complete! Models saved to models/ directory.")
    print("Feature importances:", dict(zip(X.columns, model.feature_importances_)))

if __name__ == "__main__":
    train_and_save()
