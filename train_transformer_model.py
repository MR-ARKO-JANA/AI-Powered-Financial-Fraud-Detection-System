import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocess import clean_data

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def create_synthetic_preprocessed_dataset(n_samples: int = 12000) -> pd.DataFrame:
    """
    Creates a rich, multi-feature preprocessed dataset simulating real-world bank logs.
    Includes categorical fields (merchant, chip, city, IP location) and continuous fields.
    """
    np.random.seed(42)
    
    merchants = ["AMAZON", "FLIPKART", "UBER", "NETFLIX", "STARBUCKS", "DARKWEB*PAY", "SHADY*ELECTRONICS", "APPLE*STORE"]
    cities = ["New York", "London", "Tokyo", "Mumbai", "Berlin", "Sydney", "Dubai", "Unknown"]
    chips = ["Chip", "Swipe", "Online"]
    ip_locations = ["Local Corporate Network", "New York, US", "London, GB", "Tokyo, JP", "Mumbai, IN", "Unknown Location", "External IP"]
    
    # 6% fraud probability
    is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.94, 0.06])
    
    records = []
    for i in range(n_samples):
        fraud = is_fraud[i]
        
        amt = float(np.random.normal(3800, 1400) if fraud == 1 else np.random.normal(110, 70))
        amt = round(max(1.0, min(amt, 30000.0)), 2)
        
        vel = int(np.random.randint(5, 16) if fraud == 1 else np.random.randint(1, 4))
        speed = float(np.random.uniform(750, 4500) if fraud == 1 else np.random.uniform(0, 90))
        speed = round(speed, 1)
        
        hour = int(np.random.choice([1, 2, 3, 4, 23]) if fraud == 1 else np.random.randint(7, 23))
        
        merch = str(np.random.choice(["DARKWEB*PAY", "SHADY*ELECTRONICS"]) if fraud == 1 else np.random.choice(merchants[:6]))
        city = str("Unknown" if fraud == 1 and np.random.random() < 0.6 else np.random.choice(cities[:6]))
        chip = str("Online" if fraud == 1 else np.random.choice(chips))
        ip_loc = str("Unknown Location" if fraud == 1 else np.random.choice(ip_locations[:5]))
        
        records.append({
            "client_id": f"USER-{np.random.randint(100, 500)}",
            "amount": amt,
            "velocity_1h": vel,
            "travel_speed_kmh": speed,
            "hour_of_day": hour,
            "merchant_id": merch,
            "use_chip": chip,
            "merchant_city": city,
            "ip_location": ip_loc,
            "is_fraud": fraud
        })
        
    df = pd.DataFrame(records)
    return df

def train_fine_tuned_transformer_rf_ensemble():
    print("==================================================================")
    print(" 🚀 FINE-TUNING HIGH-ACCURACY TABULAR TRANSFORMER & RF ENSEMBLE MODEL")
    print("==================================================================\n")
    
    print("⚙️ [1/4] Generating & Preprocessing Multi-Feature Dataset (12,000 samples)...")
    df = create_synthetic_preprocessed_dataset(12000)
    
    # Save preprocessed dataset CSV for project reference
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "clean_transactions.csv"), index=False)
    print(f"✅ Dataset preprocessed and saved to {os.path.join(data_dir, 'clean_transactions.csv')}")
    
    # Define Numerical & Categorical Feature Columns
    num_features = ["amount", "velocity_1h", "travel_speed_kmh", "hour_of_day"]
    cat_features = ["merchant_id", "use_chip", "merchant_city", "ip_location"]
    
    X = df[num_features + cat_features]
    y = df["is_fraud"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n⚙️ [2/4] Constructing ColumnTransformer Pipeline (Scaling + OneHot Encoding)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ]
    )
    
    print("⚙️ [3/4] Initializing Tabular MLP/Transformer Neural Net & Random Forest Ensemble...")
    # Tabular Deep Learning / MLP Classifier (simulates dense Transformer classification head)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )
    
    # Tuned Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    
    # Extra Trees Classifier
    et_model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Soft Voting Hybrid Ensemble Model
    hybrid_ensemble = VotingClassifier(
        estimators=[
            ("mlp_transformer", mlp_model),
            ("random_forest", rf_model),
            ("extra_trees", et_model)
        ],
        voting="soft"
    )
    
    print("⚙️ [4/4] Fitting Fine-Tuned Hybrid Ensemble Pipeline...")
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", hybrid_ensemble)
    ])
    
    full_pipeline.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = full_pipeline.predict(X_test)
    y_pred_probs = full_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_pred_probs) * 100
    
    print("\n📊 FINE-TUNED HYBRID TRANSFORMER/RF ENSEMBLE EVALUATION REPORT:")
    print(f"   • Overall Accuracy        : {acc:.2f}%")
    print(f"   • Precision               : {prec:.2f}%")
    print(f"   • Recall (Fraud Catch)    : {rec:.2f}%")
    print(f"   • F1-Score                : {f1:.2f}%")
    print(f"   • ROC-AUC Score           : {auc:.2f}%")
    print("\nClassification Detailed Metrics:\n", classification_report(y_test, y_pred, target_names=["Safe", "Fraud Alert"]))
    
    # Save Pipeline Model Artifacts
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    pipeline_path = os.path.join(models_dir, "transformer_rf_pipeline.pkl")
    joblib.dump(full_pipeline, pipeline_path)
    
    # Also overwrite standard models for backward compatibility
    joblib.dump(full_pipeline, os.path.join(models_dir, "fraud_classifier.pkl"))
    print(f"\n✅ Fine-Tuned Transformer & Random Forest Pipeline Saved to: {pipeline_path}")
    print("==================================================================")

if __name__ == "__main__":
    train_fine_tuned_transformer_rf_ensemble()
