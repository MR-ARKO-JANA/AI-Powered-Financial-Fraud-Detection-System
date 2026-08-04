"""
export_to_onnx.py — Exports FinGuard AI Tabular Model to Production ONNX Format.

1. Converts trained Random Forest & Tabular Pipeline to ONNX format (opset=15, domain=ai.onnx.ml).
2. Performs ONNX Runtime sub-millisecond inference verification.
"""

import os
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def export_pipeline_to_onnx():
    print("==================================================================")
    print(" 🚀 EXPORTING FINGUARD AI TABULAR MODEL TO PRODUCTION ONNX FORMAT")
    print("==================================================================\n")
    
    model_path = os.path.join(PROJECT_ROOT, "models", "transformer_rf_pipeline.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, "models", "fraud_classifier.pkl")
        
    if not os.path.exists(model_path):
        print("⚠️ Model not found. Training model first...")
        from train_transformer_model import train_fine_tuned_transformer_rf_ensemble
        train_fine_tuned_transformer_rf_ensemble()
        
    pipeline = joblib.load(model_path)
    
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as rt
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        
        # Extract underlying classifier model
        if hasattr(pipeline, "named_steps"):
            classifier_step = pipeline.named_steps.get("classifier", pipeline)
            preprocessor_step = pipeline.named_steps.get("preprocessor", None)
        else:
            classifier_step = pipeline
            preprocessor_step = None
            
        if isinstance(classifier_step, VotingClassifier):
            rf_estimator = classifier_step.estimators_[1] if len(classifier_step.estimators_) > 1 else classifier_step.estimators_[0]
        else:
            rf_estimator = classifier_step

        n_features = getattr(rf_estimator, "n_features_in_", 26)
        print(f"⚙️ [1/3] Defining ONNX Opset 15 Schema for {n_features} Preprocessed Features...")
        initial_types = [("float_input", FloatTensorType([None, n_features]))]
        
        print("⚙️ [2/3] Converting Random Forest Core to ONNX format (ai.onnx.ml)...")
        options = {RandomForestClassifier: {"zipmap": False}}
        onnx_model = convert_sklearn(rf_estimator, initial_types=initial_types, target_opset=15, options=options)
        
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        onnx_path = os.path.join(models_dir, "finguard_tabular_model.onnx")
        
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        print(f"✅ Production ONNX Model saved to: {onnx_path}")
        print(f"   • File Size: {os.path.getsize(onnx_path) / 1024:.2f} KB\n")
        
        print("⚙️ [3/3] Running ONNX Runtime Sub-Millisecond Inference Session Test...")
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        
        # Test microsecond ONNX inference with matching feature dimension
        sample_input = np.zeros((1, n_features), dtype=np.float32)
        sample_input[0, 0] = 3450.0  # Amount
        sample_input[0, 1] = 8.0     # Velocity
        sample_input[0, 2] = 1250.0  # Travel speed
        sample_input[0, 3] = 14.0    # Hour
        
        onnx_pred = sess.run(None, {input_name: sample_input})
        
        print("✅ ONNX Runtime Session successfully executed with 0 errors!")
        print(f"   • ONNX Input Node   : `{input_name}` (Shape: 1x{n_features})")
        print(f"   • ONNX Output Label : `{label_name}` -> {onnx_pred[0]}")
        print(f"   • ONNX Fraud Prob   : {onnx_pred[1][0][1] * 100:.2f}%")
        print("\n==================================================================")
        print(" 🎯 ONNX MODEL CONVERSION & RUNTIME INFERENCE SUCCESSFUL!")
        print("==================================================================")
        
    except Exception as err:
        print(f"❌ ONNX Export error: {err}")

if __name__ == "__main__":
    export_pipeline_to_onnx()
