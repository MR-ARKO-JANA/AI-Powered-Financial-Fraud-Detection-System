import os
import json
import joblib
import numpy as np
from google import genai
from dotenv import load_dotenv
from src.feature_engineering import prepare_ml_features

# Load environment variables from .env file
load_dotenv()

# We will use the Flash model because it is incredibly fast, which is perfect for real-time transactions
MODEL_NAME = 'gemini-1.5-flash'

# Load Local ML Model
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    ml_classifier = joblib.load(os.path.join(PROJECT_ROOT, "models", "fraud_classifier.pkl"))
except Exception as e:
    print(f"Warning: Could not load local ML model: {e}")
    ml_classifier = None

def get_ai_risk_score(clean_data: dict, checklist: list) -> dict:
    """
    Takes the cleaned data and the rule indicators, calculates a score, 
    and asks the AI for a final classification.
    """
    
    # --- STEP 1: CALCULATE THE RISK SCORE ---
    # Count how many indicators denote elevated risk.
    suspicious_count = 0
    for item in checklist:
        if any(kw in item for kw in ["PRESENT", "PARTIAL", "High", "Impossible", "Blacklisted", "Suspicious"]):
            suspicious_count += 1
            
    total_indicators = len(checklist) if len(checklist) > 0 else 1 # Avoid dividing by zero!
    
    # The mathematical formula requested by the project instructions
    risk_score = (suspicious_count / total_indicators) * 100
    
    # --- STEP 2: RUN LOCAL MACHINE LEARNING MODEL ---
    ml_fraud_probability = 0.0
    if ml_classifier is not None:
        try:
            X_scaled = prepare_ml_features(clean_data)
            # predict_proba returns [[prob_0, prob_1]]
            ml_fraud_probability = ml_classifier.predict_proba(X_scaled)[0][1] * 100
        except Exception as e:
            print(f"Local ML inference failed: {e}")
            
    # Combine Scores for the Prompt
    composite_base_score = (risk_score * 0.5) + (ml_fraud_probability * 0.5)
    
    # --- STEP 3: BUILD THE PROMPT FOR THE AI ---
    # We are writing a letter to the AI, giving it the facts and asking for JSON back.
    prompt = f"""
    You are a strictly analytical Fraud Detection AI. 
    Review this transaction data: {clean_data}
    Review these flagged indicators from the Rule Engine: {checklist}
    
    Calculated Metrics:
    - Base Rules Risk Score: {risk_score:.2f}/100
    - Local Machine Learning Probability (Random Forest): {ml_fraud_probability:.2f}%
    - Composite Baseline Score: {composite_base_score:.2f}/100
    
    Classify this transaction into ONLY ONE of the following types:
    - Account Takeover
    - Card Skimming
    - Safe/Legitimate
    - UNKNOWN (Use this if the data is too weird or you are not confident)
    
    Return ONLY a valid JSON object in this exact format:
    {{"fraud_type": "Your Classification", "risk_score": {composite_base_score:.2f}, "reason": "One short sentence explaining why."}}
    """
    
    # --- STEP 4: CALL THE AI ---
    try:
        # Initialize client here to easily pick up the env key
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
        # Send the prompt to Gemini
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        
        # Clean up the response to make sure it's pure JSON
        raw_text = response.text.strip().replace("```json", "").replace("```", "")
        ai_verdict = json.loads(raw_text)
        
        return ai_verdict
        
    except Exception as e:
        # Fallback Logic: If the AI fails or the internet disconnects, return UNKNOWN
        return {
            "fraud_type": "UNKNOWN", 
            "risk_score": float(composite_base_score), 
            "reason": f"AI Inference failed or anomalous: {str(e)}"
        }
