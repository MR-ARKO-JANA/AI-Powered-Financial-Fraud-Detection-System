import os
import json
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# We will use the Flash model because it is incredibly fast, which is perfect for real-time transactions
MODEL_NAME = 'gemini-1.5-flash'

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
    
    # --- STEP 2: BUILD THE PROMPT FOR THE AI ---
    # We are writing a letter to the AI, giving it the facts and asking for JSON back.
    prompt = f"""
    You are a strictly analytical Fraud Detection AI. 
    Review this transaction data: {clean_data}
    Review these flagged indicators: {checklist}
    Calculated Base Risk Score: {risk_score}
    
    Classify this transaction into ONLY ONE of the following types:
    - Account Takeover
    - Card Skimming
    - Safe/Legitimate
    - UNKNOWN (Use this if the data is too weird or you are not confident)
    
    Return ONLY a valid JSON object in this exact format:
    {{"fraud_type": "Your Classification", "risk_score": {risk_score}, "reason": "One short sentence explaining why."}}
    """
    
    # --- STEP 3: CALL THE AI ---
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
            "risk_score": 50, 
            "reason": f"AI Inference failed or anomalous: {str(e)}"
        }