import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# This tells the google library to look for the hidden key we will put in your .env file
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# We will use the Flash model because it is incredibly fast, which is perfect for real-time transactions
model = genai.GenerativeModel('gemini-1.5-flash')

def get_ai_risk_score(clean_data: dict, checklist: list) -> dict:
    """
    Takes the cleaned data and the rule indicators, calculates a score, 
    and asks the AI for a final classification.
    """
    
    # --- STEP 1: CALCULATE THE RISK SCORE ---
    # We need to count how many indicators are "PRESENT" or "PARTIAL"
    suspicious_count = 0
    for item in checklist:
        if "PRESENT" in item or "PARTIAL" in item:
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
        # Send the prompt to Gemini
        response = model.generate_content(prompt)
        
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