import json
import pandas as pd
from typing import Dict, Any, List

def load_config(config_path: str = "configs/fraud_typology.json") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)

def apply_rules(df: pd.DataFrame, config: Dict[str, Any], historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies fraud typology rules to generate indicators.
    Historical DF can be provided to compare against user's historical norms.
    """
    df = df.copy()
    
    thresholds = config.get("thresholds", {})
    lists = config.get("lists", {})
    behavioral = config.get("behavioral_rules", {})
    
    # Use historical_df if provided, else use df itself as reference for norms
    context_df = historical_df if historical_df is not None else df
    
    # Compute base norms (Location, Device, Avg Amount) per user over the context window
    user_norms = {}
    if not context_df.empty:
        for sender, group in context_df.groupby('sender'):
            user_norms[sender] = {
                'known_devices': set(group['device'].dropna().astype(str).str.lower().unique()),
                'known_locations': set(group['location'].dropna().astype(str).str.lower().unique()),
                'avg_amount': group['amount'].mean() if len(group['amount']) > 0 else 0.0
            }

    df['suspicious_indicators_checklist'] = [[] for _ in range(len(df))]
    
    def check_row(row):
        indicators = []
        
        # 1. Static Thresholds
        if pd.notna(row.get('amount')) and row.get('amount') > thresholds.get('high_amount', 10000.0):
            indicators.append("High Transaction Amount")
            
        if pd.notna(row.get('velocity_1h')) and row.get('velocity_1h') > thresholds.get('high_velocity_count', 5):
            indicators.append("High Transaction Velocity")
            
        if pd.notna(row.get('travel_speed_kmh')) and row.get('travel_speed_kmh') > thresholds.get('impossible_travel_speed_kmh', 1000.0):
            indicators.append("Impossible Travel Speed")
            
        # 2. Lists Matching
        if str(row.get('receiver')).lower() in lists.get('blacklisted_merchants', []):
            indicators.append("Blacklisted Merchant")
            
        dev = str(row.get('device')).lower()
        if dev in lists.get('suspicious_devices', []):
            indicators.append("Suspicious Device")
            
        # 3. Behavioral Logic - Compare to historical norms (PRESENT, MISSING, PARTIAL)
        sender = row.get('sender')
        norms = user_norms.get(sender, {'known_devices': set(), 'known_locations': set(), 'avg_amount': 0.0})
        
        dev_val = str(row.get('device')).lower() if pd.notna(row.get('device')) else ""
        if behavioral.get('check_new_device', True):
            if not dev_val or dev_val == 'nan' or dev_val == '':
                indicators.append("New Device: PARTIAL (Missing Data)")
            elif not norms['known_devices']:
                indicators.append("New Device: PARTIAL (No Prior History)")
            elif dev_val not in norms['known_devices']:
                indicators.append("New Device: PRESENT")
            else:
                indicators.append("New Device: MISSING")
                
        loc_val = str(row.get('location')).lower() if pd.notna(row.get('location')) else ""
        if behavioral.get('check_location_change', True):
            if not loc_val or loc_val == 'nan' or loc_val == '':
                indicators.append("Location Change: PARTIAL (Missing Data)")
            elif not norms['known_locations']:
                indicators.append("Location Change: PARTIAL (No Prior History)")
            elif loc_val not in norms['known_locations']:
                indicators.append("Location Change: PRESENT")
            else:
                indicators.append("Location Change: MISSING")
                
        amt = row.get('amount', 0.0)
        amt_multiplier = behavioral.get('amount_anomaly_multiplier', 3.0)
        
        # Only flag amount if it's statistically much higher than baseline
        if pd.notna(amt) and norms['avg_amount'] > 0:
            if amt > norms['avg_amount'] * amt_multiplier:
                indicators.append("Amount Anomaly: PRESENT")
                
        return indicators

    df['suspicious_indicators_checklist'] = df.apply(check_row, axis=1)
    
    # Calculate Ground Truth Risk Score (0-100) based on severity of signals
    def calc_risk(inds):
        score = 0
        for i in inds:
            if "PRESENT" in i or "High" in i or "Impossible" in i or "Blacklisted" in i or "Suspicious" in i:
                score += 25
            elif "PARTIAL" in i:
                score += 10
        return min(score, 100)
        
    df['rule_risk_score'] = df['suspicious_indicators_checklist'].apply(calc_risk)
    
    return df
