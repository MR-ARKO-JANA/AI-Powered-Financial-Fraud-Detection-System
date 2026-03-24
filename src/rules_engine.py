import json
import pandas as pd
from typing import Dict, Any, List

def load_config(config_path: str = "configs/fraud_typology.json") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)

def apply_rules(df: pd.DataFrame, config: Dict[str, Any], historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies fraud typology rules to generate indicators.
    Historical DF can be provided to pre-build chronological norms.
    """
    df = df.copy()
    
    thresholds = config.get("thresholds", {})
    lists = config.get("lists", {})
    behavioral = config.get("behavioral_rules", {})
    
    user_norms = {}
    
    # 1. Pre-fill norms using ONLY historical_df (fixes the self-inclusion bug)
    if historical_df is not None and not historical_df.empty:
        for client_id, group in historical_df.groupby('client_id'):
            user_norms[client_id] = {
                'known_methods': set(group['use_chip'].dropna().astype(str).str.lower().unique()),
                'known_locations': set(group['merchant_city'].dropna().astype(str).str.lower().unique()),
                'sum_amount': group['amount'].sum(),
                'count_amount': len(group['amount'])
            }
            if user_norms[client_id]['count_amount'] > 0:
                user_norms[client_id]['avg_amount'] = user_norms[client_id]['sum_amount'] / user_norms[client_id]['count_amount']
            else:
                user_norms[client_id]['avg_amount'] = 0.0

    df['suspicious_indicators_checklist'] = [[] for _ in range(len(df))]
    
    # Sort chronologically if date exists, so we build history properly across the batch
    if 'date' in df.columns:
        df = df.sort_values(by='date')
    
    def check_row(row):
        indicators = []
        client_id = row.get('client_id')
        
        # Initialize norms for a new user we haven't seen yet
        if client_id not in user_norms:
            user_norms[client_id] = {'known_methods': set(), 'known_locations': set(), 'sum_amount': 0.0, 'count_amount': 0, 'avg_amount': 0.0}
            
        norms = user_norms[client_id]
        
        # 1. Static Thresholds
        amt = row.get('amount', 0.0)
        if pd.notna(amt) and isinstance(amt, (int, float)) and amt > thresholds.get('high_amount', 10000.0):
            indicators.append("High Transaction Amount")
            
        if pd.notna(row.get('velocity_1h')) and row.get('velocity_1h') > thresholds.get('high_velocity_count', 5):
            indicators.append("High Transaction Velocity")
            
        if pd.notna(row.get('travel_speed_kmh')) and row.get('travel_speed_kmh') > thresholds.get('impossible_travel_speed_kmh', 1000.0):
            indicators.append("Impossible Travel Speed")
            
        # 2. Lists Matching
        merch = str(row.get('merchant_id')).lower() if pd.notna(row.get('merchant_id')) else ""
        if merch in lists.get('blacklisted_merchants', []):
            indicators.append("Blacklisted Merchant")
            
        method_val = str(row.get('use_chip')).lower() if pd.notna(row.get('use_chip')) else ""
        if method_val in lists.get('suspicious_devices', []):
            indicators.append("Suspicious Device")
            
        # 3. Behavioral Logic - Compare to historical norms (PRESENT, MISSING, PARTIAL)
        if behavioral.get('check_new_device', True):
            if not method_val or method_val == 'nan' or method_val == '':
                indicators.append("New Device: PARTIAL (Missing Data)")
            elif not norms['known_methods']:
                indicators.append("New Device: PARTIAL (No Prior History)")
            elif method_val not in norms['known_methods']:
                indicators.append("New Device: PRESENT")
            else:
                indicators.append("New Device: MISSING")
                
        loc_val = str(row.get('merchant_city')).lower() if pd.notna(row.get('merchant_city')) else ""
        if behavioral.get('check_location_change', True):
            if not loc_val or loc_val == 'nan' or loc_val == '':
                indicators.append("Location Change: PARTIAL (Missing Data)")
            elif not norms['known_locations']:
                indicators.append("Location Change: PARTIAL (No Prior History)")
            elif loc_val not in norms['known_locations']:
                indicators.append("Location Change: PRESENT")
            else:
                indicators.append("Location Change: MISSING")
                
        amt_multiplier = behavioral.get('amount_anomaly_multiplier', 3.0)
        
        # Only flag if amount is statistically much higher than baseline
        if pd.notna(amt) and isinstance(amt, (int, float)) and norms['avg_amount'] > 0:
            if amt > norms['avg_amount'] * amt_multiplier:
                indicators.append("Amount Anomaly: PRESENT")
                
        # 4. Update the running norms for the NEXT row to fix the self-inclusion bug
        if method_val and method_val != 'nan':
            norms['known_methods'].add(method_val)
        if loc_val and loc_val != 'nan':
            norms['known_locations'].add(loc_val)
        if pd.notna(amt) and isinstance(amt, (int, float)):
            norms['sum_amount'] += amt
            norms['count_amount'] += 1
            norms['avg_amount'] = norms['sum_amount'] / norms['count_amount']
            
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
    
    # If we sorted by date, re-sort it back using original index
    if 'date' in df.columns:
        df = df.sort_index()
    
    return df
