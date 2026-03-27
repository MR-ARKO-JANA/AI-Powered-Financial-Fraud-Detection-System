import streamlit as st
import requests

# 1. Configure the page layout and title
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ FinGuard AI")
st.subheader("AI Financial Fraud Detection System")
st.markdown("Upload transaction logs or enter details manually to detect fraudulent activity.")

# 2. Create tabs to organize the layout nicely
tab1, tab2 = st.tabs(["Manual Entry", "Bulk Upload (CSV/JSON)"])

# --- TAB 1: Single Transaction Testing ---
with tab1:
    st.subheader("Single Transaction Analysis")
    
    # Create a clean form for user input
    with st.form("single_transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            tx_id = st.text_input("Transaction ID", "TXN-1001")
            amount = st.number_input("Amount ($)", value=2500.00)
            timestamp = st.text_input("Timestamp", "2026-03-22T10:30:00Z")
        with col2:
            ip_address = st.text_input("IP Address", "192.168.1.50")
            merchant = st.text_input("Merchant Name", "AMZN*MKTP")
        
        # The submit button
        submit_button = st.form_submit_button("Run Analysis")

    # What happens when the user clicks the button:
    if submit_button:
        # Step A: Package the data exactly as your FastAPI contract expects
        payload = {
            "transaction_id": tx_id,
            "client_id": "USER-999",   
            "amount": amount,
            "date": timestamp,         
            "merchant_id": merchant,   
            "use_chip": "Online",      
            "merchant_city": "Unknown",
            "velocity_1h": 1,
            "travel_speed_kmh": 0.0
        }
        
        # Step B: Send it to your running FastAPI server
        try:
            with st.spinner("AI is analyzing transaction..."):
                # Notice it matches your FastAPI URL and route
                response = requests.post("http://127.0.0.1:8000/analyze", json=payload)
                
            if response.status_code == 200:
                result = response.json()
                
                st.divider()
                st.subheader("Analysis Results")
                
                # Output B: The Verdict (Color-coded!)
                verdict = result.get("Output_B_Verdict", {})
                if verdict.get("risk_score", 0) > 50:
                    st.error(f"🚨 FRAUD DETECTED: {verdict.get('fraud_type')} (Risk Score: {verdict.get('risk_score')}/100)")
                    st.write(f"**AI Reason:** {verdict.get('reason')}")
                else:
                    st.success(f"✅ Safe Transaction (Risk Score: {verdict.get('risk_score')}/100)")
                
                # Place Output A and Output C side-by-side using columns
                col_a, col_c = st.columns(2)
                with col_a:
                    st.write("**Output A: Structured Summary**")
                    st.json(result.get("Output_A_Summary", {}))
                    
                with col_c:
                    st.write("**Output C: Suspicious Indicators**")
                    st.table(result.get("Output_C_Checklist", []))
            else:
                st.error("An error occurred while processing the request.")
                
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to the backend. Is your FastAPI server running in another terminal?")

# --- TAB 2: File Uploading ---
with tab2:
    st.subheader("Batch Log Processing")
    st.write("Upload a messy log file to process multiple transactions at once.")
    
    # This creates the drag-and-drop file upload box
    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])
    
    if uploaded_file is not None:
        st.info(f"File '{uploaded_file.name}' uploaded successfully! (Batch processing will connect here once Arko finishes the data parser).")