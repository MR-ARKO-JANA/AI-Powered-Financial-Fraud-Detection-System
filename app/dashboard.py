import streamlit as st
import requests
import pandas as pd
import sqlite3
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "transactions_log.db")

# 1. Configure the page layout and title
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ FinGuard AI")
st.subheader("AI Financial Fraud Detection System")
st.markdown("Upload transaction logs or enter details manually to detect fraudulent activity.")

# 2. Create tabs to organize the layout nicely
tab1, tab2, tab3, tab4 = st.tabs(["Manual Entry", "Bulk Upload (CSV/JSON)", "About FinGuard AI", "Live Analytics"])

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
            "travel_speed_kmh": 0.0,
            "ip_address": ip_address
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
        st.info(f"File '{uploaded_file.name}' uploaded successfully! Processing batch logs...")
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
                
            st.write(f"Detected {len(df)} transactions in file.")
            
            # Formatting the dataframe to matching RawTransaction BaseModel dicts
            tx_list = []
            for _, row in df.iterrows():
                # Provide defaults if cols are missing
                tx = {
                    "transaction_id": str(row.get("transaction_id", f"TXN-BATCH-{_}")),
                    "client_id": str(row.get("client_id", "USER-UNKNOWN")),
                    "amount": float(str(row.get("amount", 0)).replace('$', '').replace(',', '') or 0),
                    "date": str(row.get("date", "2026-03-22T10:30:00Z")),
                    "merchant_id": str(row.get("merchant_id", "Unknown")),
                    "use_chip": str(row.get("use_chip", "Online")),
                    "merchant_city": str(row.get("merchant_city", "Unknown")),
                    "velocity_1h": int(row.get("velocity_1h") or 1),
                    "travel_speed_kmh": float(row.get("travel_speed_kmh") or 0.0),
                    "ip_address": str(row.get("ip_address", "Unknown"))
                }
                tx_list.append(tx)
                
            with st.spinner("AI is analyzing batch transactions..."):
                response = requests.post("http://127.0.0.1:8000/analyze_batch", json=tx_list)
                
            if response.status_code == 200:
                result_data = response.json()
                if "error" in result_data:
                    st.error(result_data["error"])
                else:
                    batch_results = result_data.get("batch_results", [])
                    st.success("Batch processing complete!")
                    
                    # Prepare data for rendering
                    disp_data = []
                    for res in batch_results:
                        summary = res.get("Output_A_Summary", {})
                        verdict = res.get("Output_B_Verdict", {})
                        disp_data.append({
                            "TX ID": summary.get("transaction_id"),
                            "Amount": summary.get("amount"),
                            "Fraud Type": verdict.get("fraud_type"),
                            "Risk Score": verdict.get("risk_score"),
                            "Reason": verdict.get("reason")
                        })
                    
                    st.dataframe(pd.DataFrame(disp_data), use_container_width=True)
            else:
                st.error("Failed to process batch payload via API.")

        except Exception as e:
            # Fixed typo here
            st.error(f"Error reading or parsing file: {e}")

# --- TAB 3: About FinGuard AI ---
with tab3:
    st.header("About FinGuard AI")
    st.markdown("""
    ### Overview
    FinGuard AI is an AI-powered financial fraud detection system designed to analyze transaction logs, detect suspicious activities, and classify potential fraud in real-time. 

    ### The Problem
    As digital transactions increase, so do sophisticated fraud patterns like **Account Takeovers** and **Card Skimming**. Traditional systems are often too slow or unable to detect these complex behaviors.

    ### Our Objective
    To build a modular AI/ML pipeline that:
    1.  **Processes** messy and inconsistent transaction data.
    2.  **Extracts** meaningful behavioral indicators.
    3.  **Detects** suspicious patterns (velocity, location, device changes).
    4.  **Classifies** fraud types with interpretable risk scores.

    ### System Workflow
    - **Inbound Logs**: Receives CSV or JSON data with transaction and device details.
    - **Smart Preprocessing**: Cleans data and extracts locations from IP addresses.
    - **Feature Engineering**: Compares current behavior with historical per-client norms.
    - **AI Inference**: Uses Google's Gemini Flash model for high-speed fraud classification.
    - **Checklist Output**: Provides a transparent **PRESENT/MISSING/PARTIAL** breakdown of all fraud indicators.

    ### Built with ❤️
    FastAPI • Streamlit • Pandas • Google Gemini AI • Pydantic
    """)

# --- TAB 4: Live Analytics ---
with tab4:
    st.header("📈 Live Analytics Dashboard")
    st.markdown("Monitor system performance, view historical transactions, and analyze fraud trends.")
    
    # Auto-refresh button
    col_a, col_b = st.columns([8, 1])
    with col_b:
        if st.button("🔄 Refresh Data"):
            st.rerun()
            
    # Load Data from SQLite
    try:
        conn = sqlite3.connect(DB_PATH)
        df_logs = pd.read_sql_query("SELECT * FROM transaction_logs", conn)
        conn.close()
    except Exception as e:
        df_logs = pd.DataFrame()
        st.warning(f"Database not initialized or empty. Error: {e}")

    if not df_logs.empty:
        # Convert timestamp to datetime
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        
        # Calculate KPIs
        total_tx = len(df_logs)
        fraud_df = df_logs[df_logs['composite_risk_score'] > 50]
        total_fraud = len(fraud_df)
        fraud_prevented_amt = fraud_df['amount'].sum()
        avg_risk = df_logs['composite_risk_score'].mean()
        
        # Display KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Analyzed", f"{total_tx}")
        kpi2.metric("Fraud Detected", f"{total_fraud}", delta=f"{(total_fraud/total_tx*100):.1f}%", delta_color="inverse")
        kpi3.metric("Funds Protected", f"${fraud_prevented_amt:,.2f}")
        kpi4.metric("Avg Risk Score", f"{avg_risk:.1f}/100")
        
        st.divider()
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Risk Score Distribution")
            st.bar_chart(df_logs['composite_risk_score'].value_counts(bins=10).sort_index())
            
        with chart_col2:
            st.subheader("Transaction Volumes (Recent)")
            # Group by hour or just show raw sequence if too few
            if len(df_logs) > 10:
                time_series = df_logs.set_index('timestamp').resample('min').size()
                st.line_chart(time_series)
            else:
                st.write("Not enough data over time for a trend line yet.")
                
        st.subheader("Recent Flagged Transactions")
        st.dataframe(fraud_df[['transaction_id', 'amount', 'merchant_id', 'fraud_type', 'composite_risk_score', 'timestamp']].tail(10), use_container_width=True)
    else:
        st.info("Waiting for transactions... Process some data in Tab 1 or 2 to see analytics!")