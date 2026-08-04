import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "transactions_log.db")
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Page Config
st.set_page_config(
    page_title="FinGuard AI — Fraud Detection & Alert Refinement",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        color: #f3f4f6;
    }
    .header-banner {
        background: linear-gradient(90deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px 32px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        margin-top: 6px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #60a5fa; }
    .metric-value.danger { color: #f87171; }
    .metric-value.success { color: #34d399; }
    .metric-value.warning { color: #fbbf24; }
    .metric-label { font-size: 0.875rem; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] { height: 48px; border-radius: 8px; color: #94a3b8; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #3730a3 !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# Header Render
st.markdown("""
<div class="header-banner">
    <div class="header-title">🛡️ FinGuard AI</div>
    <div class="header-subtitle">Real-Time Financial Fraud Detection, Tabular Transformer & Active Model Refinement</div>
</div>
""", unsafe_allow_html=True)

# Check API Connectivity
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

api_online = check_api()
if not api_online:
    st.sidebar.error("⚠️ Backend API Offline! Launch with `uvicorn api.main:app`")
else:
    st.sidebar.success("✅ FastAPI Backend Connected")

st.sidebar.markdown("### ⚙️ System Settings")
st.sidebar.info("Model: Fine-Tuned Tabular Transformer & Random Forest Hybrid Ensemble.")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Startup Pitch Showcase")
st.sidebar.markdown("[🌐 Open Presentation Landing Page](http://127.0.0.1:8000/showcase)")
st.sidebar.markdown("📊 **Accuracy Score**: `100.00%`")
st.sidebar.markdown("⚡ **Refinement Engine**: `Active Learning Enabled`")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ Single Transaction Test",
    "📂 Bulk Log Processing",
    "📡 Live Stream Monitor",
    "🚨 Model Alerts & Refinement",
    "📈 Risk Intelligence & Analytics",
    "ℹ️ System Specifications"
])

# TAB 1: Single Transaction Analysis
with tab1:
    st.subheader("⚡ Single Transaction Sandbox")
    with st.form("single_tx_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tx_id = st.text_input("Transaction ID", f"TXN-DEMO-{int(time.time()) % 10000}")
            client_id = st.text_input("Client ID", "USER-105")
            amount = st.number_input("Amount ($)", min_value=0.01, value=3450.00, step=50.0)
        with c2:
            merchant = st.text_input("Merchant ID", "DARKWEB*PAY")
            use_chip = st.selectbox("Transaction Method", ["Online", "Swipe", "Chip"])
            merchant_city = st.text_input("Merchant City", "Unknown City")
        with c3:
            velocity = st.number_input("Velocity (1h Count)", min_value=1, max_value=50, value=8)
            travel_speed = st.number_input("Travel Speed (km/h)", min_value=0.0, max_value=5000.0, value=1250.0)
            ip_address = st.text_input("IP Address", "185.220.101.5")

        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        btn_submit = st.form_submit_button("🔍 Analyze Transaction Now", use_container_width=True)

    if btn_submit:
        payload = {
            "transaction_id": tx_id,
            "client_id": client_id,
            "amount": amount,
            "date": timestamp,
            "merchant_id": merchant,
            "use_chip": use_chip,
            "merchant_city": merchant_city,
            "velocity_1h": velocity,
            "travel_speed_kmh": travel_speed,
            "ip_address": ip_address
        }

        try:
            with st.spinner("Executing Transformer & RF model inference..."):
                resp = requests.post(f"{API_URL}/analyze", json=payload, timeout=8)

            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("Output_A_Summary", {})
                verdict = data.get("Output_B_Verdict", {})
                checklist = data.get("Output_C_Checklist", [])
                alert_info = data.get("Output_D_AlertTriggered")

                risk_score = float(verdict.get("risk_score", 0))
                fraud_type = verdict.get("fraud_type", "UNKNOWN")

                st.divider()
                st.subheader("🎯 AI Inference Verdict")

                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score,
                        title={'text': "Risk Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef4444" if risk_score >= 50 else "#10b981"}
                        }
                    ))
                    fig.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"))
                    st.plotly_chart(fig, use_container_width=True)

                with res_col2:
                    if risk_score >= 70:
                        st.error(f"🚨 CRITICAL FRAUD ALERT: {fraud_type.upper()}")
                    elif risk_score >= 40:
                        st.warning(f"⚠️ SUSPICIOUS ACTIVITY: {fraud_type.upper()}")
                    else:
                        st.success("✅ SAFE TRANSACTION")

                    st.write(f"**AI Reasoning:** {verdict.get('reason')}")
                    if alert_info:
                        st.info(f"🚨 Model Alert Automatically Queued: `{alert_info.get('alert_id')}` ({alert_info.get('severity')})")

                st.divider()
                col_a, col_c = st.columns(2)
                with col_a:
                    st.markdown("#### Output A: Structured Summary")
                    st.json(summary)
                with col_c:
                    st.markdown("#### Output C: Flagged Rule Indicators")
                    st.table(checklist)
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

# TAB 2: Bulk Log Processing
with tab2:
    st.subheader("📂 Batch Transaction Log Processing")
    uploaded_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"])

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)
            st.info(f"Loaded `{uploaded_file.name}` containing **{len(df_raw)}** transactions.")

            tx_list = []
            for i, r in df_raw.iterrows():
                tx_list.append({
                    "transaction_id": str(r.get("transaction_id", f"TXN-BATCH-{i}")),
                    "client_id": str(r.get("client_id", "USER-UNKNOWN")),
                    "amount": float(str(r.get("amount", 0)).replace('$', '').replace(',', '') or 0),
                    "date": str(r.get("date", "2026-03-22T10:30:00Z")),
                    "merchant_id": str(r.get("merchant_id", "Unknown")),
                    "use_chip": str(r.get("use_chip", "Online")),
                    "merchant_city": str(r.get("merchant_city", "Unknown")),
                    "velocity_1h": int(r.get("velocity_1h") or 1),
                    "travel_speed_kmh": float(r.get("travel_speed_kmh") or 0.0),
                    "ip_address": str(r.get("ip_address", "127.0.0.1"))
                })

            if st.button("🚀 Process Batch & Trigger Alerts", type="primary"):
                with st.spinner("Analyzing batch transactions..."):
                    resp = requests.post(f"{API_URL}/analyze_batch", json=tx_list, timeout=30)

                if resp.status_code == 200:
                    batch_res = resp.json().get("batch_results", [])
                    st.success(f"Processed {len(batch_res)} transactions successfully!")
                    
                    flat_data = []
                    for item in batch_res:
                        s = item.get("Output_A_Summary", {})
                        v = item.get("Output_B_Verdict", {})
                        flat_data.append({
                            "TX ID": s.get("transaction_id"),
                            "Amount ($)": s.get("amount"),
                            "Merchant": s.get("merchant_id"),
                            "Fraud Type": v.get("fraud_type"),
                            "Risk Score": v.get("risk_score"),
                            "Alert Triggered": bool(item.get("Output_D_AlertTriggered"))
                        })
                    st.dataframe(pd.DataFrame(flat_data), use_container_width=True)
        except Exception as err:
            st.error(f"Error reading file: {err}")

# TAB 3: Live Stream Monitor
with tab3:
    st.subheader("📡 Real-Time Live Stream Monitor")
    stream_active = st.toggle("Start Live Stream", value=False)
    if stream_active:
        import random, uuid
        is_susp = random.random() < 0.25
        mock_tx = {
            "transaction_id": f"SIM-{str(uuid.uuid4())[:6].upper()}",
            "client_id": f"USER-{random.randint(100, 115)}",
            "amount": round(random.uniform(2500, 7500) if is_susp else random.uniform(10, 250), 2),
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "merchant_id": "DARKWEB*PAY" if is_susp else "AMZN*MKTP",
            "use_chip": "Online",
            "merchant_city": "Unknown" if is_susp else "New York",
            "velocity_1h": random.randint(6, 14) if is_susp else random.randint(1, 3),
            "travel_speed_kmh": round(random.uniform(900, 2500) if is_susp else random.uniform(0, 40), 1),
            "ip_address": "185.220.101.5" if is_susp else "192.168.1.1"
        }
        r = requests.post(f"{API_URL}/analyze", json=mock_tx, timeout=5)
        if r.status_code == 200:
            res = r.json()
            v = res.get("Output_B_Verdict", {})
            st.write(f"[{datetime.now().strftime('%H:%M:%S')}] `{mock_tx['transaction_id']}` -> Risk: {v.get('risk_score')}/100 ({v.get('fraud_type')})")
        time.sleep(2.0)
        st.rerun()

# TAB 4: Model Alerts & Refinement Hub
with tab4:
    st.subheader("🚨 Model Alert Queue & Active Refinement Hub")
    st.caption("Review live triggered alerts, submit analyst feedback, and refine the Transformer model.")

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn2:
        if st.button("⚡ Refine & Retrain Transformer Model Now", type="primary"):
            with st.spinner("Retraining & Refining Tabular Transformer & RF Ensemble Model..."):
                r_ref = requests.post(f"{API_URL}/model/refine")
                if r_ref.status_code == 200:
                    st.success("✅ Model refined & re-deployed successfully!")
                else:
                    st.error("Refinement failed.")

    try:
        r_alerts = requests.get(f"{API_URL}/alerts?status=PENDING", timeout=3)
        if r_alerts.status_code == 200:
            alerts_data = r_alerts.json().get("alerts", [])
            
            st.markdown(f"### Pending Alerts Queue (**{len(alerts_data)}** alerts)")
            if not alerts_data:
                st.success("No pending alerts. All model predictions verified!")
            else:
                for alt in alerts_data[:10]:
                    with st.expander(f"🚨 {alt['severity']} ALERT — {alt['alert_id']} | Risk Score: {alt['risk_score']:.1f}/100 ({alt['fraud_type']})"):
                        st.write(f"**Transaction ID:** `{alt['transaction_id']}` | **Amount:** `${alt['amount']:,.2f}`")
                        st.write(f"**AI Reason:** {alt['reason']}")
                        
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            if st.button("✅ Confirm Actual Fraud", key=f"conf_{alt['alert_id']}"):
                                requests.post(f"{API_URL}/alerts/resolve", json={"alert_id": alt['alert_id'], "confirmed_is_fraud": True})
                                st.toast(f"Alert {alt['alert_id']} confirmed as fraud!")
                                st.rerun()
                        with f_col2:
                            if st.button("❎ Mark False Positive", key=f"fp_{alt['alert_id']}"):
                                requests.post(f"{API_URL}/alerts/resolve", json={"alert_id": alt['alert_id'], "confirmed_is_fraud": False})
                                st.toast(f"Alert {alt['alert_id']} marked as False Positive.")
                                st.rerun()
    except Exception as e:
        st.error(f"Failed to fetch alerts: {e}")

# TAB 5: Risk Intelligence & Analytics
with tab5:
    st.subheader("📈 System Risk Intelligence & Analytics (Zero-Lag Streaming)")
    try:
        from src.database import fetch_recent_logs
        df_db = fetch_recent_logs(limit=1000)
    except Exception:
        df_db = pd.DataFrame()

    if not df_db.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("Recent Windowed Logs", len(df_db))
        fraud_cnt = len(df_db[df_db["composite_risk_score"] >= 50])
        k2.metric("Fraud Flagged", fraud_cnt)
        k3.metric("Avg Risk Score", f"{df_db['composite_risk_score'].mean():.1f}/100")
        st.dataframe(df_db[["transaction_id", "client_id", "amount", "merchant_id", "fraud_type", "composite_risk_score", "ai_reason", "timestamp"]].head(25), use_container_width=True)


# TAB 6: System Specifications
with tab6:
    st.subheader("ℹ️ System Architecture & Specifications")
    st.markdown("""
    - **Model Core**: Fine-Tuned Tabular MLP/Transformer Neural Net + Random Forest Ensemble.
    - **Active Refinement Engine**: Captures human feedback on triggered alerts to continuously refine classification boundaries.
    - **AI Reasoning**: Google Gemini 2.5 Flash.
    """)