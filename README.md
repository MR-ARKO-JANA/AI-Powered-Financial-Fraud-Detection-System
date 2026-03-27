# FinGuard AI: AI-Powered Financial Fraud Detection System

## Overview
FinGuard AI is an intelligent system capable of automatically analyzing transaction data, detecting suspicious patterns, and identifying potential fraud in real-time. 

With the rapid increase in digital financial transactions (online banking, UPI, digital wallets), financial fraud has become more sophisticated. Traditional rule-based systems often struggle to handle the scale and complexity of modern fraud. FinGuard AI addresses this by providing an automated AI/ML pipeline to process messy logs, detect behavioral anomalies, and classify fraud into actionable categories.

## 🛡️ Key Features
- **Data Ingestion & Preprocessing**: Automatically cleans messy transaction logs (JSON/CSV), standardizes fields, and extracts location data from IP addresses.
- **Behavioral Feature Extraction**: Identifies sudden location changes, device swaps, high transaction velocity, and impossible travel speeds.
- **AI-Powered Classification**: Uses machine learning to categorize transactions into:
    - **Account Takeover**
    - **Card Skimming / Cloning**
    - **Safe / Legitimate**
    - **Unknown Suspicious Activity**
- **Risk Scoring**: Assigns an interpretable score (0-100) based on the ratio of detected suspicious indicators.
- **Suspicious Indicator Checklist**: Provides a transparent breakdown of why a transaction was flagged (PRESENT, MISSING, or PARTIAL).

## 📊 System Workflow
1. **Input**: Transaction logs (JSON/CSV) containing transaction IDs, timestamps, amounts, device IDs, and IP addresses.
2. **Preprocessing**: Handing missing values, standardizing merchant names (e.g., merging `AMZN*MKTP` into `AMAZON`), and date parsing.
3. **Feature Extraction**: Building behavioral norms (per-client history) to identify anomalies like "Amount Anomaly" or "New Device".
4. **Fraud Detection Model**: Passing structured features to the ML inference engine for classification and risk scoring.
5. **Output**: Structured summary, fraud classification, and a detailed indicators checklist.

## 🚀 Getting Started
### 1. Installation
Ensure you have a Python environment set up, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your Gemini API Key:
```env
GEMINI_API_KEY="your_actual_key_here"
```

### 3. Launching the System
Run the provided batch script to start both the backend and frontend:
```bash
./run.bat
```
- **FastAPI Backend**: `http://127.0.0.1:8000`
- **Streamlit Dashboard**: `http://localhost:8501`

## 🛠️ Built With
- **FastAPI**: Backend pipeline orchestration and REST API.
- **Streamlit**: Interactive dashboard for manual entry and batch log processing.
- **Pandas**: Data cleaning and feature engineering.
- **Google Gemini API**: Advanced AI inference for fraud classification and reasoning.
- **Pydantic**: Data validation and contract enforcement.

## 🔮 Future Improvements
- **Explainable AI**: Real-time natural language explanations for fraud detection decisions.
- **Network Analysis**: Using graph analysis to detect connections between suspicious wallets/accounts.
- **Live Monitoring**: Real-time streaming dashboards for financial security teams.

---
*Developed as part of the Final Year Project: AI-Powered Financial Fraud Detection System.*
