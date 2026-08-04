# 🛡️ FinGuard AI — Real-Time AI Financial Fraud Detection System

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.30%2B-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E)
![Google Gemini](https://img.shields.io/badge/Google--Gemini-GenAI-8E44AD)
![License](https://img.shields.io/badge/license-MIT-green)

FinGuard AI is an enterprise-ready intelligent financial fraud detection pipeline that automatically processes raw transaction logs, extracts behavioral anomalies, executes rule-based static & dynamic thresholding, runs local Random Forest machine learning classification, and generates explainable natural language fraud verdicts via Google Gemini GenAI.

---

## 🌟 Key Features

- **⚡ Real-Time & Batch Analysis**: Single transaction sandbox and bulk CSV/JSON file log processor.
- **🔍 Smart Data Preprocessing**: Currency string cleaning (`$`, `,`), alias normalization (`AMZN*MKTP` → `AMAZON`), and IP-to-Geolocation resolution.
- **📊 Multi-Layer Behavioral Profiling**:
  - Precomputes per-user historical norms (30-day average transaction amount, known devices, known cities).
  - High velocity detection (>5 tx/hr), impossible travel speed (>1,000 km/h), blacklisted merchant flags.
- **🤖 Hybrid ML & GenAI Inference**:
  - Local **Random Forest Classifier** trained on synthetic behavioral transaction data.
  - **Google Gemini API** (`google-genai` SDK) for structured natural language fraud reasoning.
  - Full **offline fallback mode** to operate without external API keys.
- **📡 Built-in Real-Time Live Monitor**: Stream transactions live directly inside the Streamlit dashboard or via the CLI streamer (`mock_stream.py`).
- **📈 Rich Risk Intelligence Dashboard**: Plotly charts for risk distributions, fraud typology pie charts, and aggregate KPI metrics.

---

## 🏗️ System Architecture

```
                               ┌───────────────────────────┐
                               │ Raw Transaction Payload   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Data Preprocessing      │
                               │ (IP Geoloc & Alias Clean) │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     Rules Engine          │
                               │ (Behavioral Norm Check)   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Random Forest Classifier │
                               │   (Scikit-Learn Model)    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Google Gemini GenAI      │
                               │  (Explainable Reasoning)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ SQLite DB Log & Analytics │
                               └───────────────────────────┘
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/MR-ARKO-JANA/AI-Powered-Financial-Fraud-Detection-System.git
cd AI-Powered-Financial-Fraud-Detection-System
pip install -r requirements.txt
```

### 2. Environment Setup (Optional for GenAI Reasoning)

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="your_google_gemini_api_key_here"
```

*Note: If no API key is set, FinGuard AI automatically runs in hybrid Random Forest + Rules fallback mode.*

### 3. Training / Generating Synthetic Models & Data

Train the local ML classifier and generate the seed dataset:

```bash
python train_model.py
```

### 4. Launching the Application

#### On Windows:
```cmd
run.bat
```

#### On Linux / macOS:
```bash
chmod +x run.sh
./run.sh
```

- **FastAPI REST API**: `http://127.0.0.1:8000`
- **Streamlit Interactive Dashboard**: `http://localhost:8501`

---

## 🌐 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Health check & system component status |
| `/analyze` | `POST` | Process & score a single raw transaction |
| `/analyze_batch` | `POST` | Process & score an array of transactions |
| `/analytics` | `GET` | Retrieve aggregate fraud KPIs & typologies |
| `/logs/clear` | `POST` | Clear SQLite transaction log database |

### Example Payload (`POST /analyze`):

```json
{
  "transaction_id": "TXN-9001",
  "client_id": "USER-105",
  "amount": 4200.00,
  "date": "2026-03-22T10:30:00Z",
  "merchant_id": "DARKWEB*PAY",
  "use_chip": "Online",
  "merchant_city": "Unknown",
  "velocity_1h": 9,
  "travel_speed_kmh": 1400.0,
  "ip_address": "185.220.101.5"
}
```

---

## 🧪 Real-Time Live Streaming Simulation

To run an autonomous CLI live data stream into the backend:

```bash
python mock_stream.py
```

Or open **Tab 3 ("Live Stream Monitor")** inside the Streamlit Dashboard and toggle **Start Live Stream**.

---

## 📄 License
Distributed under the MIT License.
