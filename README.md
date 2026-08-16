# 🛡️ FinGuard AI — Real-Time AI Financial Fraud Detection System

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.30%2B-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E)
![Google Gemini](https://img.shields.io/badge/Google--Gemini-GenAI-8E44AD)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-green)

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
- **🔐 Production Security**: API key authentication with RBAC, restricted CORS, sliding window rate limiting.
- **📡 Built-in Real-Time Live Monitor**: Stream transactions live directly inside the Streamlit dashboard or via the CLI streamer (`mock_stream.py`).
- **📈 Rich Risk Intelligence Dashboard**: Plotly charts for risk distributions, fraud typology pie charts, and aggregate KPI metrics.
- **🧪 Comprehensive Test Suite**: Full pytest coverage across preprocessing, rules engine, ML inference, database, alerts, and API.

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

### 2. Environment Setup

Copy the example environment file and configure:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | No | Google Gemini API key. Without it, FinGuard uses offline ML + Rules mode. |
| `GEMINI_MODEL` | No | Gemini model name (default: `gemini-2.5-flash`). |
| `FINGUARD_API_KEYS` | No | API key authentication. Format: `key1:admin,key2:analyst`. If not set, auth is disabled (dev mode). |
| `FINGUARD_CORS_ORIGINS` | No | Comma-separated allowed CORS origins (default: localhost ports). |

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

#### With Docker:
```bash
docker compose up --build
```

- **FastAPI REST API**: `http://127.0.0.1:8000`
- **Streamlit Interactive Dashboard**: `http://localhost:8501`

---

## 🔐 Security

### Authentication
When `FINGUARD_API_KEYS` is set, all API endpoints (except `/health`, `/`, and `/showcase`) require an `X-API-Key` header:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/v1/analyze -X POST -d '...'
```

### Roles
| Role | Permissions |
|---|---|
| `admin` | Full access: analyze, alerts, model refinement, log clearing |
| `analyst` | Analyze transactions, view/resolve alerts, view analytics |
| `viewer` | Read-only access to alerts and analytics |

### Rate Limiting
Built-in sliding window rate limiter protects against abuse:
- `/analyze`: 60 requests/minute
- `/analyze_batch`: 10 requests/minute
- `/model/refine`: 3 requests/5 minutes

---

## 🌐 API Reference

All endpoints are available at both `/endpoint` and `/v1/endpoint` (versioned).

| Endpoint | Method | Auth Required | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | No | Health check & system component status |
| `/v1/analyze` | `POST` | Yes | Process & score a single raw transaction |
| `/v1/analyze_batch` | `POST` | Yes | Process & score an array of transactions (max 100) |
| `/v1/analytics` | `GET` | Yes | Retrieve aggregate fraud KPIs & typologies |
| `/v1/alerts` | `GET` | Yes | View pending or resolved alerts |
| `/v1/alerts/resolve` | `POST` | Admin | Submit analyst feedback on alerts |
| `/v1/model/refine` | `POST` | Admin | Trigger model retraining |
| `/v1/logs/clear` | `POST` | Admin | Clear SQLite transaction log database |
| `/v1/metrics` | `GET` | Yes | Operational metrics for monitoring |

### Example Payload (`POST /v1/analyze`):

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

## 🧪 Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run specific test files:

```bash
python -m pytest tests/test_preprocess.py -v
python -m pytest tests/test_api.py -v
```

Test coverage areas:
- **Preprocessing**: Currency cleaning, alias normalization, IP geolocation, deduplication, input sanitization
- **Rules Engine**: Threshold detection, blacklist matching, behavioral analysis, score calculations
- **ML Inference**: Model loading, offline fallback, Gemini response parsing, risk scoring
- **Database**: CRUD operations, connection safety, health checks
- **Alert System**: Trigger thresholds, feedback loop, severity classification
- **API Integration**: All endpoints, validation, batch limits, request IDs, error handling

---

## 📡 Real-Time Live Streaming Simulation

To run an autonomous CLI live data stream into the backend:

```bash
python mock_stream.py
```

Or open **Tab 3 ("Live Stream Monitor")** inside the Streamlit Dashboard and toggle **Start Live Stream**.

---

## 📁 Project Structure

```
├── api/
│   └── main.py              # FastAPI application with auth, rate limiting, versioning
├── app/
│   └── dashboard.py          # Streamlit interactive dashboard
├── src/
│   ├── auth.py               # API key authentication & RBAC middleware
│   ├── rate_limiter.py        # Sliding window rate limiter
│   ├── logging_config.py      # Centralized structured JSON logging
│   ├── preprocess.py          # Data cleaning & normalization pipeline
│   ├── rules_engine.py        # Configurable rule-based fraud detection
│   ├── ml_inference.py        # ML + GenAI hybrid inference engine
│   ├── feature_engineering.py # Feature extraction for ML pipeline
│   ├── database.py            # SQLite storage with connection safety
│   └── alert_system.py        # Real-time alert trigger & model refinement
├── tests/
│   ├── conftest.py            # Pytest fixtures & test configuration
│   ├── test_preprocess.py     # Preprocessing pipeline tests
│   ├── test_rules_engine.py   # Rules engine tests
│   ├── test_ml_inference.py   # ML inference tests
│   ├── test_database.py       # Database operation tests
│   ├── test_alert_system.py   # Alert system tests
│   └── test_api.py            # API integration tests
├── configs/
│   └── fraud_typology.json    # Fraud detection rules configuration
├── models/                    # Trained ML model artifacts
├── data/                      # Transaction data & SQLite database
├── website/                   # Static showcase landing page
├── Dockerfile                 # Production Docker image (non-root, healthcheck)
├── docker-compose.yml         # Multi-service orchestration with volumes
├── requirements.txt           # Pinned Python dependencies
├── .env.example               # Environment variable template
├── .dockerignore              # Docker build exclusions
└── .gitignore                 # Comprehensive Git ignore rules
```

---

## 📄 License
Distributed under the MIT License.
