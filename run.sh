#!/bin/bash

echo "=================================================="
echo "FinGuard AI - Starting Fraud Detection Architecture"
echo "=================================================="

echo "[1/2] Launching FastAPI Backend Server..."
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "[2/2] Launching Streamlit Dashboard..."
streamlit run app/dashboard.py &
DASHBOARD_PID=$!

echo "Both services launched!"
echo "- FastAPI Backend: http://127.0.0.1:8000"
echo "- Streamlit Dashboard: http://localhost:8501"

trap "kill $BACKEND_PID $DASHBOARD_PID" EXIT
wait
