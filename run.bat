@echo off
echo ==================================================
echo FinGuard AI - Starting Fraud Detection Architecture
echo ==================================================

echo [1/2] Launching FastAPI Backend Server...
start cmd /k "cd /d "%~dp0" && python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000"

echo [2/2] Launching Streamlit Dashboard...
start cmd /k "cd /d "%~dp0" && python -m streamlit run app/dashboard.py"

echo Both services launched successfully!
echo - Startup Presentation Site: http://127.0.0.1:8000/showcase
echo - FastAPI Backend API: http://127.0.0.1:8000
echo - Streamlit Dashboard: http://localhost:8501
pause
