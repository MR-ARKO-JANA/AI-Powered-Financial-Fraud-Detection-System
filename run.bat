@echo off
echo =========================================
echo FinGuard AI - Starting Architecture Pipeline
echo =========================================

echo [1] Launching Local FastAPI Backend...
start cmd /k "cd "%~dp0" && uvicorn api.main:app --reload --host 127.0.0.1 --port 8000"

echo [2] Launching Streamlit Dashboard Application...
start cmd /k "cd "%~dp0" && streamlit run app/dashboard.py"

echo Both services triggered. Close these windows to stop.
pause
