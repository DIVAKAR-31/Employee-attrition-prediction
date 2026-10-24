@echo off
echo ==========================================================
echo Starting Employee Attrition Prediction Full Stack System
echo ==========================================================
start "HR Attrition Flask Backend" run_backend.bat
timeout /t 3 /nobreak >nul
start "HR Attrition React Frontend" run_frontend.bat
echo Both services launched! Opening web browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://localhost:5173/
