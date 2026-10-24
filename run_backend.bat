@echo off
echo ===================================================
echo Starting Employee Attrition Prediction Backend API
echo ===================================================
cd /d "%~dp0backend"
python app.py
if errorlevel 1 (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" app.py
)
pause
