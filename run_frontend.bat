@echo off
echo ===================================================
echo Starting Employee Attrition Prediction React App
echo ===================================================
cd /d "%~dp0frontend"
npm run dev
if errorlevel 1 (
    npm.cmd run dev
)
pause
