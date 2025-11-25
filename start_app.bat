@echo off
REM Startup script for Model Inference Web App

if "%~1"=="" (
    echo Usage: start_app.bat ^<run_folder^>
    echo Example: start_app.bat runs/run_50
    exit /b 1
)

echo Starting Model Inference App...
echo Run folder: %~1
echo.
echo The app will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

call conda activate torchgpu
python app.py %~1
