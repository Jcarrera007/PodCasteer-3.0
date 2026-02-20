@echo off
echo 🎙️  PodCasteer Whisper Server
echo ==============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.9 or higher.
    exit /b 1
)

REM Navigate to server directory
cd /d "%~dp0server"

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -q -r requirements.txt

REM Start server
echo.
echo 🚀 Starting Whisper Server...
echo    URL: ws://localhost:8765
echo.
echo Press Ctrl+C to stop
echo ==============================
echo.

python whisper_server.py
