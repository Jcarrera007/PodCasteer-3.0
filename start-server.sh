#!/bin/bash

echo "🎙️  PodCasteer Whisper Server"
echo "=============================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Navigate to server directory
cd "$(dirname "$0")/server"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Start server
echo ""
echo "🚀 Starting Whisper Server..."
echo "   URL: ws://localhost:8765"
echo ""
echo "Press Ctrl+C to stop"
echo "=============================="
echo ""

python whisper_server.py
