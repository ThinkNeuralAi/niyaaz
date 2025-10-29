#!/bin/bash

# Sakshi.AI Startup Script
# This script helps start the application with proper environment setup

echo "🎯 Starting Sakshi.AI - Intelligent Video Analytics Platform"
echo "🏢 Powered by ThinkNeural.AI"
echo "========================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f ".requirements_installed" ]; then
    echo "📥 Installing requirements..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch .requirements_installed
        echo "✅ Requirements installed successfully"
    else
        echo "❌ Failed to install requirements"
        exit 1
    fi
fi

# Run setup if needed
if [ ! -f "data/sakshi.db" ]; then
    echo "🔧 Running initial setup..."
    python setup.py
fi

# Check if videos directory exists and has files
if [ ! -d "videos" ] || [ -z "$(ls -A videos)" ]; then
    echo "⚠️  Warning: No video files found in videos/ directory"
    echo "   Please add your video files (.mp4, .avi, .mov) to the videos/ directory"
    echo "   Example: cp /path/to/your/video.mp4 videos/"
fi

# Start the application
echo "🚀 Starting Sakshi.AI application..."
echo "🌐 Access the dashboard at: http://localhost:5000"
echo "📊 Press Ctrl+C to stop the application"
echo "========================================================"

python app.py