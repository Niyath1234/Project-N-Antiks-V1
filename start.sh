#!/bin/bash
# Quick start script for SQL Analytics AI

cd "$(dirname "$0")"

echo "🚀 Starting SQL Analytics AI"
echo ""

# Check venv
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check Ollama
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "⚠️  Ollama not running. Please start it in another terminal:"
    echo "   ollama serve"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CSV
if [ -z "$CSV_PATH" ]; then
    echo "⚠️  CSV_PATH not set"
    echo "Set it now or use ':load <path>' in the app"
    echo ""
fi

# Run app
python base_v2.py

