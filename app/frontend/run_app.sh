#!/bin/bash

# Test script for the Multi-Agent Doc Research applications

echo "=== Multi-Agent Doc Research Test Script ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    exit 1
fi

echo "✅ Python3 is available"

# Check if uv is available  
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✅ uv is available"

# Check if dependencies are installed
echo "📦 Checking dependencies..."

# Check if environment variables are set
echo "🔧 Checking environment variables..."

if [ -f .env ]; then
    echo "✅ .env file found"
    source .env
else
    echo "⚠️  .env file not found. Using default values."
    export API_URL="http://localhost:8000/plan_search"
fi

echo "API_URL: $API_URL"

# Test connection to backend
echo ""
echo "🔗 Testing backend connection..."

if curl -s --connect-timeout 5 "$API_URL" > /dev/null 2>&1; then
    echo "✅ Backend is reachable at $API_URL"
else
    echo "⚠️  Backend is not reachable at $API_URL"
    echo "   Make sure the backend server is running"
fi


echo "Starting Multi-Agent Doc Research (Chainlit Version)..."

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | xargs)
fi

# Set default values if not provided
export API_URL=${API_URL:-"http://localhost:8000/plan_search"}

echo "API_URL: $API_URL"

# Run the Chainlit application
chainlit run src/app.py --host 127.0.0.1 --port 7860 -w