#!/bin/bash

# GraphRAG MCP Server 실행 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 환경 변수 로드
if [ -f "$SCRIPT_DIR/../.env" ]; then
    source "$SCRIPT_DIR/../.env"
    echo "✅ Loaded environment variables from .env"
fi

# Python 가상환경 확인
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
    
    echo "📦 Installing dependencies..."
    pip install -e .
else
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo "🚀 Starting GraphRAG MCP Server..."
cd "$SCRIPT_DIR"
python server.py
