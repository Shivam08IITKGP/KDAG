#!/bin/bash

# Narrative Auditor Quick Start Script
# Usage: bash quickstart.sh

echo "================================"
echo "Narrative Auditor - Quick Start"
echo "================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "✓ Python version: $python_version"

# Create directories
echo "Creating directories..."
mkdir -p input output
echo "✓ Directories created"

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env template..."
    cat > .env << 'EOF'
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here
LLM_MODEL=mistralai/mistral-7b-instruct

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Logging
LOG_LEVEL=INFO
VERBOSE=True
EOF
    echo "✓ Created .env file (UPDATE WITH YOUR CREDENTIALS!)"
    echo ""
    echo "IMPORTANT: Edit .env and add:"
    echo "  1. Your OpenRouter API key from https://openrouter.ai"
    echo "  2. Your Neo4j credentials"
else
    echo "✓ .env file already exists"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Check if novel exists
echo ""
if [ ! -f "input/novel.txt" ]; then
    echo "⚠ Novel file not found at input/novel.txt"
    echo "  Please download/create your novel and place at: input/novel.txt"
    echo ""
    echo "Example: Download from Project Gutenberg"
    echo "  curl https://www.gutenberg.org/cache/epub/1661/pg1661.txt -o input/novel.txt"
else
    echo "✓ Novel file found"
fi

# Check Neo4j connection
echo ""
echo "Checking Neo4j connection..."
python3 << 'PYEOF'
import sys
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    
    driver = GraphDatabase.driver(uri, auth=(user, pwd), encrypted=False)
    driver.verify_connectivity()
    driver.close()
    print("✓ Neo4j connection verified")
except Exception as e:
    print(f"✗ Neo4j connection failed: {e}")
    print("  Make sure Neo4j is running and credentials are correct")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi

# Check OpenRouter API
echo "Checking OpenRouter API..."
python3 << 'PYEOF'
import sys
import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key or api_key == "your_api_key_here":
    print("✗ OpenRouter API key not configured")
    print("  Update OPENROUTER_API_KEY in .env file")
    sys.exit(1)

try:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Check available models
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=10
    )
    if response.status_code == 200:
        print("✓ OpenRouter API connection verified")
    else:
        print(f"✗ OpenRouter API error: {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ OpenRouter API connection failed: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi

# Setup complete
echo ""
echo "================================"
echo "Setup Complete! ✓"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Edit input/novel.txt with your novel"
echo "2. Run: python pipeline.py"
echo "3. View graph: http://localhost:7474"
echo ""
echo "For help: cat README.md"
