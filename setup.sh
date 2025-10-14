#!/bin/bash

# VN30 Stock Prediction System - Setup Script
# This script helps set up the environment

set -e

echo "=========================================="
echo "VN30 Stock Prediction System - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Found Python $python_version"

# Install dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
pip3 install -r requirements.txt

# Check PostgreSQL
echo ""
echo "[3/5] Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "PostgreSQL is installed"
elif command -v docker &> /dev/null; then
    echo "PostgreSQL not found. Starting with Docker..."
    docker run -d \
      --name stock-postgres \
      -e POSTGRES_PASSWORD=postgres \
      -e POSTGRES_DB=stock_prediction \
      -p 5432:5432 \
      postgres:15
    echo "PostgreSQL container started"
else
    echo "WARNING: PostgreSQL not found and Docker not available"
    echo "Please install PostgreSQL manually"
fi

# Initialize Airflow
echo ""
echo "[4/5] Initializing Airflow..."
export AIRFLOW_HOME=~/airflow
airflow db init

# Create output directory
echo ""
echo "[5/5] Creating output directory..."
mkdir -p output

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit config.py with your settings"
echo "2. Create Airflow user:"
echo "   airflow users create --username admin --password admin --role Admin --email admin@example.com --firstname Admin --lastname User"
echo ""
echo "3. Copy DAGs to Airflow:"
echo "   cp -r dags/* ~/airflow/dags/"
echo ""
echo "4. Start Airflow:"
echo "   airflow scheduler &"
echo "   airflow webserver -p 8080"
echo ""
echo "5. Access: http://localhost:8080"
echo ""

