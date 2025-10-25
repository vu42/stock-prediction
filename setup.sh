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
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
echo ""
echo "[2b] Checking Flask-Session version..."
python3 - <<'PY'
import pkg_resources
try:
    v = pkg_resources.get_distribution('Flask-Session').version
except Exception:
    v = 'not installed'
print(f"Flask-Session: {v}")
PY

# Initialize Airflow
echo ""
echo "[4/5] Initializing Airflow..."
export AIRFLOW_HOME=~/airflow

# Run initial migration to create config file
airflow db migrate 2>/dev/null || true

# Fix auth manager configuration for Airflow 3.0 (if needed)
if [ -f ~/airflow/airflow.cfg ]; then
    if grep -q "^auth_manager = airflow.auth.managers.fab.fab_auth_manager.FabAuthManager" ~/airflow/airflow.cfg; then
        echo "Fixing deprecated FAB auth manager configuration..."
        sed -i.bak 's/^auth_manager = airflow.auth.managers.fab.fab_auth_manager.FabAuthManager/# auth_manager = airflow.auth.managers.fab.fab_auth_manager.FabAuthManager  # Disabled for Airflow 3.0/' ~/airflow/airflow.cfg
    fi
fi

# Run migration again (will complete if first attempt failed due to auth manager)
airflow db migrate

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

