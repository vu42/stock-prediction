#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${YELLOW}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║ ${1} ${NC}"
    echo -e "${YELLOW}╚═══════════════════════════════════════════════════════╝${NC}"
}

print_header "FULL DEMO BATCH - Real API + LSTM Training + Airflow"

# Set PostgreSQL path for Homebrew (keg-only formula)
export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check PostgreSQL
if ! psql -U postgres -d stock_prediction -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${RED}✗ PostgreSQL not ready!${NC}"
    echo "Run: ./start_database.sh"
    exit 1
fi
echo -e "${GREEN}✓ PostgreSQL ready${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 installed${NC}"

# Check and activate Airflow venv
export AIRFLOW_HOME=~/airflow
VENV_DIR="$HOME/.airflow-venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Activating Airflow environment...${NC}"
    source "$VENV_DIR/bin/activate"
    export PATH="$VENV_DIR/bin:$PATH"
    echo -e "${GREEN}✓ Airflow environment activated${NC}"
else
    echo -e "${RED}✗ Airflow not installed!${NC}"
    echo "Run: ./install_airflow.sh"
    exit 1
fi

# Check Airflow
if [ ! -d "$AIRFLOW_HOME" ]; then
    echo -e "${YELLOW}⚠ Airflow not initialized${NC}"
    echo "Initializing Airflow..."
    airflow db init
    mkdir -p ~/airflow/dags
fi
echo -e "${GREEN}✓ Airflow ready${NC}"

echo ""
print_header "STEP 1: Initialize Database Schema"

python3 << 'PYEOF'
from modules.database import init_database
print("Initializing database schema...")
init_database()
print("✓ Database initialized")
PYEOF

echo ""
print_header "STEP 2: Preparing Airflow"

# Stop any existing Airflow
echo -e "${BLUE}Stopping any existing Airflow...${NC}"
pkill -f "airflow scheduler" > /dev/null 2>&1
pkill -f "airflow webserver" > /dev/null 2>&1
sleep 2

# Copy BATCH DAGs only
echo -e "${BLUE}Copying BATCH DAGs to Airflow...${NC}"
mkdir -p ~/airflow/dags
rm -rf ~/airflow/dags/*  # Clean old DAGs
cp -r dags/batch/* ~/airflow/dags/
echo -e "${GREEN}✓ Batch DAGs copied${NC}"

# Create logs directory
mkdir -p logs

# Start Airflow Scheduler (background)
echo -e "${BLUE}Starting Airflow Scheduler...${NC}"
nohup airflow scheduler > logs/batch_scheduler_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SCHEDULER_PID=$!
echo "$SCHEDULER_PID" > logs/batch_scheduler.pid
echo -e "${GREEN}✓ Scheduler started (PID: $SCHEDULER_PID)${NC}"

# Start Airflow Webserver (background)
echo -e "${BLUE}Starting Airflow Webserver...${NC}"
nohup airflow webserver -p 8080 > logs/batch_webserver_$(date +%Y%m%d_%H%M%S).log 2>&1 &
WEBSERVER_PID=$!
echo "$WEBSERVER_PID" > logs/batch_webserver.pid
echo -e "${GREEN}✓ Webserver started (PID: $WEBSERVER_PID)${NC}"

echo -e "${YELLOW}Waiting for Airflow to initialize (20 seconds)...${NC}"
sleep 20

echo ""
print_header "STEP 3: Opening Airflow UI"

# Open Airflow UI
echo -e "${GREEN}✓ Opening Airflow UI...${NC}"
open http://localhost:8080 2>/dev/null || echo "  → http://localhost:8080"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         BATCH DEMO IS LIVE! 🚀                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}🌐 AIRFLOW UI:${NC}"
echo "  http://localhost:8080"
echo "  Login: admin / admin"
echo ""

echo -e "${BLUE}📊 BATCH DAGs (in batch/ folder):${NC}"
echo "  1. vn30_data_crawler      (Daily 5:00 PM)"
echo "  2. vn30_model_training    (Daily 5:30 PM)"
echo ""

echo -e "${BLUE}🎯 DEMO STEPS:${NC}"
echo ""
echo -e "${YELLOW}Step 1: Trigger Data Crawler${NC}"
echo "  1. Go to http://localhost:8080"
echo "  2. Browse → DAGs"
echo "  3. Find: vn30_data_crawler (in batch/ folder)"
echo "  4. Click ▶️ Play button → Trigger DAG"
echo "  5. Wait for it to complete (~2-5 minutes for 30 stocks)"
echo ""
echo -e "${YELLOW}Step 2: Monitor Database${NC}"
echo "  # Connect"
echo "  psql -U postgres -d stock_prediction"
echo ""
echo "  # Watch records increase"
echo "  SELECT COUNT(*) FROM stock_prices;"
echo ""
echo "  # Check by stock"
echo "  SELECT stock_symbol, COUNT(*) FROM stock_prices GROUP BY stock_symbol;"
echo ""
echo "  # View VCB data"
echo "  SELECT date, ROUND(close::numeric, 2) as close"
echo "  FROM stock_prices WHERE stock_symbol = 'VCB'"
echo "  ORDER BY date DESC LIMIT 10;"
echo ""
echo "  # Exit"
echo "  \\q"
echo ""
echo -e "${YELLOW}Step 3: Trigger Model Training (Optional - Takes Time!)${NC}"
echo "  1. Go back to Airflow UI"
echo "  2. Find: vn30_model_training"
echo "  3. Click ▶️ Play button → Trigger DAG"
echo "  4. This takes 10-20 minutes for 30 stocks"
echo "  5. Better to show pre-trained results in output/"
echo ""
echo -e "${YELLOW}Step 4: Show Output Files${NC}"
echo "  # List VCB output"
echo "  ls -lh output/VCB/"
echo ""
echo "  # Show model"
echo "  ls -lh output/VCB/VCB_model.h5"
echo ""
echo "  # Show evaluation chart"
echo "  open output/VCB/VCB_evaluation.png"
echo ""
echo "  # Show predictions"
echo "  cat output/VCB/VCB_future_predictions.csv | head -20"
echo ""
echo "  # Show prediction chart"
echo "  open output/VCB/VCB_future.png"
echo ""

echo -e "${BLUE}🔍 DATABASE CONNECTION:${NC}"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: stock_prediction"
echo "  User: postgres"
echo "  Password: postgres"
echo ""


echo -e "${BLUE}🛑 STOP BATCH DEMO:${NC}"
echo "  ./stop_batch_demo.sh"
echo ""

# Save PIDs for cleanup
cat > logs/batch_demo.pids << EOF
SCHEDULER_PID=$SCHEDULER_PID
WEBSERVER_PID=$WEBSERVER_PID
EOF

echo -e "${YELLOW}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  Trigger vn30_data_crawler in Airflow UI now!        ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Ready to present! 🎬${NC}"
echo ""

