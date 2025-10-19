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

print_header "FULL DEMO STREAMING - Kafka + Airflow + Database"

# Set PostgreSQL path for Homebrew (keg-only formula)
export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker not running!${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo -e "${GREEN}✓ Docker running${NC}"

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
print_header "STEP 1: Starting Kafka Infrastructure"

# Stop any existing containers
docker-compose down > /dev/null 2>&1

# Start Kafka
echo -e "${BLUE}Starting Kafka, Zookeeper, Kafka UI...${NC}"
docker-compose up -d

# Wait for Kafka
echo -e "${YELLOW}Waiting for Kafka to initialize (30 seconds)...${NC}"
sleep 30

echo -e "${GREEN}✓ Kafka infrastructure ready${NC}"

echo ""
print_header "STEP 2: Setup Kafka Topic with Partitions"

echo -e "${BLUE}Creating topic 'vn30-stock-prices' with 6 partitions...${NC}"
python3 setup_kafka_topic.py
echo -e "${GREEN}✓ Kafka topic configured${NC}"

echo ""
print_header "STEP 3: Initialize Database Schema"

python3 << 'PYEOF'
from modules.database import init_database
print("Initializing database schema...")
init_database()
print("✓ Database initialized")
PYEOF

echo ""
print_header "STEP 4: Starting Kafka Producer & Consumer"

# Create logs directory
mkdir -p logs

# Start Kafka Producer (background)
echo -e "${BLUE}Starting Kafka Producer...${NC}"
nohup python3 -m modules.kafka_producer > logs/streaming_producer_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PRODUCER_PID=$!
echo "$PRODUCER_PID" > logs/streaming_producer.pid
echo -e "${GREEN}✓ Producer started (PID: $PRODUCER_PID)${NC}"

# Start Kafka Consumer (background)
echo -e "${BLUE}Starting Kafka Consumer...${NC}"
nohup python3 -m modules.kafka_consumer > logs/streaming_consumer_$(date +%Y%m%d_%H%M%S).log 2>&1 &
CONSUMER_PID=$!
echo "$CONSUMER_PID" > logs/streaming_consumer.pid
echo -e "${GREEN}✓ Consumer started (PID: $CONSUMER_PID)${NC}"

echo ""
print_header "STEP 5: Starting Airflow"

# Stop any existing Airflow
echo -e "${BLUE}Stopping any existing Airflow...${NC}"
pkill -f "airflow scheduler" > /dev/null 2>&1
pkill -f "airflow webserver" > /dev/null 2>&1
sleep 2

# Copy DAGs
echo -e "${BLUE}Copying DAGs to Airflow...${NC}"
mkdir -p ~/airflow/dags
cp -r dags/streaming/* ~/airflow/dags/
echo -e "${GREEN}✓ DAGs copied (streaming folder)${NC}"

# Start Airflow Scheduler (background)
echo -e "${BLUE}Starting Airflow Scheduler...${NC}"
nohup airflow scheduler > logs/streaming_scheduler_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SCHEDULER_PID=$!
echo "$SCHEDULER_PID" > logs/streaming_scheduler.pid
echo -e "${GREEN}✓ Scheduler started (PID: $SCHEDULER_PID)${NC}"

# Start Airflow Webserver (background)
echo -e "${BLUE}Starting Airflow Webserver...${NC}"
nohup airflow webserver -p 8081 > logs/streaming_webserver_$(date +%Y%m%d_%H%M%S).log 2>&1 &
WEBSERVER_PID=$!
echo "$WEBSERVER_PID" > logs/streaming_webserver.pid
echo -e "${GREEN}✓ Webserver started (PID: $WEBSERVER_PID)${NC}"

echo -e "${YELLOW}Waiting for Airflow to initialize (15 seconds)...${NC}"
sleep 15

echo ""
print_header "STEP 6: Opening UIs in Browser"

# Open Kafka UI
echo -e "${GREEN}✓ Opening Kafka UI...${NC}"
open http://localhost:8080 2>/dev/null || echo "  → http://localhost:8080"
sleep 2

# Open Airflow UI
echo -e "${GREEN}✓ Opening Airflow UI...${NC}"
open http://localhost:8081 2>/dev/null || echo "  → http://localhost:8081"
sleep 2

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      STREAMING DEMO IS LIVE! 🚀                       ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}🌐 OPEN UIs:${NC}"
echo "  1. Kafka UI:      http://localhost:8080"
echo "  2. Airflow UI:    http://localhost:8081 (admin/admin)"
echo "  3. Database:      psql -U postgres -d stock_prediction"
echo ""

echo -e "${BLUE}📊 WHAT'S RUNNING:${NC}"
echo "  ✓ Kafka Producer (30 ticks/sec)"
echo "  ✓ Kafka Consumer (aggregating to OHLCV)"
echo "  ✓ Data flowing to PostgreSQL"
echo "  ✓ Airflow orchestrator monitoring"
echo ""

echo -e "${BLUE}🔍 MONITORING:${NC}"
echo ""
echo -e "${YELLOW}1. Check Kafka UI:${NC}"
echo "   http://localhost:8080"
echo "   → See messages in 'vn30-stock-prices' topic"
echo ""
echo -e "${YELLOW}2. Check Database:${NC}"
echo "   psql -U postgres -d stock_prediction"
echo ""
echo "   # Count records (should increase)"
echo "   SELECT COUNT(*) FROM stock_prices;"
echo ""
echo "   # Show recent data"
echo "   SELECT stock_symbol, date, ROUND(close::numeric, 2) as close, created_at"
echo "   FROM stock_prices ORDER BY created_at DESC LIMIT 10;"
echo ""
echo "   # Group by stock"
echo "   SELECT stock_symbol, COUNT(*) FROM stock_prices GROUP BY stock_symbol;"
echo ""
echo -e "${YELLOW}3. Watch Logs:${NC}"
echo "   tail -f logs/streaming_producer_*.log"
echo "   tail -f logs/streaming_consumer_*.log"
echo ""
echo -e "${YELLOW}4. Check Airflow DAGs:${NC}"
echo "   http://localhost:8081"
echo "   → Enable: kafka_health_monitor (runs every 5 min)"
echo "   → Enable: vn30_streaming_orchestrator (runs every 30 min)"
echo ""

echo -e "${BLUE}🛑 STOP STREAMING DEMO:${NC}"
echo "  ./stop_streaming_demo.sh"
echo ""


# Save PIDs for cleanup
cat > logs/streaming_demo.pids << EOF
PRODUCER_PID=$PRODUCER_PID
CONSUMER_PID=$CONSUMER_PID
SCHEDULER_PID=$SCHEDULER_PID
WEBSERVER_PID=$WEBSERVER_PID
EOF

echo -e "${YELLOW}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  Wait 1-2 minutes for data, then check Kafka UI!     ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Ready to present! 🎬${NC}"
echo ""

