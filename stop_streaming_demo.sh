#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║         STOPPING STREAMING DEMO                       ║${NC}"
echo -e "${RED}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop Kafka Producer
echo -e "${BLUE}Stopping Kafka Producer...${NC}"
if [ -f logs/streaming_producer.pid ]; then
    PID=$(cat logs/streaming_producer.pid)
    kill $PID 2>/dev/null && echo -e "${GREEN}✓ Producer stopped${NC}" || echo -e "${YELLOW}⚠ Already stopped${NC}"
    rm logs/streaming_producer.pid
fi

# Stop Kafka Consumer
echo -e "${BLUE}Stopping Kafka Consumer...${NC}"
if [ -f logs/streaming_consumer.pid ]; then
    PID=$(cat logs/streaming_consumer.pid)
    kill $PID 2>/dev/null && echo -e "${GREEN}✓ Consumer stopped${NC}" || echo -e "${YELLOW}⚠ Already stopped${NC}"
    rm logs/streaming_consumer.pid
fi

# Stop Airflow Scheduler
echo -e "${BLUE}Stopping Airflow Scheduler...${NC}"
if [ -f logs/streaming_scheduler.pid ]; then
    PID=$(cat logs/streaming_scheduler.pid)
    kill $PID 2>/dev/null && echo -e "${GREEN}✓ Scheduler stopped${NC}" || echo -e "${YELLOW}⚠ Already stopped${NC}"
    rm logs/streaming_scheduler.pid
fi

# Stop Airflow Webserver
echo -e "${BLUE}Stopping Airflow Webserver...${NC}"
if [ -f logs/streaming_webserver.pid ]; then
    PID=$(cat logs/streaming_webserver.pid)
    kill $PID 2>/dev/null && echo -e "${GREEN}✓ Webserver stopped${NC}" || echo -e "${YELLOW}⚠ Already stopped${NC}"
    rm logs/streaming_webserver.pid
fi

# Force kill any remaining processes
pkill -f "kafka_producer" 2>/dev/null
pkill -f "kafka_consumer" 2>/dev/null
pkill -f "airflow scheduler" 2>/dev/null
pkill -f "airflow webserver" 2>/dev/null

# Stop Kafka Infrastructure
echo -e "${BLUE}Stopping Kafka Infrastructure...${NC}"
docker-compose down
echo -e "${GREEN}✓ Kafka containers stopped${NC}"

# Clean up PID file
rm -f logs/streaming_demo.pids

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        STREAMING DEMO STOPPED! ✓                      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Clean database for BATCH demo:${NC}"
echo "  psql -U postgres -d stock_prediction << EOF"
echo "  DELETE FROM stock_prices;"
echo "  DELETE FROM crawl_metadata;"
echo "  SELECT COUNT(*) as remaining FROM stock_prices;"
echo "  EOF"
echo ""

echo -e "${BLUE}Or run cleanup script:${NC}"
echo "  ./cleanup_database.sh"
echo ""

echo -e "${GREEN}Ready for BATCH demo! 🎯${NC}"
echo ""

