#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║            STOPPING BATCH DEMO                        ║${NC}"
echo -e "${RED}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop Airflow Scheduler
echo -e "${BLUE}Stopping Airflow Scheduler...${NC}"
if [ -f logs/batch_scheduler.pid ]; then
    PID=$(cat logs/batch_scheduler.pid)
    kill $PID 2>/dev/null && echo -e "${GREEN}✓ Scheduler stopped${NC}" || echo -e "${YELLOW}⚠ Already stopped${NC}"
    rm logs/batch_scheduler.pid
fi

# Stop Airflow Webserver
echo -e "${BLUE}Stopping Airflow Webserver...${NC}"
if [ -f logs/batch_webserver.pid ]; then
    PID=$(cat logs/batch_webserver.pid)
    kill $PID 2>/dev/null && echo -e "${GREEN}✓ Webserver stopped${NC}" || echo -e "${YELLOW}⚠ Already stopped${NC}"
    rm logs/batch_webserver.pid
fi

# Force kill any remaining Airflow processes
pkill -f "airflow scheduler" 2>/dev/null
pkill -f "airflow webserver" 2>/dev/null

# Clean up PID file
rm -f logs/batch_demo.pids

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          BATCH DEMO STOPPED! ✓                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Note: PostgreSQL is still running (as designed)${NC}"
echo "  To stop: brew services stop postgresql@15"
echo ""

echo -e "${BLUE}Database still has data (for next demo or analysis)${NC}"
echo "  To clean: ./cleanup_database.sh"
echo ""

echo -e "${BLUE}To view data:${NC}"
echo "  psql -U postgres -d stock_prediction -c \"SELECT COUNT(*) FROM stock_prices;\""
echo ""

echo -e "${GREEN}Demo complete! 🎉${NC}"
echo ""

