#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║         CLEANING DATABASE FOR NEXT DEMO               ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if PostgreSQL is running
if ! psql -U postgres -d stock_prediction -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot connect to database!${NC}"
    echo "Make sure PostgreSQL is running: ./start_database.sh"
    exit 1
fi

echo -e "${BLUE}Current database status:${NC}"
psql -U postgres -d stock_prediction -c "SELECT COUNT(*) as total_records FROM stock_prices;"
echo ""

# Ask for confirmation
echo -e "${RED}⚠️  WARNING: This will DELETE ALL DATA!${NC}"
echo -e "${YELLOW}Press Enter to continue, or Ctrl+C to cancel...${NC}"
read

# Delete all data
echo -e "${BLUE}Deleting all data...${NC}"

psql -U postgres -d stock_prediction << 'EOF'
DELETE FROM stock_prices;
DELETE FROM crawl_metadata;
EOF

echo ""
echo -e "${GREEN}✓ Database cleaned!${NC}"
echo ""

# Verify
echo -e "${BLUE}Verification:${NC}"
psql -U postgres -d stock_prediction -c "SELECT COUNT(*) as remaining_records FROM stock_prices;"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         DATABASE READY FOR NEXT DEMO! ✓               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}You can now run:${NC}"
echo "  ./demo_streaming_full.sh   (for streaming demo)"
echo "  ./demo_batch_full.sh       (for batch demo)"
echo ""

