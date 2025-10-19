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

print_header "START DATABASE - PostgreSQL Setup + Start"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${YELLOW}PostgreSQL not installed. Installing via Homebrew...${NC}"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew not found! Install it first:${NC}"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    
    brew install postgresql@15
    brew services start postgresql@15
    sleep 5
    
    echo -e "${GREEN}✓ PostgreSQL installed and started${NC}"
else
    echo -e "${GREEN}✓ PostgreSQL already installed${NC}"
fi

# Check if service is running
if pg_isready -q; then
    echo -e "${GREEN}✓ PostgreSQL already running${NC}"
else
    echo -e "${BLUE}Starting PostgreSQL service...${NC}"
    brew services start postgresql@15
    
    # Wait for it to be ready
    echo -e "${YELLOW}Waiting for PostgreSQL to start...${NC}"
    for i in {1..10}; do
        if pg_isready -q; then
            echo -e "${GREEN}✓ PostgreSQL started successfully${NC}"
            break
        fi
        sleep 1
    done
    
    if ! pg_isready -q; then
        echo -e "${RED}✗ PostgreSQL failed to start${NC}"
        exit 1
    fi
fi

echo ""
print_header "Creating Database and User"

# Create user (if not exists)
psql postgres -c "CREATE USER postgres WITH PASSWORD 'postgres';" 2>/dev/null || echo -e "${YELLOW}User 'postgres' already exists${NC}"

# Grant superuser
psql postgres -c "ALTER USER postgres WITH SUPERUSER;" 2>/dev/null

# Create database
psql postgres -U $USER -c "CREATE DATABASE stock_prediction;" 2>/dev/null || echo -e "${YELLOW}Database 'stock_prediction' already exists${NC}"

# Grant privileges
psql postgres -U $USER -c "GRANT ALL PRIVILEGES ON DATABASE stock_prediction TO postgres;" 2>/dev/null

echo -e "${GREEN}✓ Database and user configured${NC}"

echo ""
print_header "Creating Tables and Indexes"

psql -U postgres -d stock_prediction << 'EOF'
-- Stock prices table
CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    stock_symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(15, 2),
    high DECIMAL(15, 2),
    low DECIMAL(15, 2),
    close DECIMAL(15, 2) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stock_symbol, date)
);

-- Crawl metadata table
CREATE TABLE IF NOT EXISTS crawl_metadata (
    stock_symbol VARCHAR(10) PRIMARY KEY,
    last_crawl_date DATE,
    last_data_date DATE,
    total_records INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(stock_symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_prices_created_at ON stock_prices(created_at DESC);
EOF

echo -e "${GREEN}✓ Tables and indexes created${NC}"

# Verify connection
if psql -U postgres -d stock_prediction -c "SELECT COUNT(*) FROM stock_prices;" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Database connection verified${NC}"
else
    echo -e "${RED}✗ Cannot connect to database${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        PostgreSQL is READY! ✓                         ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}Database Info:${NC}"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: stock_prediction"
echo "  User: postgres"
echo "  Password: postgres"
echo ""

echo -e "${BLUE}Connect to database:${NC}"
echo "  psql -U postgres -d stock_prediction"
echo ""

echo -e "${BLUE}Quick query:${NC}"
echo "  psql -U postgres -d stock_prediction -c \"SELECT COUNT(*) FROM stock_prices;\""
echo ""

echo -e "${GREEN}Ready for demos! 🚀${NC}"
echo ""

