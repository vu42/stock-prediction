#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "╔═══════════════════════════════════════════════════════╗"
echo "║        Installing Apache Airflow                      ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check if Python 3.12 is installed
if command -v python3.12 &> /dev/null; then
    echo -e "${GREEN}✓ Python 3.12 found${NC}"
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION="$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
    if [ "$PYTHON_VERSION" = "3.12" ]; then
        echo -e "${GREEN}✓ Python 3.12 found${NC}"
        PYTHON_CMD="python3"
    else
        echo -e "${RED}✗ Python 3.12 not found (found: $PYTHON_VERSION)${NC}"
        echo ""
        echo "Airflow requires Python 3.12 or lower"
        echo ""
        echo "Install Python 3.12:"
        echo "  brew install python@3.12"
        echo ""
        echo "Then run this script again"
        exit 1
    fi
else
    echo -e "${RED}✗ Python not found${NC}"
    echo ""
    echo "Install Python 3.12:"
    echo "  brew install python@3.12"
    exit 1
fi

PYTHON_VERSION="$($PYTHON_CMD --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
echo -e "${BLUE}Using Python version: $PYTHON_VERSION${NC}"

# Check if venv exists
VENV_DIR="$HOME/.airflow-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment with Python 3.12...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate venv
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Set Airflow home
export AIRFLOW_HOME=~/airflow

# Set Airflow version
AIRFLOW_VERSION=2.9.3
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"

echo -e "${YELLOW}Installing Apache Airflow ${AIRFLOW_VERSION}...${NC}"
echo -e "${YELLOW}(This may take 5-10 minutes...)${NC}"
echo ""

# Upgrade pip first
pip install --upgrade pip

# Install Airflow
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Check installation
if [ -f "$VENV_DIR/bin/airflow" ]; then
    echo ""
    echo -e "${GREEN}✓ Airflow installed successfully!${NC}"
    echo ""
    "$VENV_DIR/bin/airflow" version
    echo ""
    
    # Initialize database
    echo -e "${BLUE}Initializing Airflow database...${NC}"
    "$VENV_DIR/bin/airflow" db init
    
    # Create admin user
    echo ""
    echo -e "${BLUE}Creating admin user...${NC}"
    "$VENV_DIR/bin/airflow" users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin 2>/dev/null || echo -e "${YELLOW}Admin user may already exist${NC}"
    
    # Create activation helper
    echo ""
    echo -e "${BLUE}Creating activation script...${NC}"
    cat > ~/activate_airflow.sh << 'HEREDOC'
#!/bin/bash
# Activate Airflow virtual environment
source ~/.airflow-venv/bin/activate
export AIRFLOW_HOME=~/airflow
export PATH="$HOME/.airflow-venv/bin:$PATH"
echo "✓ Airflow environment activated"
echo "Run: airflow webserver -p 8080"
echo "     airflow scheduler"
HEREDOC
    chmod +x ~/activate_airflow.sh
    
    echo ""
    echo -e "${GREEN}✓ Airflow setup complete!${NC}"
    echo ""
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║              INSTALLATION SUCCESSFUL!                 ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo "Airflow Location: $VENV_DIR"
    echo "Airflow Home: $AIRFLOW_HOME"
    echo "Python Version: $PYTHON_VERSION"
    echo "Username: admin"
    echo "Password: admin"
    echo ""
    echo "To activate Airflow environment:"
    echo "  source ~/activate_airflow.sh"
    echo ""
    echo "Then run demos:"
    echo "  ./start_streaming_demo.sh"
    echo "  ./start_batch_demo.sh"
    echo ""
    
    # Add to shell profile
    if ! grep -q "activate_airflow" ~/.zshrc 2>/dev/null; then
        echo ""
        echo -e "${BLUE}Adding Airflow to your shell...${NC}"
        echo 'alias activate_airflow="source ~/activate_airflow.sh"' >> ~/.zshrc
        echo -e "${GREEN}✓ Added 'activate_airflow' alias to ~/.zshrc${NC}"
        echo ""
        echo "You can now run: activate_airflow"
    fi
    
else
    echo ""
    echo -e "${RED}✗ Airflow installation failed!${NC}"
    exit 1
fi
