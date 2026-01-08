#!/bin/bash
# Auto-restart script for Sakshi.AI application (Linux/Mac)
# This script will automatically restart the app if it crashes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if app is running
check_app_running() {
    pgrep -f "python.*app.py" > /dev/null
}

# Function to kill existing app processes
kill_existing_app() {
    if check_app_running; then
        echo -e "${YELLOW}Killing existing app processes...${NC}"
        pkill -f "python.*app.py"
        sleep 2
    fi
}

# Trap Ctrl+C to gracefully stop
trap 'echo -e "\n${YELLOW}Stopping application...${NC}"; kill_existing_app; exit 0' INT TERM

# Main restart loop
while true; do
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Starting Sakshi.AI Application...${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    # Activate virtual environment if it exists
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    # Kill any existing instances
    kill_existing_app
    
    # Run the application
    python app.py
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}Application crashed with exit code $EXIT_CODE${NC}"
        echo -e "${YELLOW}Waiting 5 seconds before restart...${NC}"
        echo -e "${RED}========================================${NC}"
        echo ""
        sleep 5
    else
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Application exited normally${NC}"
        echo -e "${GREEN}========================================${NC}"
        break
    fi
done


