#!/bin/bash
# Quick setup script for PostgreSQL on Linux/Mac

echo "========================================"
echo "PostgreSQL Setup for Sakshi.AI"
echo "========================================"
echo ""

# Set PostgreSQL connection details
read -p "Enter PostgreSQL host [localhost]: " DB_HOST
DB_HOST=${DB_HOST:-localhost}

read -p "Enter PostgreSQL port [5432]: " DB_PORT
DB_PORT=${DB_PORT:-5432}

read -p "Enter database name [sakshiai]: " DB_NAME
DB_NAME=${DB_NAME:-sakshiai}

read -p "Enter PostgreSQL user [postgres]: " DB_USER
DB_USER=${DB_USER:-postgres}

read -sp "Enter PostgreSQL password: " DB_PASSWORD
echo ""

if [ -z "$DB_PASSWORD" ]; then
    echo "Error: Password is required"
    exit 1
fi

echo ""
echo "Setting environment variables..."

# Add to ~/.bashrc or ~/.zshrc
SHELL_CONFIG=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
fi

if [ -n "$SHELL_CONFIG" ]; then
    # Remove old entries
    sed -i.bak '/^export DB_HOST=/d' "$SHELL_CONFIG"
    sed -i.bak '/^export DB_PORT=/d' "$SHELL_CONFIG"
    sed -i.bak '/^export DB_NAME=/d' "$SHELL_CONFIG"
    sed -i.bak '/^export DB_USER=/d' "$SHELL_CONFIG"
    sed -i.bak '/^export DB_PASSWORD=/d' "$SHELL_CONFIG"
    
    # Add new entries
    echo "" >> "$SHELL_CONFIG"
    echo "# PostgreSQL configuration for Sakshi.AI" >> "$SHELL_CONFIG"
    echo "export DB_HOST=$DB_HOST" >> "$SHELL_CONFIG"
    echo "export DB_PORT=$DB_PORT" >> "$SHELL_CONFIG"
    echo "export DB_NAME=$DB_NAME" >> "$SHELL_CONFIG"
    echo "export DB_USER=$DB_USER" >> "$SHELL_CONFIG"
    echo "export DB_PASSWORD=$DB_PASSWORD" >> "$SHELL_CONFIG"
    
    echo "✅ Added to $SHELL_CONFIG"
    echo "   Run 'source $SHELL_CONFIG' or restart terminal"
fi

# Set for current session
export DB_HOST=$DB_HOST
export DB_PORT=$DB_PORT
export DB_NAME=$DB_NAME
export DB_USER=$DB_USER
export DB_PASSWORD=$DB_PASSWORD

echo ""
echo "========================================"
echo "Next steps:"
echo "========================================"
echo "1. Create PostgreSQL database (if not exists):"
echo "   psql -U $DB_USER -c 'CREATE DATABASE $DB_NAME;'"
echo ""
echo "2. Install Python driver:"
echo "   pip install psycopg2-binary"
echo ""
echo "3. Run migration script:"
echo "   python migrate_to_postgresql.py"
echo ""
echo "4. Start application:"
echo "   python app.py"
echo ""



