#!/bin/bash
# Setup Telegram environment variables on Linux server

echo "=========================================="
echo "Telegram Environment Variables Setup"
echo "=========================================="
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✅ .env file exists"
    echo ""
    echo "Add these lines to your .env file:"
    echo "  TELEGRAM_BOT_TOKEN=your_bot_token_here"
    echo "  TELEGRAM_CHAT_ID=your_chat_id_here"
    echo ""
    read -p "Do you want to add them now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter TELEGRAM_BOT_TOKEN: " bot_token
        read -p "Enter TELEGRAM_CHAT_ID: " chat_id
        echo "" >> .env
        echo "TELEGRAM_BOT_TOKEN=$bot_token" >> .env
        echo "TELEGRAM_CHAT_ID=$chat_id" >> .env
        echo "✅ Added to .env file"
    fi
else
    echo "Creating .env file..."
    read -p "Enter TELEGRAM_BOT_TOKEN: " bot_token
    read -p "Enter TELEGRAM_CHAT_ID: " chat_id
    echo "TELEGRAM_BOT_TOKEN=$bot_token" > .env
    echo "TELEGRAM_CHAT_ID=$chat_id" >> .env
    echo "✅ Created .env file"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Restart your application"
echo "2. Run: python test_telegram.py"
echo "3. Check logs for: '✅ Telegram notifier initialized'"
echo ""



