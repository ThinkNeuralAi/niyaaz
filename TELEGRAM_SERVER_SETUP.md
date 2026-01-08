# Setting Up Telegram on Linux Server

## Quick Setup (Using .env file - Recommended)

### Step 1: Create or Edit .env file

In your project root directory (`~/workspace/Anjana/Niyaz`), create or edit `.env`:

```bash
cd ~/workspace/Anjana/Niyaz
nano .env
```

Add these lines:
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Save and exit (Ctrl+X, then Y, then Enter in nano).

### Step 2: Verify .env file

```bash
cat .env
```

You should see your token and chat ID (don't share these publicly!).

### Step 3: Test Configuration

```bash
python test_telegram.py
```

You should now see:
- ✅ TELEGRAM_BOT_TOKEN: Set
- ✅ TELEGRAM_CHAT_ID: Set
- ✅ Telegram notifier is ENABLED

### Step 4: Restart Your Application

```bash
# Stop your application (Ctrl+C or kill process)
# Then restart it
python app.py
```

## Alternative: Set Environment Variables in Shell

### For Current Session Only:

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

### For Persistent (Add to ~/.bashrc or ~/.profile):

```bash
echo 'export TELEGRAM_BOT_TOKEN="your_bot_token_here"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="your_chat_id_here"' >> ~/.bashrc
source ~/.bashrc
```

## If Using systemd Service

Edit your systemd service file (usually `/etc/systemd/system/sakshi-ai.service`):

```ini
[Service]
Environment="TELEGRAM_BOT_TOKEN=your_bot_token_here"
Environment="TELEGRAM_CHAT_ID=your_chat_id_here"
```

Then reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart sakshi-ai
```

## Verify It's Working

After setting environment variables and restarting:

1. **Check startup logs** - Look for:
   ```
   ✅ Telegram notifier initialized successfully
   ```

2. **Trigger a test alert** - Any alert should send to Telegram

3. **Check logs for Telegram messages**:
   ```
   ✅ Telegram message sent successfully
   ✅ Telegram photo sent successfully
   ```

## Troubleshooting

### Still showing "Not set" after adding to .env?

1. **Check if python-dotenv is installed:**
   ```bash
   pip list | grep python-dotenv
   ```
   If not installed:
   ```bash
   pip install python-dotenv
   ```

2. **Verify .env file location:**
   - Must be in project root (same directory as `app.py`)
   - Check: `ls -la .env`

3. **Check .env file format:**
   - No spaces around `=`
   - No quotes needed (but quotes are OK)
   - One variable per line

### Environment variables not persisting?

- If using systemd, add to service file
- If using screen/tmux, set in that session
- If using supervisor, add to config file

## Quick Test Commands

```bash
# Check if variables are set
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID

# Test Telegram
python test_telegram.py

# Check application logs
tail -f logs/app.log  # or wherever your logs are
```



