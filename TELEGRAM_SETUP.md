# Telegram Integration Setup Guide

This guide explains how to set up Telegram notifications for all alerts in the Sakshi.AI platform.

## Overview

The Telegram integration automatically sends all alerts to a configured Telegram group or channel. This includes:
- Fall detection alerts
- Smoking/Fire detection alerts
- Unauthorized entry alerts
- Cash detection alerts
- Dress code violations
- PPE violations
- Queue violations
- Table service violations
- Table cleanliness violations
- Phone usage alerts
- Grooming violations
- And all other alert types

## Setup Steps

### 1. Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` command
3. Follow the instructions to create a bot:
   - Choose a name for your bot (e.g., "Sakshi AI Alerts")
   - Choose a username (must end with `bot`, e.g., `sakshi_ai_alerts_bot`)
4. BotFather will give you a **Bot Token** (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
5. **Save this token** - you'll need it in step 3

### 2. Get Your Chat ID

You need to get the ID of the Telegram group or channel where you want to receive alerts.

#### Option A: For a Group
1. Add your bot to the group
2. Send a message in the group
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Replace `<YOUR_BOT_TOKEN>` with your actual bot token
4. Look for `"chat":{"id":-123456789}` in the response
5. The negative number is your **Chat ID** (e.g., `-123456789`)

#### Option B: For a Channel
1. Create a channel
2. Add your bot as an administrator
3. Send a test message in the channel
4. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
5. Look for `"chat":{"id":-1001234567890}` in the response
6. The number is your **Chat ID**

#### Option C: For Personal Messages
1. Start a conversation with your bot
2. Send a message to the bot
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Look for `"chat":{"id":123456789}` in the response
5. The positive number is your **Chat ID**

### 3. Configure Environment Variables

Set the following environment variables:

#### On Windows (PowerShell):
```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token_here"
$env:TELEGRAM_CHAT_ID="your_chat_id_here"
```

#### On Linux/Mac:
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

#### Using .env file:
Create a `.env` file in your project root:
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**Note:** The system also supports the legacy variable names:
- `bot_token` (instead of `TELEGRAM_BOT_TOKEN`)
- `chat_id` (instead of `TELEGRAM_CHAT_ID`)

### 4. Disable Telegram (Optional)

If you want to temporarily disable Telegram notifications without removing the configuration:

```bash
export TELEGRAM_DISABLED=true
```

Or in `.env`:
```
TELEGRAM_DISABLED=true
```

## Alert Format

Alerts sent to Telegram include:
- **Alert Type**: Formatted alert type (e.g., "Fall Detection", "Smoking Detection")
- **Channel ID**: Which camera/channel triggered the alert
- **Timestamp**: When the alert occurred
- **Message**: Detailed alert message
- **Image/GIF**: Snapshot or GIF of the alert (if available)
- **Additional Data**: Detection counts, violations, wait times, etc.

### Example Alert Message:
```
🚨 Alert: Fall Detection
📍 Channel: camera_5
⏰ Time: 2025-01-15 14:30:25

📝 Message: Fall detected (duration: 5.2s)
🔢 Detections: 1
```

## Testing

1. Set up your bot token and chat ID
2. Restart your application
3. Trigger a test alert (e.g., by simulating a fall detection)
4. Check your Telegram group/channel for the alert

## Troubleshooting

### Alerts Not Being Sent

1. **Check Environment Variables:**
   ```python
   import os
   print(os.getenv("TELEGRAM_BOT_TOKEN"))
   print(os.getenv("TELEGRAM_CHAT_ID"))
   ```

2. **Check Logs:**
   Look for messages like:
   - `"Telegram notifier initialized"` - Configuration loaded
   - `"Telegram message sent successfully"` - Alert sent
   - `"Failed to send Telegram notification"` - Error occurred

3. **Verify Bot Token:**
   - Make sure the token is correct (no extra spaces)
   - Token should look like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

4. **Verify Chat ID:**
   - For groups: Should be negative (e.g., `-123456789`)
   - For channels: Usually starts with `-100` (e.g., `-1001234567890`)
   - For personal: Positive number (e.g., `123456789`)

5. **Check Bot Permissions:**
   - Bot must be added to the group/channel
   - Bot must have permission to send messages
   - For channels, bot must be an administrator

6. **Test Bot Manually:**
   ```bash
   curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/sendMessage?chat_id=<YOUR_CHAT_ID>&text=Test"
   ```

### Image/GIF Not Sending

- Check if the file path exists
- Verify file permissions
- Check file size (Telegram has limits: 10MB for photos, 50MB for documents)
- Look for errors in logs about file paths

### Rate Limiting

Telegram has rate limits:
- 30 messages per second per bot
- If you have many alerts, they may be queued

If you hit rate limits, alerts will still be saved to the database but may be delayed in Telegram.

## Security Notes

- **Never commit your bot token to version control**
- Use environment variables or secure configuration files
- Rotate your bot token if it's compromised
- Use a dedicated bot for production alerts

## Advanced Configuration

### Custom Alert Formatting

You can customize alert formatting by modifying `modules/telegram_notifier.py`:
- Change emoji mappings in `_get_alert_emoji()`
- Modify alert type names in `_format_alert_type()`
- Customize message format in `send_alert()`

### Multiple Telegram Destinations

To send to multiple Telegram groups, you can:
1. Create multiple bot instances
2. Modify the code to send to multiple chat IDs
3. Use a Telegram channel and forward messages

## Support

For issues or questions:
1. Check application logs
2. Verify Telegram API status: https://status.telegram.org/
3. Test bot manually using curl or Postman
4. Review Telegram Bot API documentation: https://core.telegram.org/bots/api



