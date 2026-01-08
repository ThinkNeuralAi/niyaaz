# Quick Start: Auto-Restart

## 🚀 Quick Setup (Choose One)

### Option 1: Simple Script (Easiest)

**Windows:**
```batch
run_with_restart.bat
```

**Linux/Mac:**
```bash
chmod +x run_with_restart.sh
./run_with_restart.sh
```

This will automatically restart the app if it crashes.

### Option 2: Python Wrapper (More Features)

**Basic (auto-restart on crash):**
```bash
python auto_restart.py
```

**Development Mode (restart on file changes):**
```bash
python auto_restart.py --watch
```

## 📋 What Each Option Does

| Option | Auto-Restart on Crash | File Watching | Logging | Best For |
|--------|---------------------|---------------|---------|----------|
| `run_with_restart.bat/sh` | ✅ | ❌ | Console only | Quick testing |
| `auto_restart.py` | ✅ | ❌ | File + Console | Production |
| `auto_restart.py --watch` | ✅ | ✅ | File + Console | Development |

## 🛑 Stopping the App

- Press `Ctrl+C` in the terminal
- The app will stop gracefully

## 📝 Logs

- **Simple scripts**: Output goes to console
- **Python wrapper**: Creates `app_restart.log` file

## 🔧 Troubleshooting

**App keeps restarting?**
- Check `app_restart.log` for errors
- Verify database connection
- Check port 5000 is available

**Not starting?**
- Make sure virtual environment is activated
- Check all dependencies are installed: `pip install -r requirements.txt`

## 📚 More Details

See `AUTO_RESTART_GUIDE.md` for:
- Production deployment (systemd, supervisor, PM2)
- Advanced configuration
- Monitoring setup


