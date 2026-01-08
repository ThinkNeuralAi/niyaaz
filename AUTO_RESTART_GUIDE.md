# Auto-Restart Guide for Sakshi.AI

This guide explains how to set up automatic restart functionality for the Sakshi.AI application.

## Options Available

### 1. Simple Batch/Script Files (Recommended for Quick Setup)

#### Windows (`run_with_restart.bat`)
- Automatically restarts the app if it crashes
- Simple to use - just double-click or run from command prompt
- No additional dependencies required

**Usage:**
```batch
run_with_restart.bat
```

#### Linux/Mac (`run_with_restart.sh`)
- Same functionality as Windows version
- Requires execute permissions

**Usage:**
```bash
chmod +x run_with_restart.sh
./run_with_restart.sh
```

### 2. Python Auto-Restart Wrapper (Advanced)

The `auto_restart.py` script provides more advanced features:
- Automatic restart on crash
- Optional file watching (restart on code changes)
- Better logging and monitoring
- Graceful shutdown handling

**Basic Usage (Auto-restart on crash only):**
```bash
python auto_restart.py
```

**Development Mode (Auto-restart on file changes):**
```bash
python auto_restart.py --watch
```

**Custom restart delay:**
```bash
python auto_restart.py --delay 10  # Wait 10 seconds before restart
```

**Options:**
- `--watch`: Enable file watching (restarts on .py, .json, .yaml file changes)
- `--delay N`: Wait N seconds before restarting after crash (default: 5)

## Production Deployment Options

### Option A: Windows Task Scheduler (Windows Server)

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: "When the computer starts"
4. Action: Start a program
   - Program: `C:\TN\Code\Niyaz\run_with_restart.bat`
   - Start in: `C:\TN\Code\Niyaz`
5. Check "Run whether user is logged on or not"
6. Save and test

### Option B: systemd Service (Linux)

Create `/etc/systemd/system/sakshiai.service`:

```ini
[Unit]
Description=Sakshi.AI Video Analytics Platform
After=network.target postgresql.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/Niyaz
Environment="PATH=/path/to/Niyaz/venv/bin"
ExecStart=/path/to/Niyaz/venv/bin/python /path/to/Niyaz/auto_restart.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable sakshiai.service
sudo systemctl start sakshiai.service
sudo systemctl status sakshiai.service
```

### Option C: Supervisor (Linux/Cross-platform)

Install supervisor:
```bash
sudo apt-get install supervisor  # Ubuntu/Debian
# or
pip install supervisor
```

Create `/etc/supervisor/conf.d/sakshiai.conf`:

```ini
[program:sakshiai]
command=/path/to/Niyaz/venv/bin/python /path/to/Niyaz/auto_restart.py
directory=/path/to/Niyaz
user=your_username
autostart=true
autorestart=true
stderr_logfile=/var/log/sakshiai.err.log
stdout_logfile=/var/log/sakshiai.out.log
environment=PATH="/path/to/Niyaz/venv/bin"
```

**Manage with supervisor:**
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start sakshiai
sudo supervisorctl status sakshiai
```

### Option D: PM2 (Node.js-based, Cross-platform)

Install PM2:
```bash
npm install -g pm2
```

Create `ecosystem.config.js`:

```javascript
module.exports = {
  apps: [{
    name: 'sakshiai',
    script: 'auto_restart.py',
    interpreter: 'venv/bin/python',
    cwd: '/path/to/Niyaz',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production'
    }
  }]
};
```

**Usage:**
```bash
pm2 start ecosystem.config.js
pm2 save
pm2 startup  # Enable auto-start on system boot
```

## Monitoring and Logs

### Log Files

- **auto_restart.py**: Creates `app_restart.log` in the project directory
- **Batch/Script files**: Output goes to console (redirect to file if needed)

### Redirecting Output to File

**Windows:**
```batch
run_with_restart.bat > app_output.log 2>&1
```

**Linux/Mac:**
```bash
./run_with_restart.sh > app_output.log 2>&1 &
```

## Stopping the Application

### Simple Scripts
- Press `Ctrl+C` in the terminal
- Close the terminal window

### Python Wrapper
- Press `Ctrl+C` (graceful shutdown)
- The wrapper will stop the app process cleanly

### systemd
```bash
sudo systemctl stop sakshiai.service
```

### Supervisor
```bash
sudo supervisorctl stop sakshiai
```

### PM2
```bash
pm2 stop sakshiai
```

## Troubleshooting

### App keeps restarting immediately
- Check application logs for errors
- Verify database connection
- Check port 5000 is not already in use
- Review `app_restart.log` for details

### App not starting
- Verify virtual environment is activated
- Check Python path is correct
- Ensure all dependencies are installed
- Check file permissions (Linux/Mac)

### File watching not working
- Ensure `--watch` flag is used
- Check that files are being saved (not just edited)
- Verify file permissions

## Recommendations

- **Development**: Use `auto_restart.py --watch` for automatic restarts on code changes
- **Testing**: Use simple batch/script files for quick testing
- **Production**: Use systemd (Linux) or Task Scheduler (Windows) for system-level auto-restart
- **Cloud/Container**: Use supervisor or PM2 for process management

## Notes

- The restart delay (default 5 seconds) prevents rapid restart loops
- File watching is only recommended for development (adds overhead)
- Always test restart functionality before deploying to production
- Monitor logs regularly to catch issues early


