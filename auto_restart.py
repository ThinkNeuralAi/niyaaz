"""
Auto-restart wrapper for Sakshi.AI application
Monitors the app and restarts it automatically on crash
Also supports file watching for development mode
"""
import os
import sys
import time
import subprocess
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_restart.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AppRestarter:
    def __init__(self, watch_files=False, restart_delay=5):
        """
        Initialize the app restarter
        
        Args:
            watch_files: If True, restart on file changes (development mode)
            restart_delay: Seconds to wait before restarting after crash
        """
        self.watch_files = watch_files
        self.restart_delay = restart_delay
        self.process = None
        self.running = True
        self.venv_python = self._find_python()
        
        # Files to watch (if watch_files is True)
        self.watch_extensions = {'.py', '.json', '.yaml', '.yml'}
        self.last_modified = {}
        
    def _find_python(self):
        """Find the Python executable to use"""
        # Check for virtual environment
        venv_python = Path('venv') / 'Scripts' / 'python.exe'  # Windows
        if not venv_python.exists():
            venv_python = Path('venv') / 'bin' / 'python'  # Linux/Mac
        if venv_python.exists():
            return str(venv_python)
        return sys.executable
    
    def start_app(self):
        """Start the application"""
        try:
            logger.info("🚀 Starting Sakshi.AI application...")
            logger.info(f"Using Python: {self.venv_python}")
            
            # Start the app process
            self.process = subprocess.Popen(
                [self.venv_python, 'app.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logger.info(f"✅ Application started with PID: {self.process.pid}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start application: {e}")
            return False
    
    def stop_app(self):
        """Stop the application gracefully"""
        if self.process:
            try:
                logger.info(f"🛑 Stopping application (PID: {self.process.pid})...")
                
                # Try graceful shutdown first
                if sys.platform == 'win32':
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                    logger.info("✅ Application stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop
                    logger.warning("⚠️ Application didn't stop, forcing kill...")
                    self.process.kill()
                    self.process.wait()
                    logger.info("✅ Application force stopped")
                
                self.process = None
            except Exception as e:
                logger.error(f"❌ Error stopping application: {e}")
    
    def check_file_changes(self):
        """Check if any watched files have changed"""
        if not self.watch_files:
            return False
        
        changed = False
        for root, dirs, files in os.walk('.'):
            # Skip virtual environment and other directories
            if 'venv' in root or '__pycache__' in root or '.git' in root:
                continue
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.watch_extensions:
                    try:
                        mtime = file_path.stat().st_mtime
                        file_str = str(file_path)
                        
                        if file_str in self.last_modified:
                            if mtime > self.last_modified[file_str]:
                                logger.info(f"📝 File changed: {file_path}")
                                changed = True
                                self.last_modified[file_str] = mtime
                        else:
                            self.last_modified[file_str] = mtime
                    except Exception:
                        pass
        
        return changed
    
    def monitor_app(self):
        """Monitor the application and restart if needed"""
        while self.running:
            if self.process is None:
                # App not running, start it
                if not self.start_app():
                    logger.error("Failed to start app, waiting before retry...")
                    time.sleep(self.restart_delay)
                    continue
            
            # Check if process is still running
            if self.process.poll() is not None:
                # Process has exited
                exit_code = self.process.returncode
                logger.warning(f"⚠️ Application exited with code: {exit_code}")
                
                # Read any remaining output
                if self.process.stdout:
                    try:
                        output = self.process.stdout.read()
                        if output:
                            logger.info(f"Application output:\n{output}")
                    except Exception:
                        pass
                
                self.process = None
                
                if exit_code != 0:
                    logger.error(f"❌ Application crashed! Restarting in {self.restart_delay} seconds...")
                    time.sleep(self.restart_delay)
                else:
                    logger.info("✅ Application exited normally")
                    break
            
            # Check for file changes (if watching)
            if self.watch_files and self.check_file_changes():
                logger.info("🔄 File changes detected, restarting application...")
                self.stop_app()
                time.sleep(1)
                continue
            
            # Read and log output
            if self.process and self.process.stdout:
                try:
                    line = self.process.stdout.readline()
                    if line:
                        print(line.rstrip())
                except Exception:
                    pass
            
            time.sleep(0.1)  # Small delay to prevent CPU spinning
    
    def run(self):
        """Run the restarter"""
        try:
            # Handle Ctrl+C gracefully
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("=" * 60)
            logger.info("Sakshi.AI Auto-Restart Wrapper")
            logger.info("=" * 60)
            logger.info(f"Watch files: {self.watch_files}")
            logger.info(f"Restart delay: {self.restart_delay}s")
            logger.info("Press Ctrl+C to stop")
            logger.info("=" * 60)
            logger.info("")
            
            self.monitor_app()
        except KeyboardInterrupt:
            logger.info("\n🛑 Received stop signal, shutting down...")
        finally:
            self.stop_app()
            logger.info("✅ Restarter stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-restart wrapper for Sakshi.AI')
    parser.add_argument('--watch', action='store_true', 
                       help='Watch for file changes and auto-restart (development mode)')
    parser.add_argument('--delay', type=int, default=5,
                       help='Seconds to wait before restarting after crash (default: 5)')
    
    args = parser.parse_args()
    
    restarter = AppRestarter(watch_files=args.watch, restart_delay=args.delay)
    restarter.run()

if __name__ == '__main__':
    main()


