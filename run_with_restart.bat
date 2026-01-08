@echo off
REM Auto-restart script for Sakshi.AI application (Windows)
REM This script will automatically restart the app if it crashes

setlocal enabledelayedexpansion

:start
echo ========================================
echo Starting Sakshi.AI Application...
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the application
python app.py

REM Check exit code
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% NEQ 0 (
    echo.
    echo ========================================
    echo Application crashed with exit code %EXIT_CODE%
    echo Waiting 5 seconds before restart...
    echo ========================================
    echo.
    timeout /t 5 /nobreak >nul
    goto start
) else (
    echo.
    echo ========================================
    echo Application exited normally
    echo ========================================
    pause
)

endlocal


