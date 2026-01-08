@echo off
REM Quick setup script for PostgreSQL on Windows
REM Run this after installing PostgreSQL

echo ========================================
echo PostgreSQL Setup for Sakshi.AI
echo ========================================
echo.

REM Set PostgreSQL connection details
set /p DB_HOST="Enter PostgreSQL host [localhost]: "
if "%DB_HOST%"=="" set DB_HOST=localhost

set /p DB_PORT="Enter PostgreSQL port [5432]: "
if "%DB_PORT%"=="" set DB_PORT=5432

set /p DB_NAME="Enter database name [sakshiai]: "
if "%DB_NAME%"=="" set DB_NAME=sakshiai

set /p DB_USER="Enter PostgreSQL user [postgres]: "
if "%DB_USER%"=="" set DB_USER=postgres

set /p DB_PASSWORD="Enter PostgreSQL password: "
if "%DB_PASSWORD%"=="" (
    echo Error: Password is required
    exit /b 1
)

echo.
echo Setting environment variables...
setx DB_HOST "%DB_HOST%" >nul 2>&1
setx DB_PORT "%DB_PORT%" >nul 2>&1
setx DB_NAME "%DB_NAME%" >nul 2>&1
setx DB_USER "%DB_USER%" >nul 2>&1
setx DB_PASSWORD "%DB_PASSWORD%" >nul 2>&1

echo.
echo Environment variables set (requires new terminal to take effect)
echo.
echo Current session variables:
set DB_HOST=%DB_HOST%
set DB_PORT=%DB_PORT%
set DB_NAME=%DB_NAME%
set DB_USER=%DB_USER%
set DB_PASSWORD=%DB_PASSWORD%

echo.
echo ========================================
echo Next steps:
echo ========================================
echo 1. Create PostgreSQL database (if not exists):
echo    psql -U %DB_USER% -c "CREATE DATABASE %DB_NAME%;"
echo.
echo 2. Install Python driver:
echo    pip install psycopg2-binary
echo.
echo 3. Run migration script:
echo    python migrate_to_postgresql.py
echo.
echo 4. Start application:
echo    python app.py
echo.
pause



