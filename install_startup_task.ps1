<#
.SYNOPSIS
    Install / uninstall a Windows Scheduled Task that auto-starts Sakshi.AI on
    system boot and keeps it running.

.DESCRIPTION
    Registers a Scheduled Task ("SakshiAI-AutoStart") that runs at system
    startup, under the SYSTEM account, whether or not any user is logged on.

    The task launches auto_restart.py, which supervises app.py and restarts it
    automatically if it crashes. So:
        * System reboots            -> task starts auto_restart.py on boot
        * app.py crashes / exits    -> auto_restart.py relaunches it
        * task itself dies          -> Task Scheduler restarts it (RestartCount)

    Must be run from an ELEVATED (Administrator) PowerShell prompt.

.PARAMETER Uninstall
    Remove the scheduled task instead of installing it.

.PARAMETER RestartIntervalHours
    Optional. If > 0, also bounce the whole app (and all cameras) on this fixed
    schedule. Passed through to auto_restart.py --restart-interval. Default 0.

.EXAMPLE
    # From an elevated PowerShell in the project folder:
    powershell -ExecutionPolicy Bypass -File .\install_startup_task.ps1

.EXAMPLE
    # Remove it:
    powershell -ExecutionPolicy Bypass -File .\install_startup_task.ps1 -Uninstall
#>

[CmdletBinding()]
param(
    [switch]$Uninstall,
    [double]$RestartIntervalHours = 0
)

$ErrorActionPreference = 'Stop'
$TaskName = 'SakshiAI-AutoStart'

# --- Require Administrator -------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Error "This script must be run from an ELEVATED (Administrator) PowerShell prompt. Right-click PowerShell -> 'Run as administrator', then re-run."
    exit 1
}

# --- Uninstall path --------------------------------------------------------
if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Green
    } else {
        Write-Host "Scheduled task '$TaskName' does not exist. Nothing to do." -ForegroundColor Yellow
    }
    exit 0
}

# --- Resolve project dir + interpreter -------------------------------------
$ProjectDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ProjectDir)) { $ProjectDir = (Get-Location).Path }

$AppScript = Join-Path $ProjectDir 'app.py'
if (-not (Test-Path $AppScript)) {
    Write-Error "app.py not found in '$ProjectDir'. Run this script from the project root."
    exit 1
}

$RestarterScript = Join-Path $ProjectDir 'auto_restart.py'
if (-not (Test-Path $RestarterScript)) {
    Write-Error "auto_restart.py not found in '$ProjectDir'."
    exit 1
}

# Prefer .venv (Windows interpreter), then venv, then system python.
$Python = $null
foreach ($candidate in @(
    (Join-Path $ProjectDir '.venv\Scripts\python.exe'),
    (Join-Path $ProjectDir 'venv\Scripts\python.exe')
)) {
    if (Test-Path $candidate) { $Python = $candidate; break }
}
if (-not $Python) {
    $sysPy = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($sysPy) {
        $Python = $sysPy
        Write-Host "WARNING: no venv interpreter found; falling back to system python: $Python" -ForegroundColor Yellow
    } else {
        Write-Error "No Python interpreter found (.venv\Scripts\python.exe, venv\Scripts\python.exe, or python on PATH)."
        exit 1
    }
}

Write-Host "Project dir : $ProjectDir"
Write-Host "Interpreter : $Python"
Write-Host "Runs        : auto_restart.py (supervises app.py)"

# --- Build the task --------------------------------------------------------
$argLine = "`"$RestarterScript`""
if ($RestartIntervalHours -gt 0) {
    $argLine += " --restart-interval $RestartIntervalHours"
    Write-Host "Periodic restart: every $RestartIntervalHours hour(s)"
}

$action = New-ScheduledTaskAction -Execute $Python -Argument $argLine -WorkingDirectory $ProjectDir

# Start at boot; 30s delay lets network + PostgreSQL come up first.
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = 'PT30S'

# Run as SYSTEM (no password needed, runs without anyone logged in).
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

# Resilience: restart the task itself if it ever exits, never time out,
# start late if the machine wasn't ready at boot, ignore battery state.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew

# Replace any existing task with the same name (idempotent).
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Existing task '$TaskName' found - replacing it." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description 'Auto-start and supervise the Sakshi.AI video analytics application at system boot.' | Out-Null

Write-Host ""
Write-Host "Installed scheduled task '$TaskName'." -ForegroundColor Green
Write-Host "It will start automatically after every reboot." -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  Start now :  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Status    :  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "  Stop      :  Stop-ScheduledTask  -TaskName '$TaskName'"
Write-Host "  Remove    :  powershell -ExecutionPolicy Bypass -File .\install_startup_task.ps1 -Uninstall"
Write-Host ""
Write-Host "Logs: app_restart.log in the project folder." -ForegroundColor Cyan
