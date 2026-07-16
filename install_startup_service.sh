#!/bin/bash
#
# install_startup_service.sh
# --------------------------------------------------------------------------
# Linux counterpart to install_startup_task.ps1.
#
# Installs / uninstalls a systemd service ("sakshiai") that auto-starts
# Sakshi.AI on system boot and keeps it running:
#
#     * server reboots        -> systemd starts the service on boot
#     * app.py crashes/exits  -> auto_restart.py relaunches it
#     * supervisor dies       -> systemd restarts the service (Restart=always)
#
# Must be run with sudo (writes to /etc/systemd/system and enables the unit).
#
# Usage:
#     sudo ./install_startup_service.sh            # install + enable + start
#     sudo ./install_startup_service.sh --uninstall
# --------------------------------------------------------------------------
set -euo pipefail

SERVICE_NAME="sakshiai"
UNIT_DEST="/etc/systemd/system/${SERVICE_NAME}.service"

# Resolve the project directory (where this script lives).
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC="${PROJECT_DIR}/${SERVICE_NAME}.service"

# --- Require root ----------------------------------------------------------
if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: this script must be run with sudo/root." >&2
    echo "  sudo ./install_startup_service.sh" >&2
    exit 1
fi

# --- Uninstall path --------------------------------------------------------
if [[ "${1:-}" == "--uninstall" ]]; then
    if systemctl list-unit-files | grep -q "^${SERVICE_NAME}.service"; then
        systemctl disable --now "${SERVICE_NAME}.service" || true
        rm -f "${UNIT_DEST}"
        systemctl daemon-reload
        echo "Removed service '${SERVICE_NAME}'."
    else
        echo "Service '${SERVICE_NAME}' not installed. Nothing to do."
    fi
    exit 0
fi

# --- Sanity checks ---------------------------------------------------------
if [[ ! -f "${PROJECT_DIR}/app.py" ]]; then
    echo "ERROR: app.py not found in ${PROJECT_DIR}." >&2
    exit 1
fi
if [[ ! -f "${PROJECT_DIR}/auto_restart.py" ]]; then
    echo "ERROR: auto_restart.py not found in ${PROJECT_DIR}." >&2
    exit 1
fi
if [[ ! -f "${UNIT_SRC}" ]]; then
    echo "ERROR: ${SERVICE_NAME}.service not found in ${PROJECT_DIR}." >&2
    exit 1
fi

# --- Install ---------------------------------------------------------------
echo "Project dir : ${PROJECT_DIR}"
echo "Interpreter : /usr/bin/python3 ($(/usr/bin/python3 --version 2>&1))"
echo "Runs        : auto_restart.py (supervises app.py)"

install -m 644 "${UNIT_SRC}" "${UNIT_DEST}"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
systemctl restart "${SERVICE_NAME}.service"

echo ""
echo "Installed and started service '${SERVICE_NAME}'."
echo "It will start automatically after every reboot."
echo ""
echo "Useful commands:"
echo "  Status : sudo systemctl status ${SERVICE_NAME}"
echo "  Logs   : sudo journalctl -u ${SERVICE_NAME} -f"
echo "  Stop   : sudo systemctl stop ${SERVICE_NAME}"
echo "  Start  : sudo systemctl start ${SERVICE_NAME}"
echo "  Remove : sudo ./install_startup_service.sh --uninstall"
echo ""
echo "Supervisor log: app_restart.log in the project folder."
