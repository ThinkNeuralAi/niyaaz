#!/usr/bin/env python3
"""
Sakshi.AI Freshness Watchdog
============================

Why this exists
---------------
`auto_restart.py` only reacts when app.py *exits*. It cannot detect the failure
mode we actually hit in production: the process stays alive (and burns CPU) but
its RTSP capture threads die and never reconnect, so no frames flow -> no
detection -> no alerts. This went unnoticed for ~3 days.

This watchdog closes that gap. Every CHECK_INTERVAL it asks a single, robust
question:

    "Is the running app.py actually connected to any cameras right now?"

Health signal
-------------
A healthy 34-camera deployment holds many ESTABLISHED TCP connections to the
camera/NVR IPs. When the capture layer is wedged this drops to zero (only the
localhost PostgreSQL/Telegram sockets remain) - that is exactly what we observed.

So: health = number of ESTABLISHED connections from the app PID to the known
camera host IPs (discovered from the rtsp_links table).

Restart decision (deliberately conservative to avoid restart loops)
-------------------------------------------------------------------
Restart the app ONLY when ALL of these hold:
  1. app.py is running (if it isn't, auto_restart.py/systemd already handles it).
  2. It has had ZERO camera connections for STALE_LIMIT consecutive checks.
  3. The cameras are actually REACHABLE from this host (TCP connect succeeds on
     a sample). If the cameras themselves are down, this is a network/NVR outage,
     NOT an app wedge - restarting would not help, so we log and wait.
  4. We are past RESTART_COOLDOWN since the last watchdog-initiated restart.

Restart is performed by sending SIGTERM to the app.py child. The existing
auto_restart.py supervisor then respawns it (same path the manual recovery used).
The watchdog never launches app.py itself, so it cannot fight the supervisor.

Config (all overridable via environment variables)
--------------------------------------------------
  WATCHDOG_CHECK_INTERVAL   seconds between checks           (default 60)
  WATCHDOG_STALE_LIMIT      consecutive bad checks -> restart (default 5  => ~5 min)
  WATCHDOG_RESTART_COOLDOWN min seconds between restarts     (default 900 => 15 min)
  WATCHDOG_MIN_CONNECTIONS  healthy if camera conns >= this  (default 1)

Run
---
  /usr/bin/python3 scripts/freshness_watchdog.py
or install sakshiai-watchdog.service (see repo root) and:
  sudo systemctl enable --now sakshiai-watchdog
"""

import os
import re
import time
import signal
import socket
import logging
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent          # repo root
ENV_FILE = BASE_DIR / ".env"
HOST_CACHE = BASE_DIR / "scripts" / ".camera_hosts.cache"  # fallback if DB down

CHECK_INTERVAL = int(os.getenv("WATCHDOG_CHECK_INTERVAL", "60"))
STALE_LIMIT = int(os.getenv("WATCHDOG_STALE_LIMIT", "5"))
RESTART_COOLDOWN = int(os.getenv("WATCHDOG_RESTART_COOLDOWN", "900"))
MIN_CONNECTIONS = int(os.getenv("WATCHDOG_MIN_CONNECTIONS", "1"))
# Memory ceiling: restart the app if its RSS stays above MEM_LIMIT_MB for
# MEM_STALE_LIMIT consecutive checks (guards against a runaway RAM leak).
# 0 disables the memory check. Default 24000 MB leaves headroom on a 31 GB box.
MEM_LIMIT_MB = int(os.getenv("WATCHDOG_MEM_LIMIT_MB", "24000"))
MEM_STALE_LIMIT = int(os.getenv("WATCHDOG_MEM_STALE_LIMIT", "3"))
HOST_REFRESH_INTERVAL = 1800                                # re-read camera list every 30 min
REACHABILITY_TIMEOUT = 4.0                                  # seconds per TCP probe
REACHABILITY_SAMPLE = 4                                     # probe at most N cameras

IPV4_PORT_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3}):(\d{1,5})")
IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "watchdog.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("watchdog")


# --------------------------------------------------------------------------- #
# .env / DB helpers
# --------------------------------------------------------------------------- #
def _load_env():
    """Minimal .env reader (no external deps)."""
    env = {}
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception as e:
        logger.warning(f"Could not read {ENV_FILE}: {e}")
    return env


def discover_camera_endpoints():
    """
    Return (hosts, endpoints):
      hosts     - set of camera IPs the app should be connected to
      endpoints - list of (ip, port) pairs to probe for reachability

    Source of truth is the rtsp_links table. Results are cached to disk so a
    transient DB outage doesn't blind the watchdog.
    """
    env = _load_env()
    hosts, endpoints = set(), []
    try:
        cmd = [
            "psql", "-h", env.get("DB_HOST", "localhost"),
            "-U", env.get("DB_USER", "postgres"),
            "-d", env.get("DB_NAME", "sakshiai"),
            "-t", "-A", "-c",
            "SELECT rtsp_url FROM rtsp_links WHERE is_active;",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            env={**os.environ, "PGPASSWORD": env.get("DB_PASSWORD", "")},
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip())

        for url in proc.stdout.splitlines():
            url = url.strip()
            if not url:
                continue
            m = IPV4_PORT_RE.search(url)          # host:port (preferred)
            if m:
                ip, port = m.group(1), int(m.group(2))
                hosts.add(ip)
                endpoints.append((ip, port))
            else:
                m2 = IPV4_RE.search(url)          # host only, assume RTSP 554
                if m2:
                    hosts.add(m2.group(0))
                    endpoints.append((m2.group(0), 554))

        if hosts:
            try:
                HOST_CACHE.write_text("\n".join(f"{ip}:{port}" for ip, port in endpoints))
            except Exception:
                pass
            logger.info(f"Discovered {len(hosts)} camera host(s) from DB: {sorted(hosts)}")
            return hosts, endpoints
        raise RuntimeError("rtsp_links returned no active rows")

    except Exception as e:
        logger.warning(f"DB camera discovery failed ({e}); trying cache {HOST_CACHE}")
        try:
            for line in HOST_CACHE.read_text().splitlines():
                ip, _, port = line.partition(":")
                if ip:
                    hosts.add(ip)
                    endpoints.append((ip, int(port) if port else 554))
            logger.info(f"Loaded {len(hosts)} camera host(s) from cache")
        except Exception as e2:
            logger.error(f"No camera host list available (DB and cache both failed): {e2}")
    return hosts, endpoints


# --------------------------------------------------------------------------- #
# Process / connection inspection
# --------------------------------------------------------------------------- #
def find_app_pid():
    """
    PID of the running app.py worker (NOT auto_restart.py, NOT this watchdog).
    Matches a process whose final argv token's basename is 'app.py'.
    """
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\x00")
            argv = [a.decode(errors="ignore") for a in argv if a]
            if not argv:
                continue
            if any("python" in a for a in argv) and os.path.basename(argv[-1]) == "app.py":
                return int(entry.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return None


def count_camera_connections(pid, camera_hosts):
    """Number of ESTABLISHED TCP connections from `pid` to any camera host IP."""
    if not camera_hosts:
        return -1  # unknown - cannot judge health without a host list
    try:
        out = subprocess.run(
            ["ss", "-tnp"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception as e:
        logger.warning(f"ss failed: {e}")
        return -1

    needle = f"pid={pid},"
    count = 0
    for line in out.splitlines():
        if "ESTAB" not in line or needle not in line:
            continue
        # peer address is the 5th column: <state> <recvq> <sendq> <local> <peer> ...
        parts = line.split()
        if len(parts) < 5:
            continue
        peer_ip = parts[4].rsplit(":", 1)[0].strip("[]")
        if peer_ip in camera_hosts:
            count += 1
    return count


def get_app_rss_mb(pid):
    """Resident set size (RAM actually used) of `pid` in MB, or -1 if unavailable."""
    try:
        for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024  # kB -> MB
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return -1


def cameras_reachable(endpoints):
    """True if at least one sampled camera endpoint accepts a TCP connection."""
    for ip, port in endpoints[:REACHABILITY_SAMPLE]:
        try:
            with socket.create_connection((ip, port), timeout=REACHABILITY_TIMEOUT):
                return True
        except Exception:
            continue
    return False


def restart_app(pid):
    """SIGTERM the app; auto_restart.py respawns it. Escalate to SIGKILL if needed."""
    logger.warning(f"Restarting wedged app.py (PID {pid}) via SIGTERM")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(10):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            logger.info("app.py exited; supervisor will respawn it")
            return
    logger.warning("app.py did not exit after SIGTERM; sending SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    logger.info("=" * 60)
    logger.info("Sakshi.AI Freshness Watchdog starting")
    logger.info(f"  check_interval={CHECK_INTERVAL}s  stale_limit={STALE_LIMIT} "
                f"(~{CHECK_INTERVAL * STALE_LIMIT}s to trigger)  "
                f"restart_cooldown={RESTART_COOLDOWN}s  min_connections={MIN_CONNECTIONS}")
    logger.info(f"  mem_limit={MEM_LIMIT_MB}MB  mem_stale_limit={MEM_STALE_LIMIT} "
                f"({'enabled' if MEM_LIMIT_MB > 0 else 'disabled'})")
    logger.info("=" * 60)

    camera_hosts, endpoints = discover_camera_endpoints()
    last_host_refresh = time.monotonic()
    stale = 0
    mem_stale = 0
    last_restart = 0.0

    running = True

    def _stop(signum, _frame):
        nonlocal running
        logger.info(f"Received signal {signum}, watchdog shutting down")
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while running:
        try:
            now = time.monotonic()
            if now - last_host_refresh > HOST_REFRESH_INTERVAL:
                camera_hosts, endpoints = discover_camera_endpoints()
                last_host_refresh = now

            pid = find_app_pid()
            if pid is None:
                logger.info("app.py not running; leaving respawn to supervisor")
                stale = 0
                mem_stale = 0
                time.sleep(CHECK_INTERVAL)
                continue

            # --- Memory ceiling check (independent of connectivity) ---
            if MEM_LIMIT_MB > 0:
                rss = get_app_rss_mb(pid)
                if rss > MEM_LIMIT_MB:
                    mem_stale += 1
                    logger.warning(f"app.py RSS {rss}MB > limit {MEM_LIMIT_MB}MB "
                                   f"(PID {pid}); mem_stale {mem_stale}/{MEM_STALE_LIMIT}")
                    if mem_stale >= MEM_STALE_LIMIT:
                        if now - last_restart < RESTART_COOLDOWN:
                            wait = int(RESTART_COOLDOWN - (now - last_restart))
                            logger.warning(f"Memory ceiling exceeded but in restart cooldown ({wait}s left)")
                        else:
                            logger.error(f"MEMORY CEILING EXCEEDED: RSS {rss}MB for "
                                         f"{mem_stale} consecutive checks. Restarting.")
                            restart_app(pid)
                            last_restart = time.monotonic()
                            stale = 0
                            mem_stale = 0
                            time.sleep(max(CHECK_INTERVAL, 30))
                            continue
                else:
                    if mem_stale:
                        logger.info(f"Memory recovered: RSS {rss}MB (PID {pid})")
                    mem_stale = 0

            conns = count_camera_connections(pid, camera_hosts)
            if conns < 0:
                logger.warning("Cannot determine camera connections; skipping this check")
                time.sleep(CHECK_INTERVAL)
                continue

            if conns >= MIN_CONNECTIONS:
                if stale:
                    logger.info(f"Recovered: {conns} camera connection(s) (PID {pid})")
                stale = 0
                time.sleep(CHECK_INTERVAL)
                continue

            # conns below threshold -> potential wedge
            stale += 1
            logger.warning(f"No camera connections (PID {pid}); stale {stale}/{STALE_LIMIT}")

            if stale < STALE_LIMIT:
                time.sleep(CHECK_INTERVAL)
                continue

            # Sustained. Is this an app wedge or a real camera/network outage?
            if not cameras_reachable(endpoints):
                logger.error("Cameras are UNREACHABLE from this host - treating as a "
                             "network/NVR outage, NOT restarting the app.")
                time.sleep(CHECK_INTERVAL)
                continue

            if now - last_restart < RESTART_COOLDOWN:
                wait = int(RESTART_COOLDOWN - (now - last_restart))
                logger.warning(f"Wedge confirmed but in restart cooldown ({wait}s left)")
                time.sleep(CHECK_INTERVAL)
                continue

            logger.error("WEDGE CONFIRMED: app alive, 0 camera connections, cameras "
                         "reachable. Restarting.")
            restart_app(pid)
            last_restart = time.monotonic()
            stale = 0
            time.sleep(max(CHECK_INTERVAL, 30))  # give it time to come back

        except Exception as e:
            logger.error(f"Watchdog loop error: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)

    logger.info("Watchdog stopped")


if __name__ == "__main__":
    main()
