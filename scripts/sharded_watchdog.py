#!/usr/bin/env python3
"""
Sakshi.AI Sharded Watchdog
==========================

The app now runs as N worker processes (scripts/run_sharded.py), each handling a
subset of cameras. The old single-process freshness watchdog only ever watched
ONE process, so it never noticed when individual shards went "alive but blind"
(idle - cameras dropped, nothing being processed, GPU fell to 0%). That silent
degradation then persisted until a manual restart.

This watchdog is shard-aware. It watches EVERY worker and restarts any single
shard that has gone idle, letting the launcher respawn it.

Health signal (robust and simple)
----------------------------------
A healthy worker processing its cameras runs at high CPU (hundreds of % - it is
doing continuous inference). A degraded/blind worker collapses to ~0-3% CPU
because it has no frames to process. So:

    worker is DEGRADED  <=>  CPU% < CPU_MIN for STALE_LIMIT consecutive checks
                             (only judged after WARMUP_GRACE so cameras have had
                              time to connect)

Also restarts a worker whose RSS exceeds MEM_LIMIT_MB (runaway leak guard).

Restart = SIGTERM the degraded worker's PID; run_sharded.py respawns that shard
with the correct SHARD_INDEX. Per-shard cooldown prevents restart loops (e.g.
during a genuine NVR/network outage where restarting cannot help).

Config (env-overridable)
------------------------
  SWD_CHECK_INTERVAL   seconds between sweeps        (default 60)
  SWD_WARMUP_GRACE     ignore workers younger than   (default 300  => 5 min)
  SWD_CPU_MIN          % below which = idle/degraded (default 20)
  SWD_STALE_LIMIT      consecutive idle checks       (default 3)
  SWD_RESTART_COOLDOWN per-shard cooldown seconds    (default 600)
  SWD_MEM_LIMIT_MB     per-worker RSS ceiling MB     (default 6000)

Run:  /usr/bin/python3 scripts/sharded_watchdog.py
"""
import os
import json
import time
import signal
import logging
import urllib.request
from pathlib import Path

import psutil

BASE_DIR = Path(__file__).resolve().parent.parent

CHECK_INTERVAL = int(os.getenv("SWD_CHECK_INTERVAL", "60"))
WARMUP_GRACE = int(os.getenv("SWD_WARMUP_GRACE", "300"))
CPU_MIN = float(os.getenv("SWD_CPU_MIN", "20"))
STALE_LIMIT = int(os.getenv("SWD_STALE_LIMIT", "3"))
RESTART_COOLDOWN = int(os.getenv("SWD_RESTART_COOLDOWN", "600"))
MEM_LIMIT_MB = int(os.getenv("SWD_MEM_LIMIT_MB", "9000"))  # normal worker ~5.6GB; catch runaway only
# Camera-activity detection (catches the "spinning at 100% CPU with dropped
# cameras" mode that a CPU threshold alone misses). Each shard serves its API on
# port 5000+shard; we count cameras actually producing frames (fps > MIN_FPS).
# A shard is degraded if that count falls below DROP_FRAC of the most it has
# healthily run (its high-water mark), or if its API stops responding.
CAM_MIN_FPS = float(os.getenv("SWD_CAM_MIN_FPS", "0.5"))
CAM_DROP_FRAC = float(os.getenv("SWD_CAM_DROP_FRAC", "0.5"))
CAM_BASELINE_MIN = int(os.getenv("SWD_CAM_BASELINE_MIN", "4"))  # need this many before judging drops
BASE_PORT = int(os.getenv("SWD_BASE_PORT", "5000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SWD - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(BASE_DIR / "logs" / "sharded_watchdog.log"),
              logging.StreamHandler()],
)
logger = logging.getLogger("sharded_watchdog")

running = True
stale = {}          # shard_index -> consecutive degraded checks
last_restart = {}   # shard_index -> monotonic ts of last restart
cam_hwm = {}        # shard_index -> high-water mark of processing cameras


def processing_cameras(shard):
    """
    Count cameras on this shard that are actually producing frames (fps > CAM_MIN_FPS),
    via its API on port BASE_PORT+shard. Returns int, or None if the API is
    unreachable (itself a sign the worker is wedged).
    """
    port = BASE_PORT + shard
    try:
        ac = json.load(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/get_active_channels", timeout=5))
        cams = [c["channel_id"] for c in ac.get("active_channels", [])]
    except Exception:
        return None
    n = 0
    for cid in cams:
        try:
            d = json.load(urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/get_channel_status/{cid}", timeout=5))
            if (d.get("actual_fps") or 0) > CAM_MIN_FPS:
                n += 1
        except Exception:
            continue
    return n


def _stop(signum, _frame):
    global running
    running = False
    logger.info(f"Received signal {signum}, shutting down")


def find_workers():
    """
    Map shard_index -> psutil.Process for every running app.py worker.
    Reads SHARD_INDEX from each process's environment.
    """
    workers = {}
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if not any("python" in c for c in cmd):
                continue
            if not any(os.path.basename(c) == "app.py" for c in cmd):
                continue
            env = p.environ()
            if "SHARD_INDEX" not in env:
                continue
            workers[int(env["SHARD_INDEX"])] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            continue
    return workers


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    logger.info("=" * 60)
    logger.info("Sakshi.AI SHARDED watchdog starting")
    logger.info(f"  check={CHECK_INTERVAL}s  warmup_grace={WARMUP_GRACE}s  "
                f"cpu_min={CPU_MIN}%  stale_limit={STALE_LIMIT} "
                f"(~{CHECK_INTERVAL*STALE_LIMIT}s)  cooldown={RESTART_COOLDOWN}s  "
                f"mem_limit={MEM_LIMIT_MB}MB")
    logger.info("=" * 60)

    # prime CPU counters (first cpu_percent call always returns 0.0)
    for p in find_workers().values():
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(1)

    while running:
        try:
            now = time.monotonic()
            workers = find_workers()
            if not workers:
                logger.warning("No app.py workers found; launcher should respawn them")
                time.sleep(CHECK_INTERVAL)
                continue

            for shard in sorted(workers):
                p = workers[shard]
                try:
                    uptime = time.time() - p.create_time()
                    cpu = p.cpu_percent(interval=1.0)            # recent CPU%
                    rss_mb = p.memory_info().rss / (1024 * 1024)
                except psutil.Error:
                    continue

                # Don't judge a worker that is still warming up (cameras connecting).
                if uptime < WARMUP_GRACE:
                    stale[shard] = 0
                    continue

                mem_over = rss_mb > MEM_LIMIT_MB

                # --- camera-activity signal (primary) ---
                cams = processing_cameras(shard)   # None = API unreachable
                if cams is not None:
                    cam_hwm[shard] = max(cam_hwm.get(shard, 0), cams)
                hwm = cam_hwm.get(shard, 0)
                cam_degraded = (
                    cams is None                                   # API wedged
                    or (hwm >= CAM_BASELINE_MIN and cams < CAM_DROP_FRAC * hwm)
                )
                # --- CPU-idle signal (secondary; catches fully-stalled) ---
                idle_cpu = cpu < CPU_MIN

                degraded = cam_degraded or idle_cpu
                if degraded:
                    stale[shard] = stale.get(shard, 0) + 1
                    why = (f"cams={cams}/hwm={hwm}" if cam_degraded else f"cpu={cpu:.0f}%")
                    logger.warning(f"shard {shard} degraded ({why})  "
                                   f"stale {stale[shard]}/{STALE_LIMIT}")
                else:
                    if stale.get(shard):
                        logger.info(f"shard {shard} recovered: cams={cams} cpu={cpu:.0f}%")
                    stale[shard] = 0

                needs_restart = stale.get(shard, 0) >= STALE_LIMIT or mem_over
                if not needs_restart:
                    continue

                since = now - last_restart.get(shard, 0)
                if since < RESTART_COOLDOWN:
                    logger.warning(f"shard {shard} needs restart but in cooldown "
                                   f"({int(RESTART_COOLDOWN - since)}s left)")
                    continue

                reason = (f"RSS {rss_mb:.0f}MB > {MEM_LIMIT_MB}MB" if mem_over
                          else f"only {cams} cameras processing (hwm {hwm})" if cam_degraded
                          else f"idle cpu {cpu:.0f}%")
                # a restarted shard rebuilds its camera set -> reset its high-water mark
                cam_hwm[shard] = 0
                logger.error(f"RESTARTING shard {shard} (PID {p.pid}) - {reason}. "
                             f"Launcher will respawn it.")
                try:
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        p.kill()
                except psutil.Error as e:
                    logger.error(f"Failed to restart shard {shard}: {e}")

                last_restart[shard] = time.monotonic()
                stale[shard] = 0

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Watchdog loop error: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)

    logger.info("Sharded watchdog stopped")


if __name__ == "__main__":
    main()
