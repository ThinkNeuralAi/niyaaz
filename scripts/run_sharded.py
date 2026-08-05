#!/usr/bin/env python3
"""
Sharded launcher for Sakshi.AI
==============================

One Python process is hard-capped (~88 inferences/sec by the GIL + a single CUDA
context), so 28 cameras in one process crawl at ~1 fps. This launcher runs
SHARD_COUNT worker processes, each handling a disjoint subset of cameras
(SHARD_INDEX=0..N-1), multiplying total throughput and using the idle GPU.

Each worker is the normal app.py; sharding is entirely env-driven:
    SHARD_COUNT = total workers        (e.g. 4)
    SHARD_INDEX = this worker's index  (0..N-1)
Worker i serves its dashboard/API on port 5000+i (shard 0 = the usual :5000).
All workers share the same PostgreSQL DB, so violations/analytics from every
shard show up together.

Usage:
    SHARD_COUNT=4 /usr/bin/python3 scripts/run_sharded.py

Crash handling: any worker that exits is relaunched after a short delay.
Ctrl-C / SIGTERM stops all workers cleanly.
"""
import os
import sys
import time
import signal
import logging
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = "/usr/bin/python3"
SHARD_COUNT = max(1, int(os.getenv("SHARD_COUNT", "4")))
RESTART_DELAY = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SHARDED - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(BASE_DIR, "sharded_launcher.log")),
              logging.StreamHandler()],
)
logger = logging.getLogger("sharded")

procs = {}   # shard_index -> Popen
running = True


def _spawn(i):
    env = dict(os.environ)
    env["SHARD_INDEX"] = str(i)
    env["SHARD_COUNT"] = str(SHARD_COUNT)
    env["PYTHONUNBUFFERED"] = "1"
    logger.info(f"Starting worker shard {i}/{SHARD_COUNT} (port {5000 + i})")
    return subprocess.Popen([PYTHON, "app.py"], cwd=BASE_DIR, env=env)


def _stop_all(*_):
    global running
    running = False
    logger.info("Stopping all shard workers...")
    for i, p in procs.items():
        try:
            p.terminate()
        except Exception:
            pass
    for i, p in procs.items():
        try:
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    logger.info("All workers stopped.")


def main():
    signal.signal(signal.SIGTERM, _stop_all)
    signal.signal(signal.SIGINT, _stop_all)

    logger.info("=" * 60)
    logger.info(f"Sakshi.AI SHARDED launcher: {SHARD_COUNT} workers")
    logger.info("=" * 60)

    for i in range(SHARD_COUNT):
        procs[i] = _spawn(i)
        time.sleep(3)  # stagger startup so model loads / GPU allocations don't collide

    while running:
        for i in range(SHARD_COUNT):
            p = procs.get(i)
            if p is not None and p.poll() is not None and running:
                logger.warning(f"Worker shard {i} exited (code {p.returncode}); "
                               f"restarting in {RESTART_DELAY}s")
                time.sleep(RESTART_DELAY)
                if running:
                    procs[i] = _spawn(i)
        time.sleep(2)


if __name__ == "__main__":
    main()
