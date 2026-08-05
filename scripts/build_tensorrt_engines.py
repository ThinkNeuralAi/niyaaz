#!/usr/bin/env python3
"""
Build TensorRT engines for Sakshi.AI  (RUN AFTER THE GPU IS WORKING)
====================================================================

Prerequisite: `bash scripts/check_gpu_ready.sh` reports GPU is READY.
(Engines are tied to this machine's exact GPU + TensorRT/CUDA versions, so they
MUST be built here, after the driver is fixed - not copied from elsewhere.)

What it does
------------
For each active model (the generic person model and the 22-class custom model)
it exports a fresh FP16 TensorRT engine next to the .pt, then validates the
engine loads and runs one inference. It OVERWRITES any stale .engine files.

    /usr/bin/python3 scripts/build_tensorrt_engines.py

After it succeeds, enable acceleration (opt-in, default off):
    * add  USE_TENSORRT=1  to the app's environment (systemd unit or .env), then
    * restart the app and re-check:
        bash scripts/check_gpu_ready.sh   # GPU utilisation should be non-zero
    * watch the load drop.

The code side is already wired: modules/model_manager._maybe_use_engine() picks
up the sibling .engine automatically when USE_TENSORRT=1 and CUDA is available,
and falls back to .pt otherwise - so this is fully reversible (unset the flag).
"""

import os
import sys
import time

MODELS = [
    "models/yolo11n.pt",   # generic person detector (COCO)
    "models/best.pt",      # custom 22-class store model
]
IMG_SIZE = 640
HALF = True  # FP16 - big speedup, negligible accuracy loss for detection


def main():
    # --- hard gate on a working GPU ---
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available. Fix the GPU first "
                  "(bash scripts/check_gpu_ready.sh). Aborting.")
            return 1
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ torch/CUDA check failed: {e}")
        return 1

    from ultralytics import YOLO

    ok, failed = [], []
    for pt in MODELS:
        if not os.path.exists(pt):
            print(f"⚠️  skip (missing): {pt}")
            continue
        engine = pt[:-3] + ".engine"
        print(f"\n=== Building engine for {pt} -> {engine} ===")
        try:
            t0 = time.time()
            # ultralytics writes <name>.engine next to the .pt
            YOLO(pt).export(format="engine", half=HALF, imgsz=IMG_SIZE, device=0)
            # export names it after the .pt stem; ensure it landed where we expect
            if not os.path.exists(engine):
                print(f"❌ expected {engine} not found after export")
                failed.append(pt)
                continue

            # --- validate: load the engine and run one inference ---
            import numpy as np
            m = YOLO(engine)
            dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            m.predict(dummy, verbose=False, device=0)
            print(f"✅ built & validated in {time.time()-t0:.0f}s "
                  f"({os.path.getsize(engine)//(1024*1024)} MB)")
            ok.append(engine)
        except Exception as e:
            print(f"❌ failed to build/validate {pt}: {e}")
            failed.append(pt)

    print("\n" + "=" * 60)
    print(f"Engines built: {len(ok)}  failed: {len(failed)}")
    for e in ok:
        print(f"  ✅ {e}")
    for f in failed:
        print(f"  ❌ {f}")

    if ok and not failed:
        print("\nNEXT: enable acceleration (reversible):")
        print("  1. add  USE_TENSORRT=1  to the systemd unit (Environment=) or .env")
        print("  2. restart the app  (kill the app.py PID; auto_restart respawns it)")
        print("  3. bash scripts/check_gpu_ready.sh   # GPU util should be non-zero")
        print("  4. watch load fall (uptime / top)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
