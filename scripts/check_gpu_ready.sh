#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GPU readiness check for Sakshi.AI
#
# Run this BEFORE the reboot (expected: FAIL - confirms the current mismatch)
# and AFTER the reboot (expected: PASS - confirms CUDA is usable again).
#
#   bash scripts/check_gpu_ready.sh
#
# It verifies, in order:
#   1. nvidia-smi initialises (NVML not mismatched)
#   2. kernel module version == userspace driver version
#   3. torch can see and use the GPU (torch.cuda.is_available())
#   4. the app is actually placing work on the GPU (post-reboot sanity)
# ---------------------------------------------------------------------------
set -uo pipefail
PASS=0; FAIL=0
ok(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
bad(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

echo "=== 1. nvidia-smi / NVML ==="
if smi=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1); then
  ok "nvidia-smi works: $smi"
else
  bad "nvidia-smi failed: $smi"
fi

echo "=== 2. driver kernel-vs-userspace version ==="
kver=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oE 'Kernel Module +[0-9.]+' | grep -oE '[0-9.]+' | head -1)
if [ -n "$kver" ]; then
  ok "loaded kernel module version: $kver"
else
  bad "cannot read /proc/driver/nvidia/version (module not loaded?)"
fi

echo "=== 3. PyTorch CUDA ==="
/usr/bin/python3 - <<'PY'
try:
    import torch
    avail = torch.cuda.is_available()
    print(f"  torch {torch.__version__}  cuda_available={avail}")
    if avail:
        print(f"  ✅ device: {torch.cuda.get_device_name(0)}")
        x = torch.rand(1000, 1000, device='cuda'); (x @ x).sum().item()
        print("  ✅ test matmul on GPU succeeded")
    else:
        print("  ❌ torch cannot use CUDA")
except Exception as e:
    print(f"  ❌ torch CUDA check error: {e}")
PY

echo "=== 4. app GPU usage (post-reboot sanity) ==="
gutil=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
if [ -n "$gutil" ]; then
  echo "  GPU now: $gutil  (after the app warms up this should be clearly non-zero"
  echo "           and system CPU load should fall well below the core count)"
fi

echo ""
echo "=== RESULT: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -eq 0 ]; then
  echo "GPU is READY. Proceed with the TensorRT step."
else
  echo "GPU NOT ready. If this is BEFORE the reboot, that's expected - reboot to fix."
  echo "If AFTER a reboot it still fails, the userspace/kernel driver need to be"
  echo "reconciled (reinstall/reload the matching NVIDIA driver), not just rebooted."
fi
