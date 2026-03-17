# Sakshi.AI DeepStream Quickstart

## 1) Pre-checks (must pass first)

1. You are on a Linux machine with NVIDIA GPU + proper driver installed.
2. DeepStream 8.x installed and `pyservicemaker` available in your Python venv.
3. Your venv is activated from project root:
   ```bash
   source venv/bin/activate
   pwd  # should be /home/ubuntu/Niyaaz_biryani/niyaaz
   ```
4. Confirm DeepStream Python package:
   ```bash
   python -c "import pyservicemaker as psm; print('pyservicemaker', psm.__version__)"
   ```
5. Confirm model files exist:
   - `models/yolo11n.engine`
   - `models/best.engine` (if using DS full inference or preloading in app.py)

## 2) Configure cameras and modules

1. Open `config/channels.json` and ensure valid RTSP streams:
   ```json
   {
     "channels": [
       {
         "channel_id": "camera_1",
         "rtsp_url": "rtsp://user:pass@IP:554/path",
         "enabled": true,
         "modules": [
           {"type":"QueueMonitor","enabled":true},
           {"type":"DressCodeMonitoring","enabled":true}
         ]
       }
     ]
   }
   ```
2. Ensure `config/default.json` has DeepStream settings:
   - `deepstream.enabled`: `true`
   - `deepstream.decode_only`: `true` (recommended first run) or `false` (full inference)
   - `deepstream.pgie_config`: `config/ds_yolo_primary.yml`
   - `deepstream.sgie_config`: `config/ds_best_detector.yml`
   - `deepstream.tracker_config`: `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml`

## 3) Quick DeepStream health test (recommend first)

Run `test_deepstream.py`:
```bash
python test_deepstream.py --all
```
If your channels are configured and reachable, you should see logs with pipeline stats, fps, and detection counts.

If test fails, inspect:
- `pyservicemaker` import errors
- camera reachability and port 554
- DS tracker/assets path
- `config/_ds_sources.yml` generated path and URI

## 4) Run the app in DeepStream mode

1. Set environment variable (explicit):
   ```bash
   export USE_DEEPSTREAM=true
   ```
2. Start the app:
   ```bash
   python app.py
   ```
3. In logs, verify:
   - `🚀 DeepStream 8.0 mode ENABLED`
   - `DeepStream pipeline started`
   - If fallback occurs: `DeepStream failed to start, falling back to RTSP` or similar.
4. Open dashboard at `http://localhost:5000`.

## 5) Run modules and verify

1. On dashboard, choose a channel and start modules.
2. Verify `/api/deepstream_status` returns:
   ```json
   {
      "enabled": true,
      "is_running": true,
      "streams": {...},
      "channel_ids": [...]
   }
   ```
3. For each channel, check `get_active_channels` and ensure module outputs show expected stats.

## 6) Troubleshooting quick checks

### 6.1. `pyservicemaker` missing
```bash
python -c "import pyservicemaker"
```
If import fails, install DeepStream SDK and ensure Python env sees it.

### 6.2. Camera RTSP unreachable
```bash
python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('CAM_IP',554)); print('ok')"
```

### 6.3. No frames/detections
- Verify channel is enabled and `rtsp_url` in `config/channels.json`.
- Check log lines from `modules/ds_pipeline.py` for source-stream creation.
- In decode-only mode, module still runs local inference.

### 6.4. If DS fallback triggered unexpectedly
- Confirm `USE_DEEPSTREAM=true` in environment.
- Confirm `DEEPSTREAM_AVAILABLE` true in app startup logs.
- Confirm `psm` and `pyservicemaker` imported successfully.

## 7) Optional: run in a one-line launch script
Save this as `run_deepstream.sh` and execute:
```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate
export USE_DEEPSTREAM=true
python app.py
```

## 8) What your code does internally (for confidence)
- `app.py` loads channels from DB or `config/channels.json`, then calls `_load_channels_deepstream`.
- `modules/ds_pipeline.py` builds a single GPU pipeline via pyservicemaker.
- `modules/ds_module_adapter.py` converts DS detections to module-friendly format.
- `modules/shared_multi_module_processor.py` also supports per-channel DS handling if user starts non-shared module.

---

If you want, I can also add a minimal preconfigured `config/channels-demo.json` with a placeholder dummy RTSP example so you can run end-to-end quickly before adding your real cameras.