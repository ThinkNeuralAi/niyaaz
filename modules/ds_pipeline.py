"""
Sakshi.AI — DeepStream 8.0 Pipeline Manager
Uses pyservicemaker Flow API for GPU-accelerated multi-stream video analytics.

Replaces per-camera OpenCV threads with a single DeepStream pipeline:
  RTSP → NVDEC (GPU decode) → nvstreammux (batch) → nvinfer (TensorRT) → nvtracker → Python probe
"""

import os
import sys
import json
import logging
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from urllib.parse import urlparse, quote, urlencode, parse_qs

logger = logging.getLogger(__name__)

try:
    import pyservicemaker as psm
    from pyservicemaker import (
        Pipeline, Flow, BatchMetadataOperator, BufferOperator,
        Probe, Receiver, BufferRetriever, RenderMode
    )
    DEEPSTREAM_AVAILABLE = True
except ImportError:
    DEEPSTREAM_AVAILABLE = False
    logger.warning("pyservicemaker not available — DeepStream pipeline disabled")

    # Fallbacks for class definitions so the module can import in non-DeepStream mode
    class Pipeline:
        pass
    class Flow:
        pass
    class BatchMetadataOperator:
        pass
    class BufferOperator:
        pass
    class Probe:
        pass
    class Receiver:
        pass
    class BufferRetriever:
        pass
    class RenderMode:
        pass

# Tracker library path
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

# nvinfer config paths (relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGIE_CONFIG = os.path.join(BASE_DIR, "config", "ds_yolo_primary.yml")
SGIE_CONFIG = os.path.join(BASE_DIR, "config", "ds_best_detector.yml")


def _encode_rtsp_url(url: str) -> str:
    """
    Encode RTSP URL for GStreamer/nvurisrcbin.
    Passwords with special chars like @ must be URL-encoded.
    """
    if not url or not url.startswith("rtsp://"):
        return url

    # Split at :// to get scheme
    scheme_end = url.index("://") + 3
    rest = url[scheme_end:]

    # Find the last @ which separates credentials from host
    at_idx = rest.rfind("@")
    if at_idx < 0:
        return url  # No credentials

    userinfo = rest[:at_idx]
    hostpath = rest[at_idx + 1:]

    # Split userinfo into user:password
    colon_idx = userinfo.find(":")
    if colon_idx < 0:
        return url  # No password

    user = userinfo[:colon_idx]
    password = userinfo[colon_idx + 1:]

    # URL-encode the password (encode @, spaces, special chars)
    encoded_password = quote(password, safe="")

    return f"rtsp://{user}:{encoded_password}@{hostpath}"


def _check_rtsp_reachable(url: str, timeout: float = 3.0) -> bool:
    """Quick TCP connect check to see if camera RTSP port is reachable."""
    import socket
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or 554
        if not host:
            return False
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False


class DetectionMetadataHandler(BatchMetadataOperator):
    """
    Receives detection metadata from the DeepStream pipeline (after nvinfer + nvtracker).
    Extracts per-frame, per-stream detection results and dispatches to callbacks.
    """

    def __init__(self, pipeline_mgr: "DeepStreamPipeline"):
        super().__init__()
        self.pipeline_mgr = pipeline_mgr

    def handle_metadata(self, batch_meta):
        """Called by DeepStream for every batch of processed frames."""
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            channel_id = self.pipeline_mgr.stream_id_to_channel.get(source_id)
            if channel_id is None:
                continue

            frame_number = frame_meta.frame_number
            pad_index = frame_meta.pad_index

            detections = []
            for obj_meta in frame_meta.object_items:
                rect = obj_meta.rect_params
                detection = {
                    "class_id": obj_meta.class_id,
                    "class_name": obj_meta.label.decode("utf-8", errors="replace") if isinstance(obj_meta.label, bytes) else str(obj_meta.label),
                    "confidence": obj_meta.confidence,
                    "tracker_id": obj_meta.object_id if obj_meta.object_id != 0xFFFFFFFFFFFFFFFF else -1,
                    "tracker_confidence": obj_meta.tracker_confidence,
                    "bbox": {
                        "x": rect.left,
                        "y": rect.top,
                        "w": rect.width,
                        "h": rect.height,
                    },
                    "frame_number": frame_number,
                    "stream_id": source_id,
                    "channel_id": channel_id,
                    "unique_component_id": obj_meta.unique_component_id,
                }

                # Extract classifier metadata if present (from secondary GIE)
                for classifier in obj_meta.classifier_items:
                    detection["classifier"] = {
                        "class_id": classifier.class_id,
                        "label": classifier.label.decode("utf-8", errors="replace") if isinstance(classifier.label, bytes) else str(classifier.label),
                    }

                detections.append(detection)

            # Update FPS counter
            self.pipeline_mgr._update_fps(channel_id)

            # Store latest detections
            with self.pipeline_mgr._detection_lock:
                self.pipeline_mgr._latest_detections[channel_id] = detections

            # Fire detection callbacks
            for cb in self.pipeline_mgr._detection_callbacks:
                try:
                    cb(channel_id, detections, frame_number)
                except Exception as e:
                    logger.error(f"Detection callback error on {channel_id}: {e}")


class FrameExtractorHandler(BufferOperator):
    """
    Extracts raw frame data from GPU buffers for MJPEG streaming and GIF recording.
    Uses buffer.extract(batch_id) to get Tensor, converts to numpy via DLPack.
    """

    def __init__(self, pipeline_mgr: "DeepStreamPipeline"):
        super().__init__()
        self.pipeline_mgr = pipeline_mgr
        self._frame_count = 0

    def handle_buffer(self, buffer):
        """Called for every buffer. Must return True to keep buffer flowing."""
        try:
            self._frame_count += 1

            # Only extract frames every N buffers to reduce CPU load
            if self._frame_count % self.pipeline_mgr._frame_extract_interval != 0:
                return True

            batch_size = buffer.batch_size
            batch_meta = buffer.batch_meta

            for frame_meta in batch_meta.frame_items:
                source_id = frame_meta.source_id
                channel_id = self.pipeline_mgr.stream_id_to_channel.get(source_id)
                if channel_id is None:
                    continue

                batch_id = frame_meta.batch_id
                try:
                    raw_tensor = buffer.extract(batch_id)
                    if raw_tensor is None:
                        continue

                    # Sync GPU to ensure NVDEC has finished decoding this surface.
                    # This is safe now that module processing (PPE YOLO etc.) is on
                    # background worker threads — sync only waits for decode, not inference.
                    # Do NOT use .clone() — it races with NVDEC's hardware engine.
                    try:
                        import torch
                        torch.cuda.synchronize()  # Wait for NVDEC decode completion
                        torch_tensor = torch.utils.dlpack.from_dlpack(raw_tensor)
                        frame_np = torch_tensor.cpu().numpy().copy()
                    except ImportError:
                        frame_np = np.from_dlpack(raw_tensor).copy()

                    # Ensure correct shape (H, W, C) and BGR for OpenCV
                    if frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
                        # CHW → HWC
                        frame_np = np.transpose(frame_np, (1, 2, 0))

                    # Validate frame has correct dimensions
                    if frame_np.ndim != 3 or frame_np.shape[2] not in (3, 4):
                        logger.debug(f"Skipping frame with invalid shape {frame_np.shape} for {channel_id}")
                        continue

                    # Reject tiny frames (GPU decode failures)
                    if frame_np.shape[0] < 32 or frame_np.shape[1] < 32:
                        logger.debug(f"Skipping tiny frame for {channel_id}")
                        continue

                    if frame_np.shape[2] == 4:
                        # RGBA → BGR
                        frame_np = frame_np[:, :, :3][:, :, ::-1].copy()
                    elif frame_np.shape[2] == 3:
                        # RGB → BGR
                        frame_np = frame_np[:, :, ::-1].copy()

                    with self.pipeline_mgr._frame_lock:
                        self.pipeline_mgr._latest_frames[channel_id] = frame_np

                    # Fire frame callbacks
                    for cb in self.pipeline_mgr._frame_callbacks:
                        try:
                            cb(channel_id, frame_np)
                        except Exception as e:
                            logger.error(f"Frame callback error on {channel_id}: {e}")

                except Exception as e:
                    logger.debug(f"Frame extract error for {channel_id}: {e}")

        except Exception as e:
            logger.error(f"FrameExtractor error: {e}")

        return True


class DecodeOnlyMetadataHandler(BatchMetadataOperator):
    """
    Lightweight metadata handler for decode-only mode.
    Only tracks frame numbers and fires callbacks — no detection metadata.
    """

    def __init__(self, pipeline_mgr: "DeepStreamPipeline"):
        super().__init__()
        self.pipeline_mgr = pipeline_mgr

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            channel_id = self.pipeline_mgr.stream_id_to_channel.get(source_id)
            if channel_id is None:
                continue

            frame_number = frame_meta.frame_number
            self.pipeline_mgr._update_fps(channel_id)

            # In decode-only mode, detections are empty — modules do their own YOLO
            # We still fire detection callbacks with empty detections so that
            # downstream code knows a frame has arrived
            for cb in self.pipeline_mgr._detection_callbacks:
                try:
                    cb(channel_id, [], frame_number)
                except Exception as e:
                    logger.error(f"Metadata callback error on {channel_id}: {e}")


class FrameRetrieverHandler(BufferRetriever):
    """
    Retrieves frame buffers via Flow.retrieve() path.
    Flow.retrieve() automatically adds nvvideoconvert + capsfilter(RGB) + appsink,
    so we get proper RGB numpy frames.
    """

    def __init__(self, pipeline_mgr: "DeepStreamPipeline"):
        super().__init__()
        self.pipeline_mgr = pipeline_mgr
        self._frame_count = 0

    def consume(self, buffer) -> int:
        """Called by the appsink for each buffer. Returns consumed bytes (>0 = ok, <0 = error)."""
        try:
            self._frame_count += 1

            if self._frame_count <= 3:
                logger.info(f"FrameRetriever.consume called #{self._frame_count}, batch_size={buffer.batch_size}")

            # Rate limit frame extraction
            if self._frame_count % max(1, self.pipeline_mgr._frame_extract_interval) != 0:
                return 1

            batch_meta = buffer.batch_meta

            for frame_meta in batch_meta.frame_items:
                source_id = frame_meta.source_id
                channel_id = self.pipeline_mgr.stream_id_to_channel.get(source_id)
                if channel_id is None:
                    continue

                batch_id = frame_meta.batch_id
                try:
                    raw_tensor = buffer.extract(batch_id)
                    if raw_tensor is None:
                        continue

                    # Sync GPU before reading the decode surface.
                    # Safe now that PPE etc. are on background threads.
                    try:
                        import torch
                        torch.cuda.synchronize()
                        torch_tensor = torch.utils.dlpack.from_dlpack(raw_tensor)
                        frame_np = torch_tensor.cpu().numpy().copy()
                    except ImportError:
                        frame_np = np.from_dlpack(raw_tensor).copy()

                    # Ensure (H, W, C) layout
                    if frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
                        frame_np = np.transpose(frame_np, (1, 2, 0))

                    # Validate frame has correct dimensions
                    if frame_np.ndim != 3 or frame_np.shape[2] not in (3, 4):
                        logger.debug(f"Skipping frame with invalid shape {frame_np.shape} for {channel_id}")
                        continue

                    # Reject tiny frames (GPU decode failures)
                    if frame_np.shape[0] < 32 or frame_np.shape[1] < 32:
                        logger.debug(f"Skipping tiny frame for {channel_id}")
                        continue

                    # RGB → BGR for OpenCV compatibility
                    if frame_np.ndim == 3 and frame_np.shape[2] == 3:
                        frame_np = frame_np[:, :, ::-1].copy()
                    elif frame_np.ndim == 3 and frame_np.shape[2] == 4:
                        frame_np = frame_np[:, :, :3][:, :, ::-1].copy()

                    with self.pipeline_mgr._frame_lock:
                        self.pipeline_mgr._latest_frames[channel_id] = frame_np

                    # Fire frame callbacks
                    for cb in self.pipeline_mgr._frame_callbacks:
                        try:
                            cb(channel_id, frame_np)
                        except Exception as e:
                            logger.error(f"Frame callback error on {channel_id}: {e}")

                except Exception as e:
                    if self._frame_count <= 5:
                        logger.warning(f"Frame retrieve error for {channel_id} batch_id={batch_id}: {e}")
                    else:
                        logger.debug(f"Frame retrieve error for {channel_id}: {e}")

        except Exception as e:
            logger.error(f"FrameRetriever error: {e}", exc_info=True)

        return 1


class DeepStreamPipeline:
    """
    Manages a single DeepStream 8.0 pipeline processing multiple RTSP streams.

    Usage:
        pipeline = DeepStreamPipeline(config)
        pipeline.build(channels)
        pipeline.start()          # Non-blocking
        ...
        pipeline.stop()
    """

    def __init__(self, config: dict = None):
        if not DEEPSTREAM_AVAILABLE:
            raise RuntimeError("pyservicemaker not available. Install DeepStream 8.0 SDK.")

        self.config = config or {}
        self._pipeline = None
        self._flow = None  # Store the Flow object for starting
        self._pipeline_thread = None
        self.is_running = False
        self._stop_event = threading.Event()

        # Stream mapping
        self.stream_id_to_channel: Dict[int, str] = {}
        self.channel_to_stream_id: Dict[str, int] = {}
        self._channel_rtsp_urls: Dict[str, str] = {}

        # Callbacks
        self._detection_callbacks: List[Callable] = []
        self._frame_callbacks: List[Callable] = []

        # Frame storage
        self._latest_frames: Dict[str, np.ndarray] = {}
        self._frame_lock = threading.Lock()
        self._frame_extract_interval = self.config.get("frame_extract_interval", 2)

        # Detection storage
        self._latest_detections: Dict[str, list] = defaultdict(list)
        self._detection_lock = threading.Lock()

        # FPS tracking
        self._fps_data: Dict[str, dict] = defaultdict(lambda: {
            "frame_count": 0,
            "start_time": time.time(),
            "fps": 0.0,
        })

        # Per-channel frame timestamp tracking (for health/watchdog)
        self._last_frame_time: Dict[str, float] = {}

        # Pipeline config
        self._muxer_width = self.config.get("muxer_width", 1280)
        self._muxer_height = self.config.get("muxer_height", 720)
        self._pgie_config = self.config.get("pgie_config", PGIE_CONFIG)
        self._sgie_config = self.config.get("sgie_config", SGIE_CONFIG)
        self._enable_tracker = self.config.get("enable_tracker", True)
        self._enable_sgie = self.config.get("enable_sgie", False)
        self._decode_only = self.config.get("decode_only", True)  # Hybrid: DS decode + Python inference
        self._pgie_batch_size = self.config.get("pgie_batch_size", 4)
        self._pgie_interval = self.config.get("pgie_interval", 0)
        self._tracker_lib = self.config.get("tracker_lib", TRACKER_LIB)
        self._tracker_config = self.config.get("tracker_config", TRACKER_CONFIG)

        logger.info(f"DeepStreamPipeline initialized (DS 8.0 / pyservicemaker, "
                     f"mode={'decode-only' if self._decode_only else 'full-inference'})")

    def build(self, channels: list) -> bool:
        """
        Build the DeepStream pipeline for the given channels.

        Args:
            channels: List of dicts with 'channel_id', 'rtsp_url', 'enabled' keys
                      (from channels.json or database)

        Returns:
            True if pipeline was built successfully
        """
        enabled = [ch for ch in channels if ch.get("enabled", True) and ch.get("rtsp_url")]
        if not enabled:
            logger.error("No enabled channels with RTSP URLs found")
            return False

        # Pre-check: verify cameras are reachable (TCP connect test)
        # Unreachable cameras will block nvstreammux indefinitely
        # Use parallel checks with retries to avoid slow sequential timeouts
        from concurrent.futures import ThreadPoolExecutor, as_completed

        check_timeout = self.config.get("rtsp_check_timeout", 8.0)
        max_retries = self.config.get("rtsp_check_retries", 2)

        def _check_with_retry(ch):
            url = ch["rtsp_url"]
            for attempt in range(max_retries):
                if _check_rtsp_reachable(url, timeout=check_timeout):
                    return ch, True
                if attempt < max_retries - 1:
                    logger.info(f"Camera {ch['channel_id']} retry {attempt + 2}/{max_retries}...")
                    time.sleep(1)
            return ch, False

        reachable = []
        unreachable = []
        logger.info(f"🔍 Checking {len(enabled)} cameras in parallel (timeout={check_timeout}s, retries={max_retries})...")

        with ThreadPoolExecutor(max_workers=min(len(enabled), 16)) as pool:
            futures = {pool.submit(_check_with_retry, ch): ch for ch in enabled}
            for future in as_completed(futures):
                ch, ok = future.result()
                if ok:
                    reachable.append(ch)
                else:
                    unreachable.append(ch)
                    logger.warning(f"Camera {ch['channel_id']} unreachable after {max_retries} attempts, skipping: {ch['rtsp_url'][:60]}...")

        if unreachable:
            logger.warning(f"{len(unreachable)} camera(s) unreachable, using {len(reachable)}/{len(enabled)}")
            self._unreachable_channels = [ch["channel_id"] for ch in unreachable]

        if not reachable:
            logger.error("No reachable cameras found")
            return False

        enabled = reachable

        # Build stream URIs and mappings
        stream_uris = []
        for idx, ch in enumerate(enabled):
            channel_id = ch["channel_id"]
            raw_url = ch["rtsp_url"]
            encoded_url = _encode_rtsp_url(raw_url)

            self.stream_id_to_channel[idx] = channel_id
            self.channel_to_stream_id[channel_id] = idx
            self._channel_rtsp_urls[channel_id] = encoded_url

            stream_uris.append(encoded_url)
            logger.info(f"Stream {idx}: {channel_id} → {raw_url[:60]}...")

        num_streams = len(stream_uris)
        batch_size = max(num_streams, self._pgie_batch_size)

        # Create source config YAML dynamically
        source_yaml_path = os.path.join(BASE_DIR, "config", "_ds_sources.yml")
        self._write_source_config(stream_uris, source_yaml_path)

        try:
            self._pipeline = Pipeline("sakshiai-pipeline")

            if self._decode_only:
                # ====== DECODE-ONLY MODE ======
                # batch_capture → attach(metadata_probe) → retrieve(frame_retriever)
                # retrieve() adds nvvideoconvert + capsfilter(RGB) + appsink internally
                # so buffer.extract() gets proper RGB tensors.
                flow = Flow(self._pipeline).batch_capture(
                    input=source_yaml_path,
                    width=self._muxer_width,
                    height=self._muxer_height,
                    **{
                        "batch-push-timeout": 40000,   # 40ms — don't wait forever for slow sources
                        "drop-pipeline-eos": True,      # Don't EOS pipeline if one source disconnects
                        "live-source": True,            # Treat as live — don't block on missing sources
                    }
                )

                # Metadata probe — tracks which frames are flowing per channel
                metadata_handler = DecodeOnlyMetadataHandler(self)
                metadata_probe = Probe("metadata-probe", metadata_handler)
                flow = flow.attach(what=metadata_probe)

                # Frame retriever — consumes RGB buffers via appsink
                frame_retriever = FrameRetrieverHandler(self)
                flow = flow.retrieve(frame_retriever)

                self._flow = flow

            else:
                # ====== FULL INFERENCE MODE ======
                # Flow: batch_capture → infer → track → probe → render(DISCARD)
                flow = Flow(self._pipeline).batch_capture(
                    input=source_yaml_path,
                    width=self._muxer_width,
                    height=self._muxer_height,
                    **{
                        "batch-push-timeout": 40000,
                        "drop-pipeline-eos": True,
                        "live-source": True,
                    }
                )

                # Override batch-size to engine max (8) since Flow auto-sets it
                # to num_streams which exceeds the TRT engine's max batch size.
                # nvinfer will sub-batch internally when streams > batch-size.
                flow = flow.infer(self._pgie_config, **{"batch-size": min(batch_size, 8)})

                if self._enable_tracker:
                    flow = flow.track(
                        **{
                            "ll-lib-file": self._tracker_lib,
                            "ll-config-file": self._tracker_config,
                        }
                    )

                if self._enable_sgie:
                    flow = flow.infer(self._sgie_config, **{"batch-size": min(batch_size, 8)})

                # Metadata probe for nvinfer detections + frame numbers
                metadata_handler = DetectionMetadataHandler(self)
                metadata_probe = Probe("detection-probe", metadata_handler)
                flow = flow.attach(what=metadata_probe)

                # Frame retriever — consumes RGB buffers via appsink
                frame_retriever = FrameRetrieverHandler(self)
                flow = flow.retrieve(frame_retriever)

                self._flow = flow

            mode_str = "decode-only" if self._decode_only else "full-inference"
            logger.info(
                f"Pipeline built ({mode_str}): {num_streams} streams, batch_size={batch_size}"
                + (f", tracker={'ON' if self._enable_tracker else 'OFF'}, "
                   f"sgie={'ON' if self._enable_sgie else 'OFF'}"
                   if not self._decode_only else "")
            )
            return True

        except Exception as e:
            logger.error(f"Pipeline build failed: {e}", exc_info=True)
            return False

    def _write_source_config(self, uris: list, path: str):
        """Write a DS 8.0 source-list YAML file for the given RTSP URIs."""
        lines = [
            "source-list:",
        ]
        for idx, uri in enumerate(uris):
            lines.append(f'- uri: "{uri}"')
            lines.append(f"  sensor-id: stream_{idx}")
            lines.append(f"  sensor-name: camera_{idx}")

        lines.append("source-config:")
        lines.append('  source-bin: "nvurisrcbin"')
        lines.append("  properties:")
        lines.append("    rtsp-reconnect-interval: 5")
        lines.append("    latency: 500")  # 500ms jitterbuffer for RTSP network jitter
        # Enlarge NVDEC decode surface pool — prevents buffer reuse while reading.
        # Default is 1 which is too few with 28+ cameras; decoded surface gets
        # recycled before .cpu().numpy().copy() finishes → ghosting/tearing.
        lines.append("    num-extra-surfaces: 4")
        # Drop frames under load rather than accumulating stale buffers
        lines.append("    drop-frame-interval: 0")
        # Use device memory for decode surfaces (faster, avoids host staging)
        lines.append("    cudadec-memtype: 0")
        # Use TCP directly to avoid 5-second UDP timeout per camera
        lines.append("    select-rtp-protocol: 4")  # 4 = TCP

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.debug(f"Wrote source config to {path} ({len(uris)} streams)")

    def start(self) -> bool:
        """Start the pipeline in a background thread. Non-blocking."""
        if self._flow is None:
            logger.error("Pipeline not built. Call build() first.")
            return False

        if self.is_running:
            logger.warning("Pipeline already running")
            return True

        self._stop_event.clear()
        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline,
            name="ds-pipeline",
            daemon=True,
        )
        self._pipeline_thread.start()

        # Wait briefly for pipeline to start
        time.sleep(2)
        self.is_running = True
        logger.info("DeepStream pipeline started")
        return True

    def _run_pipeline(self):
        """Run the pipeline (blocking). Called in background thread."""
        try:
            logger.info("Pipeline thread starting...")

            def on_message(message):
                msg_type = type(message).__name__
                if msg_type == "EOSMessage":
                    logger.info("Pipeline EOS received")
                elif msg_type == "StateTransitionMessage":
                    pass  # Normal state changes
                else:
                    logger.debug(f"Pipeline message: {msg_type}")

            # flow() is blocking — it calls pipeline.start(on_message).wait() internally
            self._flow(on_message=on_message)

        except Exception as e:
            logger.error(f"Pipeline thread error: {e}", exc_info=True)
        finally:
            self.is_running = False
            logger.info("Pipeline thread stopped")

    def stop(self):
        """Stop the pipeline gracefully."""
        if not self.is_running and self._flow is None:
            return

        logger.info("Stopping DeepStream pipeline...")
        self._stop_event.set()

        try:
            if self._flow:
                self._flow.pipeline.stop()
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")

        if self._pipeline_thread and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=10)

        self.is_running = False
        logger.info("DeepStream pipeline stopped")

    # ─── Public API ───────────────────────────────────────────────

    def register_detection_callback(self, callback: Callable):
        """Register callback: callback(channel_id, detections_list, frame_number)"""
        self._detection_callbacks.append(callback)

    def register_frame_callback(self, callback: Callable):
        """Register callback: callback(channel_id, frame_ndarray)"""
        self._frame_callbacks.append(callback)

    def get_latest_frame(self, channel_id: str) -> Optional[np.ndarray]:
        """Get latest decoded frame for a channel (BGR numpy array)."""
        with self._frame_lock:
            frame = self._latest_frames.get(channel_id)
            if frame is None:
                return None
            # Final safety check: reject structurally invalid frames
            if frame.ndim != 3 or frame.shape[2] not in (3, 4) or frame.shape[0] < 32 or frame.shape[1] < 32:
                return None
            return frame

    def get_latest_detections(self, channel_id: str) -> list:
        """Get latest detection results for a channel."""
        with self._detection_lock:
            return list(self._latest_detections.get(channel_id, []))

    def get_fps(self, channel_id: str) -> float:
        """Get current FPS for a channel."""
        return self._fps_data.get(channel_id, {}).get("fps", 0.0)

    def get_channel_health(self) -> dict:
        """Get health status for all channels in the pipeline."""
        now = time.time()
        health = {}
        for stream_id, channel_id in self.stream_id_to_channel.items():
            last_frame = self._last_frame_time.get(channel_id, 0)
            fps = self.get_fps(channel_id)
            age = now - last_frame if last_frame > 0 else -1
            health[channel_id] = {
                "stream_id": stream_id,
                "fps": round(fps, 1),
                "last_frame_age_sec": round(age, 1),
                "has_frame": channel_id in self._latest_frames,
                "status": "ok" if age >= 0 and age < 30 else ("stale" if age >= 30 else "no_frames"),
            }
        return health

    def get_stale_channels(self, threshold_seconds: float = 60.0) -> List[str]:
        """Return list of channel_ids that haven't received a frame in threshold_seconds."""
        now = time.time()
        stale = []
        for channel_id in self.channel_to_stream_id:
            last = self._last_frame_time.get(channel_id, 0)
            if last == 0 or (now - last) > threshold_seconds:
                stale.append(channel_id)
        return stale

    def get_statistics(self) -> dict:
        """Get pipeline statistics for all streams."""
        stats = {}
        now = time.time()
        for stream_id, channel_id in self.stream_id_to_channel.items():
            with self._detection_lock:
                det_count = len(self._latest_detections.get(channel_id, []))
            last_frame = self._last_frame_time.get(channel_id, 0)
            stats[channel_id] = {
                "stream_id": stream_id,
                "fps": self.get_fps(channel_id),
                "has_frame": channel_id in self._latest_frames,
                "detection_count": det_count,
                "last_frame_age_sec": round(now - last_frame, 1) if last_frame > 0 else -1,
            }
        return stats

    def get_channel_ids(self) -> list:
        """Get list of all channel IDs in the pipeline."""
        return list(self.channel_to_stream_id.keys())

    def _update_fps(self, channel_id: str):
        """Update FPS counter and last-frame timestamp for a channel."""
        self._last_frame_time[channel_id] = time.time()
        data = self._fps_data[channel_id]
        data["frame_count"] += 1
        elapsed = time.time() - data["start_time"]
        if elapsed >= 5.0:
            data["fps"] = data["frame_count"] / elapsed
            data["frame_count"] = 0
            data["start_time"] = time.time()
