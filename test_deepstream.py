#!/usr/bin/env python3
"""
Sakshi.AI — DeepStream Pipeline Test Script
Tests the DS 8.0 pipeline with a single camera to verify everything works.

Usage:
    python3 test_deepstream.py                    # Uses first camera from channels.json
    python3 test_deepstream.py "rtsp://..."       # Uses a specific RTSP URL
    python3 test_deepstream.py --all              # Uses all cameras from channels.json
"""

import sys
import os
import json
import time
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ds_pipeline import DeepStreamPipeline, DEEPSTREAM_AVAILABLE
from modules.ds_module_adapter import DeepStreamModuleAdapter

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def load_channels_from_json(config_path="config/channels.json"):
    """Load channels from channels.json."""
    with open(config_path) as f:
        cfg = json.load(f)
    return [ch for ch in cfg.get("channels", []) if ch.get("enabled", True) and ch.get("rtsp_url")]


def test_single_camera(rtsp_url):
    """Test with a single RTSP camera."""
    channels = [{
        "channel_id": "test_camera_1",
        "rtsp_url": rtsp_url,
        "enabled": True,
    }]
    return run_test(channels)


def test_all_cameras():
    """Test with all cameras from channels.json."""
    channels = load_channels_from_json()
    if not channels:
        logger.error("No enabled channels found in channels.json")
        return False
    return run_test(channels)


def run_test(channels):
    """Run the DeepStream pipeline test."""
    if not DEEPSTREAM_AVAILABLE:
        logger.error("pyservicemaker not available. Install DeepStream 8.0 SDK.")
        return False

    logger.info(f"Testing with {len(channels)} camera(s)")
    for ch in channels:
        logger.info(f"  {ch.get('channel_id', '?')}: {ch.get('rtsp_url', '?')[:60]}...")

    config = {
        "muxer_width": 1280,
        "muxer_height": 720,
        "enable_tracker": True,
        "enable_sgie": False,
        "frame_extract_interval": 1,
    }

    pipeline = DeepStreamPipeline(config)
    adapter = DeepStreamModuleAdapter(pipeline)

    # Detection callback
    def on_detection(channel_id, detections, frame_num):
        if frame_num % 30 == 0:
            person_count = sum(1 for d in detections if d["class_id"] == 0)
            logger.info(
                f"[{channel_id}] Frame {frame_num}: "
                f"{len(detections)} detections, {person_count} persons"
            )
            for det in detections[:3]:
                logger.info(
                    f"    {det['class_name']} conf={det['confidence']:.2f} "
                    f"tracker_id={det['tracker_id']} "
                    f"bbox=({det['bbox']['x']:.0f},{det['bbox']['y']:.0f},"
                    f"{det['bbox']['w']:.0f},{det['bbox']['h']:.0f})"
                )

    pipeline.register_detection_callback(on_detection)

    # Frame callback
    def on_frame(channel_id, frame):
        logger.debug(f"[{channel_id}] Frame shape: {frame.shape}")

    pipeline.register_frame_callback(on_frame)

    # Build pipeline
    if not pipeline.build(channels):
        logger.error("Pipeline build failed!")
        return False

    # Handle Ctrl+C
    stop = [False]

    def signal_handler(sig, frame):
        stop[0] = True

    signal.signal(signal.SIGINT, signal_handler)

    # Start pipeline
    if not pipeline.start():
        logger.error("Pipeline start failed!")
        return False

    logger.info("Pipeline running. Press Ctrl+C to stop.\n")

    try:
        while not stop[0] and pipeline.is_running:
            time.sleep(5)
            stats = pipeline.get_statistics()
            logger.info("--- Pipeline Statistics ---")
            for ch_id, st in sorted(stats.items()):
                logger.info(
                    f"  {ch_id}: FPS={st['fps']:.1f}, "
                    f"Detections={st['detection_count']}, "
                    f"Frame={'Yes' if st['has_frame'] else 'No'}"
                )

            # Test adapter
            for ch_id in pipeline.get_channel_ids():
                result = adapter.get_results(ch_id)
                if result:
                    logger.info(
                        f"  {ch_id} adapter: {len(result.boxes)} boxes, "
                        f"id={'yes' if result.boxes.id is not None else 'no'}"
                    )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Stopping pipeline...")
        pipeline.stop()

    logger.info("Test complete.")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--all":
            test_all_cameras()
        elif arg.startswith("rtsp://"):
            test_single_camera(arg)
        else:
            print(f"Usage: {sys.argv[0]} [rtsp://url | --all]")
    else:
        # Default: test with first camera from channels.json
        channels = load_channels_from_json()
        if channels:
            test_single_camera(channels[0]["rtsp_url"])
        else:
            print("No cameras found in config/channels.json")
