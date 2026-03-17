"""
Idle Time Monitor
-----------------

Monitors staff idle time by detecting persons and tracking how long
they remain present. Triggers an alert if a person is detected for
longer than a configurable threshold (default: 15 minutes).

If a monitoring ROI is configured, only persons inside the ROI are
tracked. If no ROI is set, ALL detected persons are tracked.

Uses DeepSORT for persistent person tracking with track IDs.
Saves alert GIFs, stores alerts in the database, and sends
Telegram notifications.
"""

import cv2
import math
import logging
import numpy as np
import json
import time
import os
import torch
from datetime import datetime
from pathlib import Path
from collections import deque

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logging.warning("DeepSort not available. Install with: pip install deep-sort-realtime")

from .model_manager import get_shared_model
from .yolo_detector import YOLODetector
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)


class IdleTimeMonitor:
    """Monitor staff idle time in dining/break areas and alert when threshold is exceeded."""

    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app

        # Model configuration
        self.person_model_path = "models/yolo11n.engine"
        self.conf_threshold = 0.5
        self.nms_iou = 0.45

        # Person class IDs
        self.person_class_id_yolo11n = 0

        # Load person detection model
        self.person_detector = YOLODetector(
            model_path=self.person_model_path,
            confidence_threshold=0.25,
            img_size=640,
            person_class_id=self.person_class_id_yolo11n
        )
        logger.info(f"[{self.channel_id}] IdleTimeMonitor loaded model: {self.person_model_path}")

        # Initialize DeepSORT tracker
        if DEEPSORT_AVAILABLE:
            logger.info(f"[{self.channel_id}] Initializing DeepSORT tracker for idle time monitoring")
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                max_iou_distance=0.7,
                max_cosine_distance=0.3,
                nn_budget=50,
                embedder="mobilenet",
                embedder_gpu=True if torch.cuda.is_available() else False
            )
            self.tracking_enabled = True
        else:
            logger.warning(f"[{self.channel_id}] DeepSORT not available - using simple tracking")
            self.tracker = None
            self.tracking_enabled = False

        # Monitoring ROI (area to monitor, e.g., staff dining room)
        # {"polygon": [(x,y), ...], "bbox": (min_x, min_y, max_x, max_y)}
        self.monitoring_roi = None

        # Settings
        self.settings = {
            "idle_time_threshold": 900.0,   # 15 minutes in seconds
            "alert_cooldown": 600.0,        # 10 minutes between repeated alerts per person
            "track_timeout": 30.0,          # seconds before removing stale tracks
        }

        # Person tracking state
        # {track_id: {
        #   "center": (x, y),
        #   "bbox": [x1, y1, x2, y2],
        #   "first_seen": timestamp,
        #   "last_seen": timestamp,
        #   "in_roi": bool,
        #   "idle_start": timestamp | None,  # When person entered ROI
        #   "alerted": bool,                  # Whether alert was already sent
        #   "last_alert_time": timestamp | None
        # }}
        self.person_tracks = {}

        self.frame_count = 0
        self.total_alerts = 0
        self.last_update_time = time.time()

        # GIF recorder for alert snapshots
        self.gif_recorder = AlertGifRecorder(buffer_size=90, gif_duration=3.0, fps=5)

        # Track recording state for GIF management
        self._was_recording_alert = False
        self._pending_alert_info = None
        self._pending_snapshot_id = None

        self.load_configuration()

    # --- Configuration ---
    def load_configuration(self):
        """Load configuration from channels.json and/or database."""
        try:
            self._load_roi_from_config()
            self._load_settings_from_config()

            if not self.monitoring_roi and self.db_manager:
                roi_config = self.db_manager.get_channel_config(
                    self.channel_id, "IdleTimeMonitor", "monitoring_roi"
                )
                if roi_config:
                    self._set_roi_from_points(roi_config)
                    logger.info(f"[{self.channel_id}] Loaded IdleTimeMonitor ROI from database")

            logger.info(
                f"[{self.channel_id}] IdleTimeMonitor config loaded: "
                f"ROI={'configured' if self.monitoring_roi else 'not set'}, "
                f"threshold={self.settings['idle_time_threshold']}s"
            )
        except Exception as e:
            logger.error(f"Failed to load idle time monitoring configuration: {e}", exc_info=True)

    def _load_roi_from_config(self):
        """Load monitoring ROI from channels.json."""
        try:
            config_path = Path("config/channels.json")
            if not config_path.exists():
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            for channel in config.get('channels', []):
                if channel.get('channel_id') != self.channel_id:
                    continue

                for module in channel.get('modules', []):
                    if module.get('type') != 'IdleTimeMonitor':
                        continue

                    module_config = module.get('config', {})
                    roi_config = module_config.get('monitoring_roi', {})

                    if roi_config and 'points' in roi_config:
                        points = roi_config['points']
                        polygon = []
                        for p in points:
                            if isinstance(p, dict) and 'x' in p and 'y' in p:
                                polygon.append((float(p['x']), float(p['y'])))
                            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                                polygon.append((float(p[0]), float(p[1])))

                        if len(polygon) >= 3:
                            min_x = min(pt[0] for pt in polygon)
                            min_y = min(pt[1] for pt in polygon)
                            max_x = max(pt[0] for pt in polygon)
                            max_y = max(pt[1] for pt in polygon)
                            self.monitoring_roi = {
                                "polygon": polygon,
                                "bbox": (min_x, min_y, max_x, max_y)
                            }
                            logger.info(
                                f"[{self.channel_id}] Loaded IdleTimeMonitor ROI with {len(polygon)} points from channels.json"
                            )
                        return
        except Exception as e:
            logger.error(f"Failed to load ROI from channels.json: {e}", exc_info=True)

    def _load_settings_from_config(self):
        """Load settings from channels.json."""
        try:
            config_path = Path("config/channels.json")
            if not config_path.exists():
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            for channel in config.get('channels', []):
                if channel.get('channel_id') != self.channel_id:
                    continue

                for module in channel.get('modules', []):
                    if module.get('type') != 'IdleTimeMonitor':
                        continue

                    module_config = module.get('config', {})
                    settings_config = module_config.get('settings', {})

                    if settings_config:
                        self.settings.update(settings_config)
                        logger.info(f"[{self.channel_id}] Loaded IdleTimeMonitor settings from channels.json")
                    return
        except Exception as e:
            logger.error(f"Failed to load settings from channels.json: {e}")

    def _set_roi_from_points(self, roi_data):
        """Set ROI from points data (dict with 'points' key or list of points)."""
        points = roi_data if isinstance(roi_data, list) else roi_data.get('points', [])
        polygon = []
        for p in points:
            if isinstance(p, dict) and 'x' in p and 'y' in p:
                polygon.append((float(p['x']), float(p['y'])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                polygon.append((float(p[0]), float(p[1])))

        if len(polygon) >= 3:
            min_x = min(pt[0] for pt in polygon)
            min_y = min(pt[1] for pt in polygon)
            max_x = max(pt[0] for pt in polygon)
            max_y = max(pt[1] for pt in polygon)
            self.monitoring_roi = {
                "polygon": polygon,
                "bbox": (min_x, min_y, max_x, max_y)
            }

    def set_monitoring_roi(self, polygon_points):
        """Set (or update) the monitoring ROI and persist to channels.json."""
        if not polygon_points or len(polygon_points) < 3:
            logger.warning("Invalid polygon for idle time monitoring ROI")
            return

        normalized = []
        for p in polygon_points:
            if isinstance(p, dict) and 'x' in p and 'y' in p:
                normalized.append((float(p['x']), float(p['y'])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                normalized.append((float(p[0]), float(p[1])))

        if len(normalized) < 3:
            logger.warning("Not enough valid points for idle time monitoring ROI")
            return

        min_x = min(pt[0] for pt in normalized)
        min_y = min(pt[1] for pt in normalized)
        max_x = max(pt[0] for pt in normalized)
        max_y = max(pt[1] for pt in normalized)

        self.monitoring_roi = {
            "polygon": normalized,
            "bbox": (min_x, min_y, max_x, max_y)
        }

        # Save to channels.json
        try:
            self._save_roi_to_config()
            logger.info(f"[{self.channel_id}] IdleTimeMonitor ROI updated ({len(normalized)} points)")
        except Exception as e:
            logger.error(f"[{self.channel_id}] Failed to save idle time ROI: {e}", exc_info=True)

    def _save_roi_to_config(self):
        """Save monitoring ROI to channels.json."""
        config_path = Path("config/channels.json")
        if not config_path.exists():
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        for channel in config.get('channels', []):
            if channel.get('channel_id') != self.channel_id:
                continue
            for module in channel.get('modules', []):
                if module.get('type') != 'IdleTimeMonitor':
                    continue
                if 'config' not in module:
                    module['config'] = {}
                points = [{"x": p[0], "y": p[1]} for p in self.monitoring_roi["polygon"]]
                module['config']['monitoring_roi'] = {"points": points}
                break
            break

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    # --- Geometry helpers ---
    def _point_in_polygon(self, point, polygon, bbox):
        x, y = point
        min_x, min_y, max_x, max_y = bbox
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, len(polygon) + 1):
            p2x, p2y = polygon[i % len(polygon)]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # --- Main frame processing ---
    def process_frame(self, frame):
        self.frame_count += 1
        current_time = datetime.now()
        now_ts = current_time.timestamp()

        if frame is None or frame.size == 0:
            return frame

        try:
            # Always feed frames to GIF recorder buffer
            self.gif_recorder.add_frame(frame)

            clean_frame = frame.copy()
            h, w = frame.shape[:2]

            # 1. Detect persons
            person_detections = self.person_detector.detect_persons(frame)

            # Prepare DeepSORT inputs
            ds_inputs = []
            for det in person_detections:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    ds_inputs.append(
                        ([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], det.get("confidence", 0.5), "person")
                    )

            # Update DeepSORT tracker
            if self.tracking_enabled and self.tracker and ds_inputs:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tracks = self.tracker.update_tracks(ds_inputs, frame=frame)
                except Exception as e:
                    logger.error(f"DeepSORT tracking error: {e}")
                    tracks = []
            else:
                tracks = []

            # Process tracked persons
            self._process_tracked_persons(tracks, current_time, h, w)

            # Check for idle time violations
            self._check_violations(current_time, clean_frame)

            # Clean up stale tracks
            self._cleanup_stale_tracks(now_ts)

            # Periodic status log
            if self.frame_count % 300 == 0:
                in_roi_count = sum(1 for t in self.person_tracks.values() if t.get("in_roi"))
                logger.info(
                    f"[{self.channel_id}] IdleTime status (frame {self.frame_count}): "
                    f"{len(self.person_tracks)} tracks, {in_roi_count} in ROI"
                )

            # Send real-time status updates (every 1 second)
            if now_ts - self.last_update_time >= 1.0:
                self._send_realtime_update()
                self.last_update_time = now_ts

            # Handle GIF recording lifecycle
            was_recording = self.gif_recorder.is_recording_alert

            if was_recording:
                self.gif_recorder.add_alert_frame(frame)

            # Check if recording just finished
            if self._was_recording_alert and not self.gif_recorder.is_recording_alert:
                logger.info(f"[{self.channel_id}] 🎬 Idle time alert GIF recording completed!")
                gif_info = self.gif_recorder.get_last_gif_info()
                if gif_info and self.db_manager and self._pending_alert_info:
                    try:
                        gif_path = gif_info.get('gif_path', '')
                        gif_filename = os.path.basename(gif_path) if gif_path else \
                            f"idle_alert_{self.channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"

                        # Convert absolute path to relative for DB
                        snapshot_path = None
                        if gif_path:
                            normalized_path = gif_path.replace('\\', '/')
                            if 'static/' in normalized_path:
                                snapshot_path = 'static/' + normalized_path.split('static/')[1]
                            else:
                                snapshot_path = gif_path

                        # Update IdleTimeViolation record with actual GIF path
                        if self._pending_snapshot_id and self.app:
                            with self.app.app_context():
                                violation = self.db_manager.IdleTimeViolation.query.get(self._pending_snapshot_id)
                                if violation:
                                    violation.snapshot_path = snapshot_path
                                    violation.snapshot_filename = gif_filename
                                    if gif_path and os.path.exists(gif_path):
                                        violation.file_size = os.path.getsize(gif_path)
                                    self.db_manager.db.session.commit()
                                    logger.info(
                                        f"[{self.channel_id}] ✅ Updated idle time violation {self._pending_snapshot_id} with GIF: {snapshot_path}"
                                    )

                        # Save to alert_gifs table
                        gif_payload = {
                            'gif_filename': gif_filename,
                            'gif_path': gif_path,
                            'frame_count': gif_info.get('frame_count', 0),
                            'duration': gif_info.get('duration', 0.0)
                        }

                        alert_info = self._pending_alert_info
                        idle_mins = alert_info.get('idle_time', 0) / 60.0
                        alert_message = (
                            f"Staff idle time alert: Person has been idle for {idle_mins:.1f} minutes "
                            f"in {self.channel_id}"
                        )

                        if self.app:
                            with self.app.app_context():
                                self.db_manager.save_alert_gif(
                                    self.channel_id,
                                    'idle_time_alert',
                                    gif_payload,
                                    alert_message=alert_message,
                                    alert_data=alert_info
                                )

                                # Send Telegram alert with the GIF
                                if snapshot_path and os.path.exists(
                                    gif_path if gif_path else os.path.join("static", snapshot_path)
                                ):
                                    from modules.database import _send_telegram_alert
                                    _send_telegram_alert(
                                        channel_id=self.channel_id,
                                        alert_type='idle_time_alert',
                                        alert_message=alert_message,
                                        snapshot_path=snapshot_path,
                                        alert_data=alert_info
                                    )
                                    logger.info(f"[{self.channel_id}] ✅ Telegram idle time alert sent: {gif_filename}")

                        self._pending_alert_info = None
                        self._pending_snapshot_id = None
                    except Exception as e:
                        logger.error(f"[{self.channel_id}] ❌ Failed to save idle time alert GIF: {e}", exc_info=True)

            self._was_recording_alert = was_recording

            # Draw annotations
            annotated = self._draw_annotations(frame, current_time)
            return annotated

        except Exception as e:
            logger.error(f"IdleTimeMonitor processing error: {e}", exc_info=True)
            return frame

    # --- Tracking ---
    def _process_tracked_persons(self, tracks, current_time, h, w):
        """Process DeepSORT tracks. Timer starts when a person is detected.
        If ROI is configured, only persons inside it are monitored.
        If no ROI is set, ALL detected persons are monitored."""
        now_ts = current_time.timestamp()

        active_track_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            active_track_ids.add(track_id)
            ltrb = track.to_tlbr()
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            # Check if in monitoring ROI (use bottom-center / feet position)
            feet_x = cx / w
            feet_y = bbox[3] / h  # bottom of bounding box

            # If no ROI is configured, treat every detected person as "in ROI"
            if self.monitoring_roi:
                in_roi = self._point_in_polygon(
                    (feet_x, feet_y),
                    self.monitoring_roi["polygon"],
                    self.monitoring_roi["bbox"]
                )
            else:
                in_roi = True  # No ROI => monitor all detected persons

            if track_id in self.person_tracks:
                # Update existing track
                prev_in_roi = self.person_tracks[track_id]["in_roi"]
                self.person_tracks[track_id].update({
                    "center": (cx, cy),
                    "bbox": bbox,
                    "last_seen": now_ts,
                    "in_roi": in_roi,
                })
                # If just entered ROI (or just detected with no ROI), start idle timer
                if in_roi and not prev_in_roi:
                    self.person_tracks[track_id]["idle_start"] = now_ts
                    self.person_tracks[track_id]["alerted"] = False
                # If left ROI, reset idle timer
                elif not in_roi and prev_in_roi:
                    self.person_tracks[track_id]["idle_start"] = None
                    self.person_tracks[track_id]["alerted"] = False
            else:
                # New track — timer starts immediately when person is first detected
                self.person_tracks[track_id] = {
                    "center": (cx, cy),
                    "bbox": bbox,
                    "first_seen": now_ts,
                    "last_seen": now_ts,
                    "in_roi": in_roi,
                    "idle_start": now_ts if in_roi else None,
                    "alerted": False,
                    "last_alert_time": None,
                }

    def _check_violations(self, current_time, frame):
        """Check each tracked person for idle time threshold violation."""
        now_ts = current_time.timestamp()
        threshold = self.settings["idle_time_threshold"]
        cooldown = self.settings["alert_cooldown"]

        for track_id, track in self.person_tracks.items():
            if not track.get("in_roi") or track.get("idle_start") is None:
                continue

            idle_time = now_ts - track["idle_start"]

            if idle_time < threshold:
                continue

            # Check cooldown
            if track.get("last_alert_time") and (now_ts - track["last_alert_time"]) < cooldown:
                continue

            # Trigger alert
            logger.info(
                f"[{self.channel_id}] ⚠️ Idle time violation: track {track_id} idle for {idle_time:.0f}s "
                f"(threshold: {threshold}s)"
            )
            self._trigger_alert(track_id, idle_time, current_time, frame)
            track["alerted"] = True
            track["last_alert_time"] = now_ts

    def _trigger_alert(self, track_id, idle_time, current_time, frame=None):
        """Trigger an idle time alert — start GIF recording, save to DB, emit socketio event."""
        track = self.person_tracks.get(track_id)
        if not track:
            return

        idle_minutes = idle_time / 60.0

        alert_info = {
            "type": "idle_time_alert",
            "track_id": track_id,
            "idle_time": idle_time,
            "idle_minutes": round(idle_minutes, 1),
            "channel_id": self.channel_id,
            "timestamp": current_time.isoformat(),
        }

        # Start GIF recording
        logger.info(f"[{self.channel_id}] 🎬 Starting idle time alert GIF recording")
        self.gif_recorder.start_alert_recording(alert_info)
        self._pending_alert_info = alert_info
        self._pending_snapshot_id = None

        # Placeholder path (updated when GIF completes)
        placeholder_filename = f"idle_{self.channel_id}_{current_time.strftime('%Y%m%d_%H%M%S')}.gif"
        snapshot_path = f"static/idle_time/{placeholder_filename}"

        self.total_alerts += 1

        alert_message = (
            f"Staff idle time alert: Person has been idle for {idle_minutes:.1f} minutes "
            f"(threshold: {self.settings['idle_time_threshold'] / 60:.0f} min)"
        )

        # Emit socketio event
        if self.socketio:
            self.socketio.emit("idle_time_alert", {
                "channel_id": self.channel_id,
                "track_id": track_id,
                "idle_time": round(idle_time, 1),
                "idle_minutes": round(idle_minutes, 1),
                "timestamp": current_time.isoformat(),
                "snapshot_path": snapshot_path,
                "alert_message": alert_message,
            })

        # Save to database
        if self.db_manager:
            try:
                alert_data = {
                    "track_id": track_id,
                    "idle_time": idle_time,
                    "idle_minutes": round(idle_minutes, 1),
                    "threshold": self.settings["idle_time_threshold"],
                }

                if self.app:
                    with self.app.app_context():
                        result = self.db_manager.add_idle_time_violation(
                            channel_id=self.channel_id,
                            idle_time=idle_time,
                            snapshot_path=snapshot_path,
                            timestamp=current_time,
                            alert_data=alert_data,
                        )
                        if result:
                            self._pending_snapshot_id = result
                            logger.info(
                                f"[{self.channel_id}] ✅ Idle time violation saved: ID={result}, "
                                f"idle_time={idle_minutes:.1f}min"
                            )

                # Log to general alerts table
                if self.app:
                    with self.app.app_context():
                        self.db_manager.log_alert(
                            self.channel_id,
                            'idle_time_alert',
                            alert_message,
                            alert_data=alert_data,
                        )
            except Exception as e:
                logger.error(f"[{self.channel_id}] ❌ Failed to save idle time alert: {e}", exc_info=True)

        logger.info(f"[{self.channel_id}] 🚨 Idle time alert #{self.total_alerts}: {alert_message}")

    def _cleanup_stale_tracks(self, now_ts):
        """Remove tracks that haven't been seen recently."""
        timeout = self.settings["track_timeout"]
        to_remove = [tid for tid, t in self.person_tracks.items() if (now_ts - t["last_seen"]) > timeout]
        for tid in to_remove:
            del self.person_tracks[tid]

    def _send_realtime_update(self):
        """Send real-time status update via socketio."""
        if not self.socketio:
            return

        now_ts = time.time()
        idle_persons = []
        for track_id, track in self.person_tracks.items():
            if track.get("in_roi") and track.get("idle_start"):
                idle_time = now_ts - track["idle_start"]
                idle_persons.append({
                    "track_id": track_id,
                    "idle_time": round(idle_time, 1),
                    "idle_minutes": round(idle_time / 60.0, 1),
                    "alerted": track.get("alerted", False),
                })

        self.socketio.emit("idle_time_status", {
            "channel_id": self.channel_id,
            "total_tracks": len(self.person_tracks),
            "in_roi": len(idle_persons),
            "idle_persons": idle_persons,
            "total_alerts": self.total_alerts,
            "threshold_minutes": self.settings["idle_time_threshold"] / 60.0,
        })

    # --- Drawing ---
    def _draw_annotations(self, frame, current_time):
        """Draw ROI, tracking boxes and idle time info on the frame."""
        h, w = frame.shape[:2]
        now_ts = current_time.timestamp()

        # Draw monitoring ROI
        if self.monitoring_roi:
            polygon = self.monitoring_roi["polygon"]
            pts = np.array([(int(p[0] * w), int(p[1] * h)) for p in polygon], np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 255), 2)
            # Label
            if len(pts) > 0:
                cv2.putText(frame, "Idle Monitor Zone", (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Draw tracked persons
        for track_id, track in self.person_tracks.items():
            bbox = track.get("bbox")
            if not bbox:
                continue

            in_roi = track.get("in_roi", False)
            idle_start = track.get("idle_start")

            if in_roi and idle_start:
                idle_time = now_ts - idle_start
                idle_min = idle_time / 60.0
                threshold = self.settings["idle_time_threshold"]

                # Color based on idle time progress
                if idle_time >= threshold:
                    color = (0, 0, 255)  # Red — violation
                elif idle_time >= threshold * 0.7:
                    color = (0, 165, 255)  # Orange — warning
                else:
                    color = (0, 255, 0)  # Green — normal

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                label = f"Idle: {idle_min:.1f}min"
                cv2.putText(
                    frame, label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
            else:
                # Person not in ROI
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (128, 128, 128), 1)

        # Status text
        in_roi_count = sum(1 for t in self.person_tracks.values() if t.get("in_roi"))
        cv2.putText(
            frame,
            f"Idle Monitor | In Zone: {in_roi_count} | Alerts: {self.total_alerts}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        return frame

    def get_statistics(self):
        """Return current monitoring statistics."""
        now_ts = time.time()
        idle_persons = []
        for track_id, track in self.person_tracks.items():
            if track.get("in_roi") and track.get("idle_start"):
                idle_time = now_ts - track["idle_start"]
                idle_persons.append({
                    "track_id": track_id,
                    "idle_time": round(idle_time, 1),
                    "idle_minutes": round(idle_time / 60.0, 1),
                })

        return {
            "total_tracks": len(self.person_tracks),
            "in_roi": len(idle_persons),
            "idle_persons": idle_persons,
            "total_alerts": self.total_alerts,
            "threshold_minutes": self.settings["idle_time_threshold"] / 60.0,
            "roi_configured": self.monitoring_roi is not None,
        }
