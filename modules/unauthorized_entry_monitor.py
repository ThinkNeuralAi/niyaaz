"""
Unauthorized Entry Monitor Module
----------------------------------
Monitors for unauthorized entry by detecting persons in restricted areas.

CV Problem Definition:
- Detect persons using YOLO11n model (person class = 0)
- Trigger alerts when person is detected
- Save ONE 60-second GIF snapshot per alert event
- Log alerts to database (and send Telegram via db_manager.save_alert_gif)

Rules:
- Person detection = YOLO11n class 0
- Alert cooldown = 60 seconds (matches GIF duration so one alert == one GIF)
- One 60-second GIF is saved per alert event
"""

import cv2
import logging
import time
import os
from datetime import datetime
from pathlib import Path

from .model_manager import get_shared_model, release_shared_model
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)


class UnauthorizedEntryMonitor:
    """Unauthorized Entry Detection with 1-minute GIF recording and alerting"""

    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Args:
            channel_id: Unique identifier for this channel
            socketio:   Socket.IO instance for real-time updates
            db_manager: Database manager for storing alerts
            app:        Flask app instance for database context
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app

        # Detection configuration
        self.model_weight = "models/yolo11n.pt"
        self.conf_threshold = 0.5
        self.nms_iou = 0.45
        self.person_class_id = 0  # YOLO11n: person

        # Alert configuration
        # IMPORTANT: cooldown >= gif_duration so we only get ONE 60-second GIF per event.
        self.gif_duration = 60.0
        self.alert_cooldown = 60.0

        # Initialize YOLO detector with shared model manager
        logger.info(f"Loading shared YOLO model for unauthorized entry detection: {self.model_weight}")
        try:
            self.model = get_shared_model(self.model_weight, device='auto')
            logger.info("Shared unauthorized entry detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shared unauthorized entry detection model: {e}")
            raise

        # GIF recorder (1-minute violation GIF, ~5 fps)
        self.gif_recorder = AlertGifRecorder(
            buffer_size=300,
            gif_duration=self.gif_duration,
            fps=5,
        )

        # Recording state tracking (for edge-detection: recording -> stopped)
        self._was_recording_alert = False
        self._last_alert_message = None
        self._last_alert_data = None

        # Stats / state
        self.last_alert_time = 0.0
        self.detection_count = 0
        self.total_alerts = 0
        self.current_detections = []

        self.total_detections = 0
        self.peak_detections = 0
        self.detection_sessions = 0
        self.avg_detection_confidence = 0.0
        self.highest_confidence = 0.0

        self.frame_count = 0
        self.last_update_time = time.time()

        logger.info(f"UnauthorizedEntryMonitor initialized for channel {channel_id}")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def cleanup(self):
        """Explicit cleanup — call this from the channel manager when shutting down."""
        try:
            release_shared_model(self.model_weight, device='auto')
            logger.debug(f"Released shared model reference: {self.model_weight}")
        except Exception as e:
            logger.warning(f"Error releasing shared model: {e}")

    def __del__(self):
        # Best-effort fallback only. Prefer cleanup().
        try:
            if hasattr(self, 'model_weight'):
                release_shared_model(self.model_weight, device='auto')
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Main per-frame processing
    # ------------------------------------------------------------------ #
    def process_frame(self, frame):
        """Process a single frame for unauthorized entry detection."""
        if frame is None or frame.size == 0:
            return frame

        self.frame_count += 1
        current_time = datetime.now()
        now_ts = time.time()

        # Always feed the rolling buffer
        self.gif_recorder.add_frame(frame)

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.nms_iou,
                verbose=False,
            )

            detections = []
            person_detected = False

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                class_names = results[0].names

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id != self.person_class_id:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    person_detected = True
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf,
                        "class_name": class_names[cls_id],
                        "class_id": cls_id,
                    })

            self.current_detections = detections
            self.detection_count = len(detections)

            # Metrics
            if detections:
                self.total_detections += len(detections)
                confidences = [d["confidence"] for d in detections]
                self.avg_detection_confidence = sum(confidences) / len(confidences)
                self.highest_confidence = max(confidences)
                if len(detections) > self.peak_detections:
                    self.peak_detections = len(detections)

            # ---------- Alert triggering ----------
            # Only trigger if:
            #   1. A person was detected
            #   2. Cooldown has elapsed
            #   3. We are NOT already recording (extra safety — cooldown==gif_duration
            #      already enforces this, but defense-in-depth is cheap)
            if (
                person_detected
                and (now_ts - self.last_alert_time) >= self.alert_cooldown
                and not self.gif_recorder.is_recording_alert
            ):
                self._trigger_alert(detections, current_time)
                self.last_alert_time = now_ts

            # ---------- Feed alert frames into the active GIF ----------
            if self.gif_recorder.is_recording_alert:
                self.gif_recorder.add_alert_frame(frame)
                # add_alert_frame() auto-stops when duration is reached.

            # ---------- Edge detection: recording just finished ----------
            if self._was_recording_alert and not self.gif_recorder.is_recording_alert:
                self._on_gif_recording_finished()

            self._was_recording_alert = self.gif_recorder.is_recording_alert

            # Annotations for live view
            return self._draw_annotations(frame, detections, person_detected)

        except Exception as e:
            logger.error(f"Error processing frame for unauthorized entry: {e}")
            return frame

    # ------------------------------------------------------------------ #
    # Alerting
    # ------------------------------------------------------------------ #
    def _trigger_alert(self, detections, current_time):
        """Triggered ONCE per alert event. Starts the 60-second GIF."""
        try:
            # Operating-hours check
            if self.db_manager and self.app:
                try:
                    with self.app.app_context():
                        if not self.db_manager.is_within_operation_hours(self.channel_id):
                            return
                except Exception as e:
                    logger.warning(f"[{self.channel_id}] operation_hours check failed: {e}")

            person_count = len(detections)
            alert_message = f"⚠️ UNAUTHORIZED ENTRY: {person_count} person(s) detected"
            logger.warning(f"[{self.channel_id}] {alert_message}")

            alert_info = {
                'type': 'unauthorized_entry_alert',
                'message': alert_message,
                'person_count': person_count,
                'channel_id': self.channel_id,
                'timestamp': current_time.isoformat(),
            }

            # Start the 60-second GIF recording
            self.gif_recorder.start_alert_recording(alert_info)

            # Stash data so we can attach it to the DB row when the GIF finishes
            self._last_alert_message = alert_message
            self._last_alert_data = {
                'person_count': person_count,
                'detections': [
                    {'bbox': d['bbox'], 'confidence': d['confidence']}
                    for d in detections
                ],
            }
            logger.info(f"[{self.channel_id}] 📹 Started 60s GIF recording for unauthorized entry alert")

            # Realtime UI nudge — fired exactly once per event
            if self.socketio:
                self.socketio.emit("unauthorized_entry_alert", {
                    "channel_id": self.channel_id,
                    "person_count": person_count,
                    "timestamp": current_time.isoformat(),
                    "message": alert_message,
                    "detections": [
                        {"bbox": d["bbox"], "confidence": round(d["confidence"], 2)}
                        for d in detections
                    ],
                })

            # NOTE: DB write + Telegram notification are deferred to _on_gif_recording_finished,
            # which runs after the 60-second GIF is complete. This guarantees one DB row +
            # one Telegram message per event, with the GIF attached.
            self.total_alerts += 1
            self.detection_sessions += 1

        except Exception as e:
            logger.error(f"Error triggering unauthorized entry alert: {e}")

    def _on_gif_recording_finished(self):
        """Called the frame after the 60-second GIF finishes recording."""
        gif_info = self.gif_recorder.get_last_gif_info()
        if not (gif_info and self.db_manager and self._last_alert_message):
            # Nothing to persist — just clear state.
            self._last_alert_message = None
            self._last_alert_data = None
            return

        try:
            gif_path = gif_info.get('gif_path', '') or ''
            gif_filename = (
                os.path.basename(gif_path)
                if gif_path
                else f"unauthorized_entry_{self.channel_id}_"
                     f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
            )

            # Normalize path for DB (always relative to static/)
            normalized_gif_path = self._normalize_static_path(gif_path)

            gif_payload = {
                'gif_filename': gif_filename,
                'gif_path': normalized_gif_path or gif_path,
                'frame_count': gif_info.get('frame_count', 0),
                'duration': gif_info.get('duration', 0.0),
            }

            def _save():
                self.db_manager.save_alert_gif(
                    self.channel_id,
                    'unauthorized_entry_alert',
                    gif_payload,
                    alert_message=self._last_alert_message,
                    alert_data=self._last_alert_data,
                )

            if self.app:
                with self.app.app_context():
                    _save()
            else:
                _save()

            logger.info(
                f"[{self.channel_id}] ✅ Unauthorized entry alert GIF saved: {gif_filename}"
            )
        except Exception as e:
            logger.error(
                f"[{self.channel_id}] ❌ Failed to save unauthorized entry alert GIF to DB: {e}"
            )
        finally:
            self._last_alert_message = None
            self._last_alert_data = None

    @staticmethod
    def _normalize_static_path(gif_path: str):
        if not gif_path:
            return None
        normalized = gif_path.replace('\\', '/')
        if 'static/' in normalized:
            return 'static/' + normalized.split('static/', 1)[1]
        if normalized.startswith('/'):
            return normalized.lstrip('/')
        return normalized

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw_annotations(self, frame, detections, person_detected):
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Person {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if person_detected:
            cv2.putText(
                annotated,
                f"UNAUTHORIZED ENTRY: {len(detections)} person(s)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            f"Detections: {len(detections)} | Alerts: {self.total_alerts}",
            (20, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        if self.gif_recorder.is_recording_alert:
            elapsed = 0
            if self.gif_recorder.alert_start_time:
                elapsed = (datetime.now() - self.gif_recorder.alert_start_time).total_seconds()
            rec_text = f"[REC] GIF {int(elapsed)}s / {int(self.gif_recorder.gif_duration)}s"
            cv2.putText(annotated, rec_text,
                        (annotated.shape[1] - 350, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(annotated, (annotated.shape[1] - 370, 35), 8, (0, 0, 255), -1)

        return annotated

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    def get_status(self):
        return {
            "channel_id": self.channel_id,
            "frame_count": self.frame_count,
            "current_detections": self.detection_count,
            "total_detections": self.total_detections,
            "total_alerts": self.total_alerts,
            "peak_detections": self.peak_detections,
            "detection_sessions": self.detection_sessions,
            "avg_confidence": round(self.avg_detection_confidence, 2),
            "highest_confidence": round(self.highest_confidence, 2),
            "is_recording": self.gif_recorder.is_recording_alert,
            "last_alert_time": self.last_alert_time,
        }