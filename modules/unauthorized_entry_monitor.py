"""
Unauthorized Entry Monitor Module
----------------------------------
Monitors for unauthorized entry by detecting persons in restricted areas.

CV Problem Definition:
- Detect persons using YOLO11n model (person class = 0)
- Trigger alerts when person is detected
- Save GIF snapshots of violations
- Log alerts to database

Rules:
- Person detection = YOLO11n class 0
- Alert cooldown = 10 seconds (configurable)
- Saves GIF snapshots when person detected
"""

import cv2
import numpy as np
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from .model_manager import get_shared_model, release_shared_model
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)


class UnauthorizedEntryMonitor:
    """Unauthorized Entry Detection with GIF recording and alerting"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize Unauthorized Entry Monitor
        
        Args:
            channel_id: Unique identifier for this channel
            socketio: Socket.IO instance for real-time updates
            db_manager: Database manager for storing alerts
            app: Flask app instance for database context
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app
        
        # Detection configuration - Use YOLO11n for person detection
        self.model_weight = "models/yolo11n.pt"
        self.conf_threshold = 0.5  # Confidence threshold for detections
        self.nms_iou = 0.45
        self.person_class_id = 0  # Person class in YOLO11n
        
        # Alert configuration
        self.alert_cooldown = 10.0  # seconds between repeated alerts (matching provided code)
        self.snapshot_dir = Path("static/unauthorized_entry_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO detector with shared model manager
        logger.info(f"Loading shared YOLO model for unauthorized entry detection: {self.model_weight}")
        try:
            self.model = get_shared_model(self.model_weight, device='auto')
            logger.info("Shared unauthorized entry detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shared unauthorized entry detection model: {e}")
            raise
        
        # Initialize GIF recorder for 1-minute violation GIFs
        self.gif_recorder = AlertGifRecorder(buffer_size=300, gif_duration=60.0, fps=5)
        
        # Track recording state for GIF management
        self._was_recording_alert = False
        self._last_alert_message = None
        self._last_alert_data = None
        
        # State storage
        self.last_alert_time = 0.0
        self.detection_count = 0
        self.total_alerts = 0
        self.current_detections = []
        
        # Enhanced metrics for analytics
        self.total_detections = 0
        self.peak_detections = 0
        self.detection_sessions = 0
        self.avg_detection_confidence = 0.0
        self.highest_confidence = 0.0
        
        # Frame processing
        self.frame_count = 0
        self.last_update_time = time.time()
        
        logger.info(f"UnauthorizedEntryMonitor initialized for channel {channel_id}")
    
    def __del__(self):
        """Cleanup: Release shared model reference when monitor is destroyed"""
        try:
            if hasattr(self, 'model_weight'):
                release_shared_model(self.model_weight, device='auto')
                logger.debug(f"Released shared model reference: {self.model_weight}")
        except Exception as e:
            logger.warning(f"Error releasing shared model: {e}")
    
    def process_frame(self, frame):
        """
        Process a single frame for unauthorized entry detection
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Annotated frame with detection boxes and alerts
        """
        if frame is None or frame.size == 0:
            return frame
        
        self.frame_count += 1
        current_time = datetime.now()
        now_ts = current_time.timestamp()
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if self.frame_count % 100 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Add frame to GIF recorder buffer (always running)
        self.gif_recorder.add_frame(frame)
        
        # YOLO inference
        try:
            results = self.model(frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False)
            detections = []
            person_detected = False
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                class_names = results[0].names
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Only process person detections (class 0)
                    if cls_id == self.person_class_id:
                        person_detected = True
                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class_name": cls_name,
                            "class_id": cls_id
                        })
            
            self.current_detections = detections
            self.detection_count = len(detections)
            
            # Update metrics
            if detections:
                self.total_detections += len(detections)
                confidences = [d["confidence"] for d in detections]
                self.avg_detection_confidence = sum(confidences) / len(confidences)
                self.highest_confidence = max(confidences) if confidences else 0.0
                if len(detections) > self.peak_detections:
                    self.peak_detections = len(detections)
            
            # Check for alerts
            if person_detected:
                # Check cooldown
                if now_ts - self.last_alert_time >= self.alert_cooldown:
                    self._trigger_alert(detections, current_time, frame)
                    self.last_alert_time = now_ts
            
            # Handle GIF recording
            if self.gif_recorder.is_recording_alert:
                self.gif_recorder.add_alert_frame(frame)
                # stop_alert_recording() is called automatically by add_alert_frame when duration is reached
            
            # Check if recording just finished (was recording, now stopped)
            if self._was_recording_alert and not self.gif_recorder.is_recording_alert:
                # Recording just finished - get GIF info and save to DB
                gif_info = self.gif_recorder.get_last_gif_info()
                if gif_info and self.db_manager and self._last_alert_message:
                    try:
                        # Extract filename from path
                        gif_path = gif_info.get('gif_path', '')
                        gif_filename = os.path.basename(gif_path) if gif_path else f"unauthorized_entry_{self.channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                        
                        # Normalize path for database storage
                        if gif_path:
                            normalized_path = gif_path.replace('\\', '/')
                            if 'static/' in normalized_path:
                                relative_path = normalized_path.split('static/')[1]
                                normalized_gif_path = f"static/{relative_path}"
                            elif normalized_path.startswith('/'):
                                normalized_gif_path = normalized_path.lstrip('/')
                            else:
                                normalized_gif_path = normalized_path
                        else:
                            normalized_gif_path = None
                        
                        gif_payload = {
                            'gif_filename': gif_filename,
                            'gif_path': normalized_gif_path or gif_path,
                            'frame_count': gif_info.get('frame_count', 0),
                            'duration': gif_info.get('duration', 0.0)
                        }
                        if self.app:
                            with self.app.app_context():
                                self.db_manager.save_alert_gif(
                                    self.channel_id,
                                    'unauthorized_entry_alert',
                                    gif_payload,
                                    alert_message=self._last_alert_message,
                                    alert_data=self._last_alert_data
                                )
                        else:
                            self.db_manager.save_alert_gif(
                                self.channel_id,
                                'unauthorized_entry_alert',
                                gif_payload,
                                alert_message=self._last_alert_message,
                                alert_data=self._last_alert_data
                            )
                        logger.info(f"[{self.channel_id}] ✅ Unauthorized entry alert GIF saved to database: {gif_filename}")
                        # Clear stored alert info
                        self._last_alert_message = None
                        self._last_alert_data = None
                    except Exception as e:
                        logger.error(f"[{self.channel_id}] ❌ Failed to save unauthorized entry alert GIF to DB: {e}")
            
            # Track recording state for next frame (use current state, not stale captured value)
            self._was_recording_alert = self.gif_recorder.is_recording_alert
            
            # Draw annotations
            annotated_frame = self._draw_annotations(frame, detections, person_detected)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error processing frame for unauthorized entry: {e}")
            return frame
    
    def _trigger_alert(self, detections, current_time, frame):
        """
        Trigger alert for unauthorized entry detection
        
        Args:
            detections: List of person detections
            current_time: Current timestamp
            frame: Current frame
        """
        try:
            # Check if within store operation hours
            if self.db_manager and self.app:
                try:
                    with self.app.app_context():
                        if not self.db_manager.is_within_operation_hours(self.channel_id):
                            return
                except Exception:
                    pass

            person_count = len(detections)
            alert_message = f"⚠️ UNAUTHORIZED ENTRY: {person_count} person(s) detected"
            
            logger.warning(f"[{self.channel_id}] {alert_message}")
            
            # Start GIF recording (only update tracking data if recording actually starts)
            alert_info = {
                'type': 'unauthorized_entry_alert',
                'message': alert_message,
                'person_count': person_count,
                'channel_id': self.channel_id,
                'timestamp': current_time.isoformat()
            }
            gif_recording_started = False
            if not self.gif_recorder.is_recording_alert:
                self.gif_recorder.start_alert_recording(alert_info)
                # Store alert info for database saving when GIF completes
                self._last_alert_message = alert_message
                self._last_alert_data = {
                    'person_count': person_count,
                    'detections': [
                        {
                            'bbox': d['bbox'],
                            'confidence': d['confidence']
                        } for d in detections
                    ]
                }
                gif_recording_started = True
                logger.info(f"[{self.channel_id}] 📹 Started 1-minute GIF recording for unauthorized entry alert")
            else:
                logger.warning(f"[{self.channel_id}] ⚠️ GIF recording already in progress - skipping GIF for this alert")
            
            # Emit socket event
            if self.socketio:
                self.socketio.emit("unauthorized_entry_alert", {
                    "channel_id": self.channel_id,
                    "person_count": person_count,
                    "timestamp": current_time.isoformat(),
                    "message": alert_message,
                    "detections": [
                        {
                            "bbox": d["bbox"],
                            "confidence": round(d["confidence"], 2)
                        } for d in detections
                    ]
                })
            
            # NOTE: Database logging is deferred to save_alert_gif (called when 60s GIF recording completes)
            # This ensures only ONE database entry per alert — with the 60-second GIF attached
            # Telegram notification is also sent only when the GIF is ready (via save_alert_gif)
            
            self.total_alerts += 1
            self.detection_sessions += 1
            
        except Exception as e:
            logger.error(f"Error triggering unauthorized entry alert: {e}")
    
    def _draw_annotations(self, frame, detections, person_detected):
        """
        Draw annotations on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            person_detected: Whether person was detected
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detection boxes
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box (green for person)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw alert text if person detected
        if person_detected:
            alert_text = f"UNAUTHORIZED ENTRY: {len(detections)} person(s)"
            cv2.putText(annotated, alert_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Draw status info
        status_text = f"Detections: {len(detections)} | Alerts: {self.total_alerts}"
        cv2.putText(annotated, status_text, (20, annotated.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw GIF recording indicator
        if self.gif_recorder.is_recording_alert:
            elapsed = 0
            if self.gif_recorder.alert_start_time:
                elapsed = (datetime.now() - self.gif_recorder.alert_start_time).total_seconds()
            rec_text = f"[REC] GIF {int(elapsed)}s / {int(self.gif_recorder.gif_duration)}s"
            cv2.putText(annotated, rec_text, (annotated.shape[1] - 350, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            # Blinking red circle indicator
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(annotated, (annotated.shape[1] - 370, 35), 8, (0, 0, 255), -1)
        
        return annotated
    
    def get_status(self):
        """
        Get current status of the monitor
        
        Returns:
            Dictionary with status information
        """
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
            "last_alert_time": self.last_alert_time
        }

