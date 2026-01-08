"""
Smoking Detection Module for Sakshi.AI
- Detects cigarette smoking activity using custom YOLO model (best.pt)
- Triggers alerts when smoking is detected
- Takes snapshot and logs timestamp
- Stores detection events in database
"""

import time
import cv2
import numpy as np
import logging
import os
import torch
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from .model_manager import get_shared_model, release_shared_model
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)


class PersonSmokingDetection:
    """Smoking (cigarette) detection with snapshot capture and alerting"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize smoking detection module (cigarette smoking)
        
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
        
        # Detection configuration - Use PyTorch model (TensorRT engines cause segfault)
        self.model_weight = "models/best.pt"
        self.conf_threshold = 0.4  # Confidence threshold for detections (lower for better detection)
        self.nms_iou = 0.45
        
        # Target class for detection (Class 10: 'Smoking' in best.pt)
        self.smoking_class = "Smoking"
        
        # Alert configuration
        self.alert_cooldown = 30.0  # seconds between repeated alerts for same detection
        self.detection_duration_threshold = 2.0  # seconds - smoking must be detected for this long before alert
        self.snapshot_dir = Path("static/person_smoking_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO detector with shared model manager
        logger.info(f"Loading shared YOLO model for smoking detection: {self.model_weight}")
        try:
            self.yolo = get_shared_model(self.model_weight, device='auto')
            logger.info("Shared smoking detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shared smoking detection model: {e}")
            raise
        
        # Initialize GIF recorder for violation snapshots
        self.gif_recorder = AlertGifRecorder(buffer_size=90, gif_duration=3.0, fps=5)
        
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
        self.detection_sessions = 0  # Number of separate detection instances
        self.avg_detection_confidence = 0.0
        self.highest_confidence = 0.0
        
        # Continuous detection tracking
        self.first_detection_time = None  # When smoking was first detected
        self.consecutive_detection = False  # Whether smoking is currently being detected
        
        # Frame processing
        self.frame_count = 0
        self.last_update_time = time.time()
        
        logger.info(f"Smoking Detection initialized for channel {channel_id}")
    
    def __del__(self):
        """Cleanup: Release shared model reference when smoking detection is destroyed"""
        try:
            if hasattr(self, 'model_weight'):
                release_shared_model(self.model_weight, device='auto')
                logger.debug(f"Released shared model reference: {self.model_weight}")
        except Exception as e:
            logger.warning(f"Error releasing shared model: {e}")
    
    def process_frame(self, frame):
        """
        Process a single frame for smoking detection (cigarette smoking)
        
        Args:
            frame: Input video frame
            
        Returns:
            Dict with annotated frame, status, and metadata
        """
        if frame is None or frame.size == 0:
            return frame
        
        self.frame_count += 1
        t_now = time.time()
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and self.frame_count % 100 == 0:
            torch.cuda.empty_cache()
        
        # Add frame to GIF recorder buffer (always running)
        self.gif_recorder.add_frame(frame)
        
        # YOLO inference
        results = self.yolo(frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False)
        
        # Process detections
        detections = []
        smoking_detected = False
        all_detections_debug = []  # For debugging
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo.names.get(cls_id, str(cls_id))
                
                # Log all detections for debugging (every 100 frames)
                if self.frame_count % 100 == 0:
                    all_detections_debug.append(f"{cls_name}({conf:.2f})")
                
                # Only process smoking class (Class 17)
                if cls_name.lower() != self.smoking_class.lower():
                    continue
                
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_name": cls_name,
                    "class_id": cls_id
                })
                
                smoking_detected = True
        
        self.current_detections = detections
        self.detection_count = len(detections)
        
        # Debug logging every 100 frames
        if self.frame_count % 100 == 0:
            if all_detections_debug:
                logger.info(f"🔍 Smoking Detection Frame {self.frame_count} - Detected: {', '.join(all_detections_debug)}")
            else:
                logger.debug(f"Smoking Frame {self.frame_count} - No detections above threshold {self.conf_threshold}")
        
        # Update enhanced metrics
        if smoking_detected:
            self.total_detections += 1
            if len(detections) > self.peak_detections:
                self.peak_detections = len(detections)
            
            # Track confidence levels
            confidences = [d['confidence'] for d in detections]
            if confidences:
                max_conf = max(confidences)
                if max_conf > self.highest_confidence:
                    self.highest_confidence = max_conf
                
                # Update average confidence (running average)
                avg_conf = sum(confidences) / len(confidences)
                if self.avg_detection_confidence == 0.0:
                    self.avg_detection_confidence = avg_conf
                else:
                    # Exponential moving average
                    self.avg_detection_confidence = 0.9 * self.avg_detection_confidence + 0.1 * avg_conf
        
        # Alert trigger logic
        alert_triggered = False
        
        # Track continuous detection time
        if smoking_detected:
            # If this is the first detection, start the timer
            if self.first_detection_time is None:
                self.first_detection_time = t_now
                self.consecutive_detection = True
                self.detection_sessions += 1
                logger.info(f"🚬 Smoking activity detected at {datetime.fromtimestamp(t_now).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check if smoking has been detected continuously for required duration
            detection_duration = t_now - self.first_detection_time
            
            # Trigger alert if duration threshold met and cooldown passed
            if (detection_duration >= self.detection_duration_threshold and 
                (t_now - self.last_alert_time >= self.alert_cooldown)):
                self._trigger_smoking_alert(frame, detections, t_now)
                self.last_alert_time = t_now
                alert_triggered = True
                # Reset detection timer after alert
                self.first_detection_time = None
                self.consecutive_detection = False
        else:
            # Smoking not detected - reset the timer
            if self.first_detection_time is not None:
                logger.debug(f"Smoking detection interrupted after {t_now - self.first_detection_time:.2f}s")
            self.first_detection_time = None
            self.consecutive_detection = False
        
        # Handle GIF recording
        was_recording = self.gif_recorder.is_recording_alert
        
        if was_recording:
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
                    gif_filename = os.path.basename(gif_path) if gif_path else f"smoking_{self.channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                    
                    if self.app:
                        with self.app.app_context():
                            self.db_manager.save_alert_gif(
                                self.channel_id,
                                'person_smoking_alert',
                                {'filename': gif_filename, 'path': gif_path},
                                alert_message=self._last_alert_message,
                                alert_data=self._last_alert_data
                            )
                    else:
                        self.db_manager.save_alert_gif(
                            self.channel_id,
                            'person_smoking_alert',
                            {'filename': gif_filename, 'path': gif_path},
                            alert_message=self._last_alert_message,
                            alert_data=self._last_alert_data
                        )
                    logger.info(f"Smoking alert GIF saved to database: {gif_filename}")
                    # Clear stored alert info
                    self._last_alert_message = None
                    self._last_alert_data = None
                except Exception as e:
                    logger.error(f"Failed to save smoking alert GIF to DB: {e}")
        
        # Track recording state for next frame
        self._was_recording_alert = was_recording
        
        # Draw visualization
        annotated_frame = self._draw_visualization(frame.copy(), detections, t_now, alert_triggered)
        
        return annotated_frame
    
    def _trigger_smoking_alert(self, frame, detections, timestamp):
        """
        Trigger smoking detection alert and save snapshot
        
        Args:
            frame: Current video frame
            detections: List of detection dictionaries
            timestamp: Current timestamp
        """
        try:
            self.total_alerts += 1
            
            # Generate snapshot filename with timestamp
            dt = datetime.fromtimestamp(timestamp)
            snapshot_filename = f"smoking_{self.channel_id}_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
            snapshot_path = self.snapshot_dir / snapshot_filename
            
            # Draw detection boxes on snapshot
            snapshot_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                # Draw red box for smoking
                cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Add label with confidence
                label = f"SMOKING: {conf:.2%}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(snapshot_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 0, 255), -1)
                cv2.putText(snapshot_frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add timestamp and alert info
            alert_text = f"SMOKING ALERT - {dt.strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(snapshot_frame, alert_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(snapshot_frame, f"Channel: {self.channel_id}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(snapshot_frame, f"Detections: {len(detections)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save snapshot
            cv2.imwrite(str(snapshot_path), snapshot_frame)
            file_size = os.path.getsize(snapshot_path)
            
            logger.warning(f"🚬 SMOKING ALERT - Channel {self.channel_id} at {dt.strftime('%H:%M:%S')} - {len(detections)} detection(s)")
            
            # Prepare alert data
            alert_data = {
                'channel_id': self.channel_id,
                'timestamp': dt.isoformat(),
                'detection_count': len(detections),
                'detections': [
                    {
                        'bbox': det['bbox'],
                        'confidence': float(det['confidence']),
                        'class': det['class_name']
                    }
                    for det in detections
                ],
                'alert_type': 'person_smoking_detected'
            }
            
            # Start GIF recording
            alert_info = {
                'type': 'person_smoking_alert',
                'message': f'Smoking detected - {len(detections)} instance(s)',
                'detection_count': len(detections),
                'timestamp': dt.isoformat()
            }
            self.gif_recorder.start_alert_recording(alert_info)
            
            # Store alert info for database saving
            self._last_alert_message = f"Smoking detected - {len(detections)} instance(s)"
            self._last_alert_data = alert_data
            
            # Save to database
            if self.db_manager and self.app:
                try:
                    with self.app.app_context():
                        self.db_manager.log_alert(
                            self.channel_id,
                            "person_smoking_alert",
                            f"Smoking detected - {len(detections)} instance(s)",
                            alert_data
                        )
                        logger.info(f"✓ Smoking alert saved to database")
                except Exception as e:
                    logger.error(f"Failed to save smoking alert to database: {e}")
            
            # Send real-time alert via SocketIO
            if self.socketio:
                try:
                    self.socketio.emit('person_smoking_alert', {
                        'channel_id': self.channel_id,
                        'message': f'Smoking detected on {self.channel_id}',
                        'timestamp': dt.isoformat(),
                        'detection_count': len(detections),
                        'snapshot_url': f'/static/person_smoking_snapshots/{snapshot_filename}',
                        'alert_data': alert_data
                    })
                    logger.info(f"✓ Smoking alert sent via SocketIO for {self.channel_id}")
                except Exception as e:
                    logger.error(f"Failed to send smoking alert via SocketIO: {e}")
            
        except Exception as e:
            logger.error(f"Error triggering smoking alert: {e}", exc_info=True)
    
    def _draw_visualization(self, frame, detections, timestamp, alert_triggered):
        """
        Draw detection boxes and info on frame
        
        Args:
            frame: Video frame to annotate
            detections: List of detection dictionaries
            timestamp: Current timestamp
            alert_triggered: Whether an alert was just triggered
            
        Returns:
            Annotated frame
        """
        # Draw detection boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Use red for smoking
            color = (0, 0, 255)  # Red
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"SMOKING {conf:.2%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw status bar at top
        status_bg_color = (0, 0, 200) if alert_triggered else (50, 50, 50)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), status_bg_color, -1)
        
        # Status text
        dt = datetime.fromtimestamp(timestamp)
        status_text = f"Smoking Detection | {dt.strftime('%Y-%m-%d %H:%M:%S')} | Detections: {len(detections)}"
        if alert_triggered:
            status_text += " | ALERT TRIGGERED!"
        
        cv2.putText(frame, status_text, (10, 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_statistics(self):
        """
        Get current detection statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_detections': self.total_detections,
            'total_alerts': self.total_alerts,
            'current_detections': self.detection_count,
            'peak_detections': self.peak_detections,
            'detection_sessions': self.detection_sessions,
            'avg_confidence': self.avg_detection_confidence,
            'highest_confidence': self.highest_confidence,
            'consecutive_detection': self.consecutive_detection
        }
    
    def get_status(self):
        """
        Get current module status
        
        Returns:
            Dictionary with current status
        """
        return {
            'module': 'PersonSmokingDetection',
            'channel_id': self.channel_id,
            'smoking_detected': len(self.current_detections) > 0,
            'detection_count': len(self.current_detections),
            'total_alerts': self.total_alerts,
            'frame_count': self.frame_count
        }

