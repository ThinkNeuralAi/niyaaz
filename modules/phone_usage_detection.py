"""
Mobile Phone Usage Detection Module
Detects employees using mobile phones in restricted areas
Uses YOLO model to detect 'phone' class and triggers alerts
"""

import cv2
import numpy as np
import time
import logging
import os
import json
from datetime import datetime
from pathlib import Path

from .gif_alert_helper import GifAlertHelper

logger = logging.getLogger(__name__)


class PhoneUsageDetection:
    """
    Detects mobile phone usage by employees and triggers alerts
    Uses best.pt model with 'phone' class detection
    """
    
    def __init__(self, channel_id, socketio=None, db_manager=None, app=None, config=None):
        """
        Initialize Phone Usage Detection module
        
        Args:
            channel_id: Unique identifier for the camera channel
            socketio: SocketIO instance for real-time alerts
            db_manager: Database manager for storing alerts
            app: Flask app instance for database context
            config: Configuration dictionary
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app
        
        # Configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.4)
        self.alert_cooldown = self.config.get('alert_cooldown', 30)  # seconds between alerts
        self.detection_duration = self.config.get('detection_duration', 2)  # seconds to confirm detection
        
        # Model configuration
        self.model_path = self.config.get('model_path', 'models/best.pt')
        self.target_class = 'phone_use'  # Class name in best.pt model
        
        # Load shared YOLO model
        from modules.model_manager import get_shared_model
        
        logger.info(f"Loading shared YOLO model for phone detection: {self.model_path}")
        try:
            self.model = get_shared_model(self.model_path, device='auto')
            logger.info("Shared phone detection model loaded successfully")
            
            # Verify 'phone' class exists in model
            if hasattr(self.model, 'names'):
                available_classes = list(self.model.names.values())
                if self.target_class not in available_classes:
                    logger.warning(f"⚠️ '{self.target_class}' class not found in model. Available: {available_classes}")
                else:
                    logger.info(f"✓ '{self.target_class}' class found in model")
                    
        except Exception as e:
            logger.error(f"Failed to load phone detection model: {e}")
            raise
        
        # State tracking
        self.detection_start_time = None
        self.last_alert_time = 0
        self.consecutive_detections = 0
        self.detection_count = 0
        self.total_alerts = 0
        
        # Frame tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Snapshot directory (kept for compatibility, but we use GIFs now)
        self.snapshot_dir = Path('static/phone_snapshots')
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GIF alert helper for violation GIFs
        self.gif_helper = GifAlertHelper(channel_id, db_manager, app, socketio)
        self.gif_helper.initialize_gif_recorder(
            buffer_size=90,  # 3 seconds at 30fps
            gif_duration=3.0,  # 3 second GIFs
            fps=30
        )
        
        # Track pending database saves for when GIF completes
        self._pending_db_save = None
        
        logger.info(f"PhoneUsageDetection initialized for channel {channel_id}")
        logger.info(f"  - Confidence threshold: {self.confidence_threshold}")
        logger.info(f"  - Alert cooldown: {self.alert_cooldown}s")
        logger.info(f"  - Detection duration: {self.detection_duration}s")
    
    def __del__(self):
        """Cleanup: release shared model reference"""
        if hasattr(self, 'model') and self.model:
            try:
                from modules.model_manager import release_shared_model
                release_shared_model(self.model_path)
                logger.info(f"Released shared model: {self.model_path}")
            except Exception as e:
                logger.error(f"Error releasing model: {e}")
    
    def process_frame(self, frame):
        """
        Process frame for phone usage detection
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            dict: {
                'frame': annotated frame,
                'status': detection status dict,
                'metadata': additional information
            }
        """
        # Add frame to GIF buffer (for pre-alert context)
        self.gif_helper.add_frame_to_buffer(frame)
        
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Extract phone detections
        phone_detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names[cls]
                    
                    # Only process 'phone' class
                    if class_name.lower() == self.target_class:
                        conf = float(boxes.conf[i].cpu().numpy())
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        
                        phone_detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class_name': class_name
                        })
        
        # Update detection state
        phone_detected = len(phone_detections) > 0
        
        if phone_detected:
            self.detection_count += 1
            self.consecutive_detections += 1
            
            # Start tracking detection duration
            if self.detection_start_time is None:
                self.detection_start_time = current_time
            
            # Check if detection has persisted long enough
            detection_duration = current_time - self.detection_start_time
            
            # Trigger alert if conditions are met
            if (detection_duration >= self.detection_duration and 
                current_time - self.last_alert_time >= self.alert_cooldown):
                self._trigger_phone_alert(frame, phone_detections, detection_duration)
                self.last_alert_time = current_time
                self.detection_start_time = None  # Reset for next detection
        else:
            # Reset detection tracking
            self.consecutive_detections = 0
            self.detection_start_time = None
        
        # Draw visualization
        annotated_frame = self._draw_visualization(frame, phone_detections, phone_detected)
        
        # Prepare status
        status = {
            'phone_detected': phone_detected,
            'detection_count': len(phone_detections),
            'total_detections': self.detection_count,
            'total_alerts': self.total_alerts,
            'consecutive_detections': self.consecutive_detections,
            'fps': round(self.fps, 2)
        }
        
        # Prepare metadata
        metadata = {
            'channel_id': self.channel_id,
            'timestamp': datetime.now().isoformat(),
            'detections': phone_detections
        }
        
        # Continue GIF recording if in progress
        if self.gif_helper.is_recording():
            self.gif_helper.add_alert_frame(frame)
            # Check if recording completed
            if self.gif_helper.is_recording_complete():
                gif_info = self.gif_helper.get_completed_gif()
                if gif_info and self._pending_db_save:
                    gif_path = self.gif_helper.get_snapshot_path_for_violation(gif_info)
                    if gif_path:
                        # Save to database with GIF path
                        pending = self._pending_db_save
                        try:
                            if self.db_manager and self.app:
                                with self.app.app_context():
                                    snapshot_id = self.db_manager.save_phone_snapshot(
                                        channel_id=self.channel_id,
                                        snapshot_filename=os.path.basename(gif_path),
                                        snapshot_path=gif_path,
                                        alert_message=pending["alert_message"],
                                        alert_data=pending["alert_data"],
                                        file_size=os.path.getsize(gif_path) if os.path.exists(gif_path) else 0,
                                        detection_count=pending["detection_count"],
                                        detection_time=pending["timestamp"]
                                    )
                                    if snapshot_id:
                                        logger.info(f"Phone GIF saved to database with ID: {snapshot_id}")
                        except Exception as e:
                            logger.error(f"Error saving phone GIF to database: {e}")
                        finally:
                            self._pending_db_save = None
        
        return {
            'frame': annotated_frame,
            'status': status,
            'metadata': metadata
        }
    
    def _trigger_phone_alert(self, frame, detections, duration):
        """
        Trigger alert for phone usage detection with GIF recording
        
        Args:
            frame: Current video frame
            detections: List of phone detections
            duration: Detection duration in seconds
        """
        self.total_alerts += 1
        timestamp = datetime.now()
        
        try:
            if frame is None or frame.size == 0:
                logger.error(f"[{self.channel_id}] ❌ Cannot start GIF recording: frame is None or empty")
                return

            logger.info(f"[{self.channel_id}] 📸 Starting GIF recording for phone usage alert")

            # Prepare alert data
            alert_message = f"Phone usage detected ({len(detections)} instance{'s' if len(detections) > 1 else ''})"
            alert_data = {
                'channel_id': self.channel_id,
                'detection_count': len(detections),
                'duration': round(duration, 2),
                'timestamp': timestamp.isoformat(),
                'detections': detections
            }
            
            # Start GIF recording
            self.gif_helper.start_alert_recording(
                alert_type='phone_usage_alert',
                alert_message=alert_message,
                frame=frame,
                alert_data=alert_data
            )

            # Store alert data for database save when GIF completes
            self._pending_db_save = {
                "alert_message": alert_message,
                "alert_data": alert_data,
                "timestamp": timestamp,
                "detection_count": len(detections)
            }

            logger.info(f"Phone usage alert triggered for {self.channel_id}")
            logger.info(f"  - Detection count: {len(detections)}")
            logger.info(f"  - Duration: {duration:.2f}s")
            logger.info(f"  - GIF recording started")
        
        except Exception as e:
            logger.error(f"Error starting GIF recording for phone alert: {e}")
        
        # Send SocketIO alert immediately
        if self.socketio:
            try:
                self.socketio.emit('phone_alert', {
                    'channel_id': self.channel_id,
                    'message': f'Phone usage detected in {self.channel_id}',
                    'snapshot': 'recording',  # Will be updated when GIF completes
                    'detection_count': len(detections),
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'alert_data': alert_data
                }, namespace='/')
                
                logger.info(f"SocketIO alert sent for phone detection")
                
            except Exception as e:
                logger.error(f"Error sending SocketIO alert: {e}")
    
    def _draw_visualization(self, frame, detections, phone_detected):
        """
        Draw detection visualization on frame
        
        Args:
            frame: Input frame
            detections: List of phone detections
            phone_detected: Boolean indicating if phone is detected
            
        Returns:
            annotated_frame: Frame with visualization
        """
        annotated_frame = frame.copy()
        
        # Draw detection boxes
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Draw red box for phone
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add label
            label = f"Phone {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 4, y1), (0, 0, 255), -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add status overlay
        status_bg_color = (0, 0, 200) if phone_detected else (0, 100, 0)
        cv2.rectangle(annotated_frame, (0, 0), (300, 30), status_bg_color, -1)
        
        status_text = f"PHONE DETECTED!" if phone_detected else "Monitoring..."
        cv2.putText(annotated_frame, status_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add detection count
        if phone_detected:
            count_text = f"Count: {len(detections)}"
            cv2.putText(annotated_frame, count_text, (10, annotated_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add timestamp
        timestamp_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated_frame, timestamp_text, (10, annotated_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def get_statistics(self):
        """
        Get current detection statistics
        
        Returns:
            dict: Statistics dictionary
        """
        return {
            'channel_id': self.channel_id,
            'total_detections': self.detection_count,
            'total_alerts': self.total_alerts,
            'consecutive_detections': self.consecutive_detections,
            'fps': round(self.fps, 2)
        }
    
    def get_status(self):
        """
        Get current module status
        
        Returns:
            dict: Status dictionary
        """
        return {
            'module': 'PhoneUsageDetection',
            'channel_id': self.channel_id,
            'active': True,
            'total_detections': self.detection_count,
            'total_alerts': self.total_alerts,
            'fps': round(self.fps, 2)
        }
