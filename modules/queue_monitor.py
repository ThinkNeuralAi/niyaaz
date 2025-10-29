"""
Queue Monitor Module
Monitors queue length and counter staffing using ROI-based detection
"""
import cv2
import numpy as np
import logging
from datetime import datetime, timedelta
import json
from .yolo_detector import YOLODetector
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)

class QueueMonitor:
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize Queue Monitor
        
        Args:
            channel_id: Unique identifier for the video channel
            socketio: SocketIO instance for real-time updates
            db_manager: Database manager for storing analytics
            app: Flask application instance for database context
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app
        
        # Initialize YOLO detector with lower confidence for better detection
        self.detector = YOLODetector(confidence_threshold=0.2)
        
        # Initialize GIF recorder for alerts
        self.gif_recorder = AlertGifRecorder(
            buffer_size=90,  # 3 seconds at 30fps
            gif_duration=4.0,  # 4 second GIFs
            fps=30
        )
        
        # ROI configuration (normalized coordinates 0-1)
        self.roi_config = {
            'main': [],      # Queue area (yellow)
            'secondary': []  # Counter area (cyan)
        }
        
        # Queue monitoring settings
        self.settings = {
            'dwell_time_threshold': 3.0,  # Minimum time in seconds to count as waiting
            'queue_alert_threshold': 2,   # Alert when queue has this many people
            'counter_threshold': 1,       # Alert when counter has <= this many people
            'alert_cooldown': 10         # Cooldown between alerts in seconds
        }
        
        # Tracking variables
        self.person_tracking = {}  # {detection_id: {'first_seen': timestamp, 'last_position': (x,y), 'in_queue': bool}}
        self.queue_count = 0
        self.counter_count = 0
        self.last_alert_time = None
        
        # Alert condition tracking
        self.alert_condition_start_time = None  # When the current alert condition first became true
        self.alert_condition_sustained_duration = 5.0  # Must sustain for 5 seconds before alerting
        
        # GIF recording tracking
        self.pending_gif_info = None  # Store completed GIF info waiting to be saved to DB
        
        # Performance metrics
        self.frame_count = 0
        
        logger.info(f"Queue Monitor initialized for channel {channel_id}")
    
    def set_roi(self, roi_points):
        """
        Set ROI configuration
        
        Args:
            roi_points: Dictionary with 'main' and 'secondary' polygon points
        """
        self.roi_config = roi_points
        logger.info(f"ROI updated for channel {self.channel_id}")
        
        # Save to database
        if self.db_manager:
            try:
                self.db_manager.save_channel_config(
                    self.channel_id, 
                    'QueueMonitor', 
                    'roi', 
                    roi_points
                )
            except Exception as e:
                logger.error(f"Failed to save ROI config: {e}")
    
    def get_roi(self):
        """
        Get current ROI configuration
        
        Returns:
            Dictionary with 'main' and 'secondary' polygon points
        """
        return self.roi_config.copy()
    
    def set_settings(self, settings):
        """Update queue monitoring settings"""
        self.settings.update(settings)
        logger.info(f"Settings updated for channel {self.channel_id}")
        
        # Save to database
        if self.db_manager:
            try:
                self.db_manager.save_channel_config(
                    self.channel_id, 
                    'QueueMonitor', 
                    'settings', 
                    self.settings
                )
            except Exception as e:
                logger.error(f"Failed to save settings: {e}")
    
    def load_configuration(self):
        """Load saved configuration from database"""
        if self.db_manager:
            try:
                # Load ROI configuration
                roi_config = self.db_manager.get_channel_config(
                    self.channel_id, 'QueueMonitor', 'roi'
                )
                if roi_config:
                    self.roi_config = roi_config
                
                # Load settings
                settings = self.db_manager.get_channel_config(
                    self.channel_id, 'QueueMonitor', 'settings'
                )
                if settings:
                    self.settings.update(settings)
                
                logger.info(f"Configuration loaded for channel {self.channel_id}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    def point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using ray casting algorithm
        
        Args:
            point: (x, y) tuple
            polygon: List of (x, y) points defining the polygon
            
        Returns:
            Boolean indicating if point is inside polygon
        """
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_roi_pixels(self, roi_name, frame_width, frame_height):
        """Convert normalized ROI coordinates to pixel coordinates"""
        if roi_name not in self.roi_config or not self.roi_config[roi_name]:
            return []
        
        pixel_points = []
        for point in self.roi_config[roi_name]:
            x = int(point['x'] * frame_width)
            y = int(point['y'] * frame_height)
            pixel_points.append((x, y))
        
        return pixel_points
    
    def classify_detections(self, detections, frame_width, frame_height):
        """
        Classify detections into queue and counter areas
        
        Args:
            detections: List of person detections
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Tuple of (queue_detections, counter_detections)
        """
        queue_detections = []
        counter_detections = []
        
        # Get ROI polygons in pixel coordinates
        queue_polygon = self.get_roi_pixels('main', frame_width, frame_height)
        counter_polygon = self.get_roi_pixels('secondary', frame_width, frame_height)
        
        for detection in detections:
            center = detection['center']
            bottom_center = detection['bottom_center']
            
            # Use bottom center for more accurate area classification
            point_to_check = bottom_center
            
            # Check if in queue area
            if queue_polygon and self.point_in_polygon(point_to_check, queue_polygon):
                detection['area_type'] = 'queue'
                queue_detections.append(detection)
            
            # Check if in counter area
            elif counter_polygon and self.point_in_polygon(point_to_check, counter_polygon):
                detection['area_type'] = 'counter'
                counter_detections.append(detection)
            
            # If not in any defined area
            else:
                detection['area_type'] = 'none'
        
        return queue_detections, counter_detections
    
    def update_person_tracking(self, detections, area_type):
        """
        Update person tracking for dwell time analysis
        
        Args:
            detections: List of detections in specific area
            area_type: 'queue' or 'counter'
            
        Returns:
            Number of people who have been in area for minimum dwell time
        """
        current_time = datetime.now()
        valid_count = 0
        
        # Simple tracking based on position proximity
        # In a production system, you might want more sophisticated tracking
        
        for detection in detections:
            center = detection['center']
            detection_id = f"{area_type}_{center[0]//20}_{center[1]//20}"  # Grid-based ID
            
            if detection_id not in self.person_tracking:
                self.person_tracking[detection_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_position': center,
                    'area_type': area_type
                }
            else:
                # Update last seen time
                self.person_tracking[detection_id]['last_seen'] = current_time
                self.person_tracking[detection_id]['last_position'] = center
            
            # Check if person has been in area long enough
            time_in_area = (current_time - self.person_tracking[detection_id]['first_seen']).total_seconds()
            if time_in_area >= self.settings['dwell_time_threshold']:
                valid_count += 1
        
        # Clean up old tracking data
        self.cleanup_tracking(current_time)
        
        return valid_count
    
    def cleanup_tracking(self, current_time):
        """Remove old tracking data"""
        timeout_threshold = timedelta(seconds=10)  # Remove if not seen for 10 seconds
        
        to_remove = []
        for track_id, track_data in self.person_tracking.items():
            if current_time - track_data['last_seen'] > timeout_threshold:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.person_tracking[track_id]
    
    def check_alert_conditions(self):
        """
        Check if alert conditions are met
        
        Alert Conditions (must be sustained for 5 seconds):
        1. Queue >= 3 people (regardless of counter staff)
        2. Queue >= 1 people AND Counter = 0 people
        
        Returns:
            Dictionary with alert information or None
        """
        current_time = datetime.now()
        
        # Check if either alert condition is currently true
        condition1 = self.queue_count >= 3
        condition2 = self.queue_count >= 1 and self.counter_count == 0
        alert_condition_met = condition1 or condition2
        
        if alert_condition_met:
            # Start tracking when this condition first became true
            if self.alert_condition_start_time is None:
                self.alert_condition_start_time = current_time
                logger.debug(f"Alert condition started: Queue={self.queue_count}, Counter={self.counter_count}")
                return None  # Don't alert yet, condition just started
            
            # Check how long the condition has been sustained
            sustained_duration = (current_time - self.alert_condition_start_time).total_seconds()
            
            if sustained_duration < self.alert_condition_sustained_duration:
                # Condition not sustained long enough yet
                logger.debug(f"Alert condition sustained for {sustained_duration:.1f}s (need {self.alert_condition_sustained_duration}s)")
                return None
            
            # Check cooldown period
            if (self.last_alert_time and 
                (current_time - self.last_alert_time).total_seconds() < self.settings['alert_cooldown']):
                return None
            
            # Condition has been sustained long enough - trigger alert!
            # Determine which condition triggered the alert
            if condition1 and condition2:
                reason = f"Queue has {self.queue_count} people and no staff at counter"
            elif condition1:
                reason = f"Queue has {self.queue_count} people waiting (counter: {self.counter_count})"
            else:  # condition2
                reason = f"Queue has {self.queue_count} people but no staff at counter"
            
            alert_info = {
                'type': 'queue_alert',
                'message': reason,
                'queue_count': self.queue_count,
                'counter_count': self.counter_count,
                'timestamp': current_time.isoformat(),
                'channel_id': self.channel_id
            }
            
            self.last_alert_time = current_time
            # Reset the condition start time after alerting
            self.alert_condition_start_time = None
            return alert_info
        else:
            # Alert condition is no longer met - reset tracking
            if self.alert_condition_start_time is not None:
                logger.debug(f"Alert condition ended before reaching threshold: Queue={self.queue_count}, Counter={self.counter_count}")
                self.alert_condition_start_time = None
            return None
    
    def process_frame(self, frame):
        """
        Process a single frame for queue monitoring
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with ROI and detections
        """
        self.frame_count += 1
        original_frame = frame.copy()
        
        # Add frame to GIF recorder buffer
        self.gif_recorder.add_frame(original_frame)
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Detect persons in frame
        detections = self.detector.detect_persons(frame)
        
        # Classify detections into areas
        queue_detections, counter_detections = self.classify_detections(
            detections, frame_width, frame_height
        )
        
        # Update tracking and get valid counts (after dwell time)
        self.queue_count = self.update_person_tracking(queue_detections, 'queue')
        self.counter_count = self.update_person_tracking(counter_detections, 'counter')
        
        # Check for alerts
        alert_info = self.check_alert_conditions()
        if alert_info:
            logger.warning(f"Queue alert triggered: {alert_info['message']}")
            
            # Start GIF recording for this alert
            self.gif_recorder.start_alert_recording(alert_info)
            
            # Add current frame to alert recording
            self.gif_recorder.add_alert_frame(original_frame)
            
            # Log to database
            if self.db_manager:
                try:
                    if self.app:
                        with self.app.app_context():
                            self.db_manager.log_queue_analytics(
                                self.channel_id,
                                self.queue_count,
                                self.counter_count,
                                alert_triggered=True,
                                alert_message=alert_info['message']
                            )
                    else:
                        self.db_manager.log_queue_analytics(
                            self.channel_id,
                            self.queue_count,
                            self.counter_count,
                            alert_triggered=True,
                            alert_message=alert_info['message']
                        )
                except Exception as e:
                    logger.error(f"Database logging error: {e}")
            
            # Send real-time alert
            self.socketio.emit('queue_alert', alert_info)
        
        # Continue recording if alert is in progress
        elif self.gif_recorder.is_recording_alert:
            self.gif_recorder.add_alert_frame(original_frame)
            
            # Check if recording auto-completed (duration reached)
            if not self.gif_recorder.is_recording_alert and self.gif_recorder.alert_end_time:
                # Recording just finished automatically
                # The GIF should have been created by stop_alert_recording, but we need to save it to DB
                self.pending_gif_info = {
                    'needs_db_save': True,
                    'alert_time': datetime.now()
                }
        
        # Check if we have pending GIF to save to database
        if self.pending_gif_info and self.pending_gif_info.get('needs_db_save'):
            self._save_completed_alert_gif()
            self.pending_gif_info = None
        
        # Log analytics data periodically
        if self.frame_count % 60 == 0:  # Every 60 frames (roughly every 2 seconds at 30fps)
            if self.db_manager:
                try:
                    if self.app:
                        with self.app.app_context():
                            self.db_manager.log_queue_analytics(
                                self.channel_id,
                                self.queue_count,
                                self.counter_count
                            )
                    else:
                        self.db_manager.log_queue_analytics(
                            self.channel_id,
                            self.queue_count,
                            self.counter_count
                        )
                except Exception as e:
                    logger.error(f"Database logging error: {e}")
        
        # Send real-time updates
        self.socketio.emit('queue_update', {
            'channel_id': self.channel_id,
            'queue_count': self.queue_count,
            'counter_count': self.counter_count
        })
        
        # Annotate frame
        annotated_frame = self.annotate_frame(original_frame, queue_detections, counter_detections)
        
        return annotated_frame
    
    def annotate_frame(self, frame, queue_detections, counter_detections):
        """
        Annotate frame with ROI, detections, and counts
        
        Args:
            frame: Input frame
            queue_detections: Detections in queue area
            counter_detections: Detections in counter area
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Draw ROI polygons
        queue_polygon = self.get_roi_pixels('main', frame_width, frame_height)
        counter_polygon = self.get_roi_pixels('secondary', frame_width, frame_height)
        
        if queue_polygon:
            # Draw queue area outline only (yellow border)
            pts = np.array(queue_polygon, np.int32)
            cv2.polylines(annotated_frame, [pts], True, (0, 255, 255), 3)
        
        if counter_polygon:
            # Draw counter area outline only (cyan border)
            pts = np.array(counter_polygon, np.int32)
            cv2.polylines(annotated_frame, [pts], True, (255, 255, 0), 3)
        
        # Draw queue detections (yellow boxes)
        for detection in queue_detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated_frame, "Queue", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw counter detections (cyan boxes)
        for detection in counter_detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated_frame, "Counter", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw counts and labels
        cv2.putText(annotated_frame, f"Queue: {self.queue_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Counter: {self.counter_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw status
        status_color = (0, 255, 0) if self.queue_count <= self.settings['queue_alert_threshold'] else (0, 0, 255)
        cv2.putText(annotated_frame, f"Status: {'Normal' if self.queue_count <= self.settings['queue_alert_threshold'] else 'Alert'}", 
                   (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return annotated_frame
    
    def _save_completed_alert_gif(self):
        """Save the most recently created alert GIF to database"""
        try:
            import os
            import glob
            
            # Find the most recent alert GIF file
            alerts_dir = "static/alerts"
            gif_files = glob.glob(os.path.join(alerts_dir, "alert_*.gif"))
            
            if not gif_files:
                logger.warning("No alert GIF files found to save")
                return
            
            # Get the most recent file
            latest_gif = max(gif_files, key=os.path.getctime)
            gif_filename = os.path.basename(latest_gif)
            
            # Get file info
            file_size = os.path.getsize(latest_gif)
            file_ctime = os.path.getctime(latest_gif)
            
            # Create GIF info
            gif_info = {
                'gif_path': latest_gif,
                'gif_filename': gif_filename,
                'frame_count': 0,  # We don't know exact frame count, but it's in the file
                'duration': 4.0    # Our configured duration
            }
            
            # Save to database
            if self.db_manager:
                try:
                    alert_data = {
                        'queue_count': self.queue_count,
                        'counter_count': self.counter_count,
                        'alert_time': datetime.fromtimestamp(file_ctime).isoformat(),
                        'channel_id': self.channel_id
                    }
                    
                    if self.app:
                        with self.app.app_context():
                            gif_id = self.db_manager.save_alert_gif(
                                channel_id=self.channel_id,
                                alert_type='queue_alert',
                                gif_info=gif_info,
                                alert_message=f"Queue has {self.queue_count} people but only {self.counter_count} at counter",
                                alert_data=alert_data
                            )
                    else:
                        gif_id = self.db_manager.save_alert_gif(
                            channel_id=self.channel_id,
                            alert_type='queue_alert',
                            gif_info=gif_info,
                            alert_message=f"Queue has {self.queue_count} people but only {self.counter_count} at counter",
                            alert_data=alert_data
                        )
                    
                    logger.info(f"Alert GIF saved to database: ID {gif_id}, file: {gif_filename}")
                    
                    # Emit GIF created event to dashboard
                    self.socketio.emit('alert_gif_created', {
                        'gif_id': gif_id,
                        'channel_id': self.channel_id,
                        'gif_filename': gif_filename,
                        'gif_url': f"/static/alerts/{gif_filename}",
                        'alert_message': f"Queue has {self.queue_count} people but only {self.counter_count} at counter",
                        'created_at': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error saving alert GIF to database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
        except Exception as e:
            logger.error(f"Error in _save_completed_alert_gif: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _check_and_save_alert_gif(self):
        """Check if alert GIF recording is complete and save it"""
        try:
            # Check if recording finished naturally (duration reached)
            if (not self.gif_recorder.is_recording_alert and 
                self.gif_recorder.alert_frames and 
                self.gif_recorder.alert_end_time):
                
                # Create GIF info for saving
                gif_info = {
                    'gif_path': '',
                    'gif_filename': '',
                    'frame_count': len(self.gif_recorder.alert_frames),
                    'duration': (self.gif_recorder.alert_end_time - self.gif_recorder.alert_start_time).total_seconds()
                }
                
                # Create GIF manually if not already created
                if not gif_info['gif_path']:
                    timestamp = self.gif_recorder.alert_start_time.strftime("%Y%m%d_%H%M%S")
                    gif_filename = f"queue_alert_{self.channel_id}_{timestamp}.gif"
                    
                    gif_result = self.gif_recorder.create_manual_gif(
                        [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in self.gif_recorder.alert_frames],
                        {'channel_id': self.channel_id, 'alert_type': 'queue_alert'},
                        gif_filename
                    )
                    
                    if gif_result:
                        gif_info.update(gif_result)
                
                # Save to database
                if self.db_manager and gif_info.get('gif_path'):
                    try:
                        alert_data = {
                            'queue_count': self.queue_count,
                            'counter_count': self.counter_count,
                            'alert_time': self.gif_recorder.alert_start_time.isoformat(),
                            'channel_id': self.channel_id
                        }
                        
                        if self.app:
                            with self.app.app_context():
                                gif_id = self.db_manager.save_alert_gif(
                                    channel_id=self.channel_id,
                                    alert_type='queue_alert',
                                    gif_info=gif_info,
                                    alert_message=f"Queue has {self.queue_count} people but only {self.counter_count} at counter",
                                    alert_data=alert_data
                                )
                        else:
                            gif_id = self.db_manager.save_alert_gif(
                                channel_id=self.channel_id,
                                alert_type='queue_alert',
                                gif_info=gif_info,
                                alert_message=f"Queue has {self.queue_count} people but only {self.counter_count} at counter",
                                alert_data=alert_data
                            )
                        
                        logger.info(f"Alert GIF saved to database: ID {gif_id}, file: {gif_info['gif_filename']}")
                        
                        # Emit GIF created event to dashboard
                        self.socketio.emit('alert_gif_created', {
                            'gif_id': gif_id,
                            'channel_id': self.channel_id,
                            'gif_filename': gif_info['gif_filename'],
                            'gif_url': f"/static/alerts/{gif_info['gif_filename']}",
                            'alert_message': f"Queue has {self.queue_count} people but only {self.counter_count} at counter",
                            'created_at': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error saving alert GIF to database: {e}")
                
                # Clear alert frames after processing
                self.gif_recorder.alert_frames = []
                self.gif_recorder.alert_start_time = None
                self.gif_recorder.alert_end_time = None
                
        except Exception as e:
            logger.error(f"Error checking and saving alert GIF: {e}")
    
    def get_current_status(self):
        """Get current queue status"""
        return {
            'queue_count': self.queue_count,
            'counter_count': self.counter_count,
            'alert_threshold': self.settings['queue_alert_threshold'],
            'last_alert': self.last_alert_time.isoformat() if self.last_alert_time else None,
            'roi_configured': bool(self.roi_config['main'] or self.roi_config['secondary'])
        }
    
    def get_status(self):
        """Get module status information"""
        return {
            'module': 'QueueMonitor',
            'channel_id': self.channel_id,
            'status': 'active',
            'queue_count': self.queue_count,
            'counter_count': self.counter_count,
            'settings': self.settings,
            'roi_config': self.roi_config,
            'frame_count': self.frame_count,
            'tracked_persons': len(self.person_tracking)
        }