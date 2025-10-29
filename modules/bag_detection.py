"""
Unattended Baggage Detection Module for Sakshi.AI
- Detects persons and bags using YOLO
- Tracks objects across frames with DeepSORT
- Flags bags that are stationary and no person is nearby
- Triggers alerts with automatic GIF recording
"""

import time
import cv2
import numpy as np
import logging
import math
from collections import deque
from datetime import datetime
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logging.warning("DeepSort not available. Install with: pip install deep-sort-realtime")

from modules.gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)


class BagDetection:
    """Unattended baggage detection with tracking and alerting"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize bag detection module
        
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
        
        # Detection configuration
        self.model_weight = "models/yolo11n.pt"
        self.conf_threshold = 0.35
        self.nms_iou = 0.45
        
        # Bag detection classes
        self.bag_classes = {"backpack", "handbag", "suitcase", "bag"}
        self.person_class = "person"
        
        # Alert thresholds
        self.time_threshold = 20.0  # seconds to consider unattended
        self.proximity_threshold_px = 120  # pixels - person near bag
        self.stationary_disp_threshold = 10  # pixels of movement allowed
        self.max_history = 120  # frames of history per tracked object
        self.alert_cooldown = 60.0  # seconds between repeated alerts
        
        # Initialize YOLO detector
        logger.info(f"Loading YOLO model for bag detection: {self.model_weight}")
        self.yolo = YOLO(self.model_weight)
        
        # Initialize DeepSORT tracker
        if DEEPSORT_AVAILABLE:
            logger.info("Initializing DeepSORT tracker for bag detection")
            self.tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)
            self.tracking_enabled = True
        else:
            logger.warning("DeepSORT not available - tracking disabled")
            self.tracker = None
            self.tracking_enabled = False
        
        # State storage
        self.bag_history = {}  # track_id -> deque of (timestamp, centroid)
        self.bag_first_seen = {}  # track_id -> timestamp when first seen
        self.bag_last_alerted = {}  # track_id -> last alert time
        self.bag_is_flagged = {}  # track_id -> bool
        self.pending_gif_info = {}  # track_id -> gif_info dict
        
        # Statistics
        self.total_bags_detected = 0
        self.total_alerts_triggered = 0
        self.active_alerts = 0
        
        # GIF recorder
        self.gif_recorder = AlertGifRecorder(fps=30)
        
        # Frame processing
        self.frame_count = 0
        self.last_update_time = time.time()
        
        logger.info(f"BagDetection initialized for channel {channel_id}")
    
    @staticmethod
    def centroid_from_bbox(bbox):
        """Calculate centroid from bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    @staticmethod
    def euclidean(a, b):
        """Calculate Euclidean distance between two points"""
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def is_stationary(self, history_deque):
        """Check if a bag is stationary based on its movement history"""
        if len(history_deque) < 5:
            return False
        
        # Compute max displacement between any two points in history
        pts = [p for (_, p) in history_deque]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        
        displacement = math.hypot(dx, dy)
        return displacement <= self.stationary_disp_threshold
    
    def process_frame(self, frame):
        """
        Process a single frame for bag detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with detection boxes and alerts
        """
        if frame is None:
            return None
        
        self.frame_count += 1
        t_now = time.time()
        
        # Add frame to GIF recorder circular buffer
        self.gif_recorder.add_frame(frame)
        
        # YOLO inference
        results = self.yolo(frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False)
        
        # Convert YOLO results to tracker format
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo.names.get(cls_id, str(cls_id))
                
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                w = x2 - x1
                h = y2 - y1
                
                detections.append({
                    "bbox": [x1, y1, w, h],
                    "confidence": conf,
                    "class_name": cls_name
                })
        
        # Track objects with DeepSORT
        people_tracks = {}
        bag_tracks = {}
        
        if self.tracking_enabled and self.tracker:
            # Prepare DeepSORT input
            ds_inputs = []
            for d in detections:
                x, y, w, h = d["bbox"]
                ds_inputs.append(([x, y, w, h], d["confidence"], d["class_name"]))
            
            # Update tracker
            tracks = self.tracker.update_tracks(ds_inputs, frame=frame)
            
            # Categorize tracks
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                
                tid = tr.track_id
                ltr = tr.to_tlbr()  # left, top, right, bottom
                bbox = [int(x) for x in ltr]
                centroid = self.centroid_from_bbox(bbox)
                
                # Get class from detection
                cls = tr.get_det_class()
                if cls is None:
                    cls = getattr(tr, 'det_class', None)
                
                label = str(cls).lower() if cls is not None else ""
                
                if self.person_class in label:
                    people_tracks[tid] = {"bbox": bbox, "centroid": centroid, "track": tr}
                elif any(b in label for b in self.bag_classes):
                    bag_tracks[tid] = {"bbox": bbox, "centroid": centroid, "track": tr}
        else:
            # Fallback: use detections without tracking
            for i, d in enumerate(detections):
                x, y, w, h = d["bbox"]
                bbox = [x, y, x + w, y + h]
                centroid = self.centroid_from_bbox(bbox)
                label = d["class_name"].lower()
                
                if self.person_class in label:
                    people_tracks[i] = {"bbox": bbox, "centroid": centroid}
                elif any(b in label for b in self.bag_classes):
                    bag_tracks[i] = {"bbox": bbox, "centroid": centroid}
        
        # Update bag histories
        for tid, info in bag_tracks.items():
            c = info["centroid"]
            if tid not in self.bag_history:
                self.bag_history[tid] = deque(maxlen=self.max_history)
                self.bag_first_seen[tid] = t_now
                self.bag_is_flagged[tid] = False
                self.bag_last_alerted[tid] = 0.0
                self.total_bags_detected += 1
            
            self.bag_history[tid].append((t_now, c))
        
        # Cleanup lost bags
        current_bag_ids = set(bag_tracks.keys())
        for tid in list(self.bag_history.keys()):
            if tid not in current_bag_ids:
                last_ts, _ = self.bag_history[tid][-1]
                if t_now - last_ts > 5.0:
                    # Bag lost for 5 seconds - remove
                    if self.bag_is_flagged.get(tid, False):
                        self.active_alerts = max(0, self.active_alerts - 1)
                    
                    del self.bag_history[tid]
                    del self.bag_first_seen[tid]
                    del self.bag_is_flagged[tid]
                    del self.bag_last_alerted[tid]
        
        # Check each bag for alert conditions
        alerts = []
        for tid, hist in self.bag_history.items():
            last_ts, last_centroid = hist[-1]
            
            # Find nearest person
            nearest_person = None
            nearest_dist = float('inf')
            for pid, pinfo in people_tracks.items():
                dist = self.euclidean(last_centroid, pinfo["centroid"])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_person = pid
            
            person_nearby = (nearest_dist <= self.proximity_threshold_px)
            stationary = self.is_stationary(hist)
            time_seen = t_now - self.bag_first_seen.get(tid, t_now)
            
            # Alert condition: no person nearby, stationary, time threshold exceeded
            if (not person_nearby) and stationary and (time_seen >= self.time_threshold):
                # Check cooldown
                if t_now - self.bag_last_alerted.get(tid, 0.0) > self.alert_cooldown:
                    self.bag_last_alerted[tid] = t_now
                    
                    if not self.bag_is_flagged.get(tid, False):
                        self.bag_is_flagged[tid] = True
                        self.active_alerts += 1
                        self.total_alerts_triggered += 1
                        
                        # Start GIF recording
                        self._trigger_alert(tid, last_centroid, time_seen, nearest_dist)
                    
                    alerts.append((tid, last_centroid, time_seen, nearest_dist))
            else:
                # Clear flag if person returns
                if person_nearby and self.bag_is_flagged.get(tid, False):
                    self.bag_is_flagged[tid] = False
                    self.active_alerts = max(0, self.active_alerts - 1)
        
        # Draw visualization
        vis_frame = self._draw_visualization(frame, people_tracks, bag_tracks, alerts)
        
        # Send real-time updates
        if t_now - self.last_update_time >= 1.0:
            self._send_realtime_update()
            self.last_update_time = t_now
        
        return vis_frame
    
    def _trigger_alert(self, bag_id, centroid, duration, distance_to_person):
        """Trigger an unattended bag alert with GIF recording"""
        logger.warning(f"ALERT: Unattended bag {bag_id} detected for {duration:.1f}s, "
                      f"nearest person {distance_to_person:.0f}px away")
        
        # Start GIF recording
        gif_info = self.gif_recorder.start_alert_recording()
        if gif_info:
            self.pending_gif_info[bag_id] = gif_info
            logger.info(f"Started GIF recording for bag {bag_id}")
        
        # Emit real-time alert
        self.socketio.emit('bag_alert', {
            'channel_id': self.channel_id,
            'bag_id': str(bag_id),
            'duration': round(duration, 1),
            'distance_to_person': round(distance_to_person, 0),
            'centroid': {'x': int(centroid[0]), 'y': int(centroid[1])},
            'timestamp': datetime.now().isoformat()
        })
    
    def _save_completed_alert_gif(self, bag_id, gif_info):
        """Save completed alert GIF to database"""
        if not self.db_manager or not self.app:
            return
        
        try:
            with self.app.app_context():
                # Find the most recent GIF file
                import glob
                import os
                gif_files = sorted(glob.glob('static/alerts/alert_*.gif'), 
                                 key=os.path.getmtime, reverse=True)
                
                if gif_files:
                    gif_filename = os.path.basename(gif_files[0])
                    gif_path = gif_files[0]
                    file_size = os.path.getsize(gif_path)
                    
                    alert_message = f"Unattended bag detected (ID: {bag_id})"
                    alert_data = {
                        'bag_id': str(bag_id),
                        'frame_count': gif_info.get('frame_count', 0),
                        'duration_seconds': gif_info.get('duration', 0),
                        'channel_id': self.channel_id
                    }
                    
                    gif_id = self.db_manager.save_alert_gif(
                        channel_id=self.channel_id,
                        alert_type='unattended_bag',
                        gif_filename=gif_filename,
                        gif_path=gif_path,
                        alert_message=alert_message,
                        alert_data=alert_data,
                        frame_count=gif_info.get('frame_count', 0),
                        file_size=file_size,
                        duration_seconds=gif_info.get('duration', 0)
                    )
                    
                    logger.info(f"Alert GIF saved to database: ID {gif_id}, file: {gif_filename}")
                    
                    # Emit GIF created event
                    self.socketio.emit('alert_gif_created', {
                        'gif_id': gif_id,
                        'channel_id': self.channel_id,
                        'alert_type': 'unattended_bag',
                        'gif_filename': gif_filename,
                        'gif_url': f"/static/alerts/{gif_filename}",
                        'alert_message': alert_message,
                        'created_at': datetime.now().isoformat()
                    })
                    
        except Exception as e:
            logger.error(f"Error saving alert GIF to database: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _draw_visualization(self, frame, people_tracks, bag_tracks, alerts):
        """Draw bounding boxes, tracks, and alerts on frame"""
        vis = frame.copy()
        
        # Draw people in green
        for pid, pinfo in people_tracks.items():
            x1, y1, x2, y2 = pinfo["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"Person{pid}", (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw bags (blue or red if flagged)
        for bid, binfo in bag_tracks.items():
            x1, y1, x2, y2 = binfo["bbox"]
            color = (0, 0, 255) if self.bag_is_flagged.get(bid, False) else (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"Bag{bid}", (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw history path
            if bid in self.bag_history:
                pts = [tuple(map(int, p)) for (_, p) in self.bag_history[bid]]
                for i in range(1, len(pts)):
                    cv2.line(vis, pts[i-1], pts[i], color, 2)
        
        # Draw alert text
        y0 = 30
        for (tid, centroid, duration, distance) in alerts:
            text = f"ALERT: Bag {tid} unattended for {int(duration)}s"
            cv2.putText(vis, text, (10, y0), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y0 += 30
        
        # Draw statistics
        stats_y = vis.shape[0] - 60
        cv2.putText(vis, f"Bags Tracked: {len(bag_tracks)}", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Active Alerts: {self.active_alerts}", (10, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis
    
    def _send_realtime_update(self):
        """Send real-time statistics update via Socket.IO"""
        self.socketio.emit('bag_detection_update', {
            'channel_id': self.channel_id,
            'bags_tracked': len(self.bag_history),
            'active_alerts': self.active_alerts,
            'total_bags': self.total_bags_detected,
            'total_alerts': self.total_alerts_triggered,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_statistics(self):
        """Get current detection statistics"""
        return {
            'bags_tracked': len(self.bag_history),
            'active_alerts': self.active_alerts,
            'total_bags_detected': self.total_bags_detected,
            'total_alerts_triggered': self.total_alerts_triggered,
            'frame_count': self.frame_count
        }
    
    def update_config(self, config):
        """Update detection configuration"""
        if 'time_threshold' in config:
            self.time_threshold = float(config['time_threshold'])
        if 'proximity_threshold' in config:
            self.proximity_threshold_px = float(config['proximity_threshold'])
        if 'stationary_threshold' in config:
            self.stationary_disp_threshold = float(config['stationary_threshold'])
        if 'alert_cooldown' in config:
            self.alert_cooldown = float(config['alert_cooldown'])
        
        logger.info(f"BagDetection config updated for channel {self.channel_id}")
