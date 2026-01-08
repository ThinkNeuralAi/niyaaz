"""
Fall Detection Module for Sakshi.AI
- Detects person falls using YOLO and advanced tracking with IoU matching
- Uses history-based fall detection with multiple criteria:
  - Fast downward movement
  - Aspect ratio change (horizontal orientation)
  - Height collapse
- Takes snapshot and saves to database
- Emits real-time Socket.IO events
"""

import time
import cv2
import numpy as np
import logging
import os
import torch
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from pathlib import Path
from .model_manager import get_shared_model, release_shared_model

logger = logging.getLogger(__name__)


# -------------------------
# Utility functions
# -------------------------
def iou(bb1, bb2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1, y1, x2, y2 = bb1
    X1, Y1, X2, Y2 = bb2
    xi1 = max(x1, X1)
    yi1 = max(y1, Y1)
    xi2 = min(x2, X2)
    yi2 = min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (X2 - X1) * (Y2 - Y1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def bbox_center(b):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_wh(b):
    """Get width and height of bounding box"""
    x1, y1, x2, y2 = b
    return (x2 - x1), (y2 - y1)


# -------------------------
# Tracker
# -------------------------
class Track:
    """Track a person across frames"""
    
    def __init__(self, tid, bbox, frame_idx):
        self.id = tid
        self.bbox = bbox
        self.history = deque(maxlen=30)  # Store last 30 frames (increased for better baseline)
        self.history.append((frame_idx, bbox))
        self.last_seen = frame_idx
        self.last_fall_time = 0
        self.misses = 0
        self.state = "normal"
        self.fall_state_frames = 0  # Consecutive frames in "fallen" state
        self.velocity_history = deque(maxlen=5)  # Store recent velocities for smoothing

    def update(self, bbox, frame_idx):
        """Update track with new detection"""
        self.bbox = bbox
        self.last_seen = frame_idx
        self.misses = 0
        self.history.append((frame_idx, bbox))

    def mark_missed(self):
        """Mark track as missed in current frame"""
        self.misses += 1

    def is_lost(self, max_misses=15):
        """Check if track should be removed"""
        return self.misses > max_misses


class IoUTracker:
    """IoU-based tracker for person tracking"""
    
    def __init__(self, iou_threshold=0.3, max_misses=15):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses

    def step(self, detections, frame_index):
        """
        Update tracker with new detections
        
        Args:
            detections: List of bounding boxes [(x1, y1, x2, y2), ...]
            frame_index: Current frame index
            
        Returns:
            dict: {detection_index: track_id}
        """
        assignments = {}

        # If no tracks exist, create new tracks for all detections
        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                tid = self.next_id
                self.tracks[tid] = Track(tid, det, frame_index)
                assignments[i] = tid
                self.next_id += 1
            return assignments

        # If no detections, mark all tracks as missed
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid].mark_missed()
                if self.tracks[tid].is_lost(self.max_misses):
                    del self.tracks[tid]
            return {}

        # Build IoU matrix
        track_ids = list(self.tracks.keys())
        T = len(track_ids)
        D = len(detections)
        iou_matrix = np.zeros((T, D))

        for ti, tid in enumerate(track_ids):
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = iou(self.tracks[tid].bbox, det)

        # Greedy assignment
        used_tracks = set()
        used_dets = set()

        while True:
            if np.all(iou_matrix < 0):
                break
            ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[ti, di] < self.iou_threshold:
                break

            if ti not in used_tracks and di not in used_dets:
                tid = track_ids[ti]
                assignments[di] = tid
                used_tracks.add(ti)
                used_dets.add(di)

            iou_matrix[ti, :] = -1
            iou_matrix[:, di] = -1

        # Create new tracks for unassigned detections
        for di, det in enumerate(detections):
            if di not in assignments:
                tid = self.next_id
                self.tracks[tid] = Track(tid, det, frame_index)
                assignments[di] = tid
                self.next_id += 1

        # Mark missed tracks
        for tid in list(self.tracks.keys()):
            if tid not in assignments.values():
                self.tracks[tid].mark_missed()
                if self.tracks[tid].is_lost(self.max_misses):
                    del self.tracks[tid]

        # Update assigned tracks
        for di, tid in assignments.items():
            self.tracks[tid].update(detections[di], frame_index)

        return assignments


class FallDetection:
    """Person fall detection with advanced tracking and history-based detection"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize fall detection module
        
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
        self.model_weight = "models/yolo11n.pt"  # Updated to yolo11n
        self.conf_threshold = 0.5  # Confidence threshold for person detection
        self.nms_iou = 0.45
        
        # Fall detection hyperparameters (stricter to reduce false positives)
        self.down_speed_threshold = 80  # Pixels downward movement (increased from 40)
        self.aspect_ratio_threshold = 1.2  # Width/height ratio threshold (increased - person must be more horizontal)
        self.height_drop_ratio = 0.5  # Height must drop to 50% of baseline (stricter)
        self.fall_time_window = 15  # Frames to consider for fall detection
        self.iou_match_threshold = 0.3  # IoU threshold for tracking
        self.max_misses = 15  # Max frames to keep lost track
        self.cooldown_secs = 10  # Seconds between alerts for same track (increased)
        
        # Additional strictness parameters
        self.min_fall_frames = 5  # Person must be in "fallen" state for N consecutive frames
        self.max_horizontal_speed = 30  # Max horizontal movement during fall (falls have minimal horizontal movement)
        self.baseline_frames = 10  # Use last N frames for baseline (not first frame)
        self.velocity_smoothing = 3  # Frames to average velocity over
        
        # Initialize YOLO detector with shared model manager
        logger.info(f"Loading shared YOLO model for fall detection: {self.model_weight}")
        try:
            self.model = get_shared_model(self.model_weight, device='auto')
            logger.info("Shared fall detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shared fall detection model: {e}")
            raise
        
        # Initialize tracker
        self.tracker = IoUTracker(
            iou_threshold=self.iou_match_threshold,
            max_misses=self.max_misses
        )
        
        # Snapshot directory
        self.snapshot_dir = Path("static/fall_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # State storage
        self.frame_index = 0
        self.frame_count = 0
        self.fall_detection_count = 0
        self.total_alerts = 0
        self.last_update_time = time.time()
        
        logger.info(f"FallDetection initialized for channel {channel_id}")
    
    def __del__(self):
        """Cleanup: Release shared model reference when fall detection is destroyed"""
        try:
            if hasattr(self, 'model_weight'):
                release_shared_model(self.model_weight, device='auto')
                logger.debug(f"Released shared model reference: {self.model_weight}")
        except Exception as e:
            logger.warning(f"Error releasing shared model: {e}")
    
    def process_frame(self, frame):
        """
        Process a single frame for fall detection
        
        Args:
            frame: Input video frame
            
        Returns:
            dict: {
                'frame': annotated_frame,
                'status': {...},
                'metadata': {...}
            }
        """
        if frame is None:
            return None
        
        self.frame_count += 1
        self.frame_index += 1
        t_now = time.time()
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and self.frame_count % 100 == 0:
            torch.cuda.empty_cache()
        
        # YOLO inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False)
        
        # Extract person detections
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if float(box.conf) < self.conf_threshold:
                    continue
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, str(cls_id))
                
                # Only process person detections
                if cls_name.lower() == 'person':
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    detections.append((int(x1), int(y1), int(x2), int(y2)))
        
        # Update tracker
        assignments = self.tracker.step(detections, self.frame_index)
        
        # Process tracks for fall detection
        fallen_tracks = []
        annotated_frame = frame.copy()
        
        for di, tid in assignments.items():
            tr = self.tracker.tracks[tid]
            cur = tr.bbox
            cx, cy = bbox_center(cur)
            w, h = bbox_wh(cur)
            ar = w / h if h > 0 else 0
            
            # Need at least 2 frames in history for comparison
            if len(tr.history) < 2:
                # Draw normal person box
                cv2.rectangle(annotated_frame, (cur[0], cur[1]), (cur[2], cur[3]), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID:{tid}", (cur[0], cur[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                continue
            
            # Get baseline from recent frames (not first frame) to avoid false positives from walking
            # Use average of last N frames as baseline
            if len(tr.history) < self.baseline_frames:
                # Not enough history yet, use first frame
                base_idx, base_bb = tr.history[0]
                base_cx, base_cy = bbox_center(base_bb)
                base_w, base_h = bbox_wh(base_bb)
            else:
                # Use average of recent baseline frames (skip very recent frames to avoid comparing to current state)
                recent_frames = list(tr.history)[-self.baseline_frames:-5]  # Skip very recent frames
                if len(recent_frames) > 0:
                    base_cxs = [bbox_center(bb)[0] for _, bb in recent_frames]
                    base_cys = [bbox_center(bb)[1] for _, bb in recent_frames]
                    base_ws = [bbox_wh(bb)[0] for _, bb in recent_frames]
                    base_hs = [bbox_wh(bb)[1] for _, bb in recent_frames]
                    base_cx = np.mean(base_cxs)
                    base_cy = np.mean(base_cys)
                    base_w = np.mean(base_ws)
                    base_h = np.mean(base_hs)
                    base_idx = recent_frames[0][0]
                else:
                    # Fallback to first frame
                    base_idx, base_bb = tr.history[0]
                    base_cx, base_cy = bbox_center(base_bb)
                    base_w, base_h = bbox_wh(base_bb)
            
            # Calculate velocities (smoothed)
            if len(tr.history) >= 2:
                prev_idx, prev_bb = tr.history[-2]
                prev_cx, prev_cy = bbox_center(prev_bb)
                vx = cx - prev_cx  # Horizontal velocity
                vy = cy - prev_cy  # Vertical velocity (positive = downward)
                tr.velocity_history.append((vx, vy))
            else:
                vx, vy = 0, 0
                tr.velocity_history.append((vx, vy))
            
            # Smooth velocity over recent frames
            if len(tr.velocity_history) >= self.velocity_smoothing:
                recent_vx = [v[0] for v in list(tr.velocity_history)[-self.velocity_smoothing:]]
                recent_vy = [v[1] for v in list(tr.velocity_history)[-self.velocity_smoothing:]]
                avg_vx = np.mean(recent_vx)
                avg_vy = np.mean(recent_vy)
            else:
                avg_vx = vx
                avg_vy = vy
            
            # Calculate fall indicators (stricter criteria)
            fast_drop = avg_vy > self.down_speed_threshold  # Sustained downward movement
            ar_flip = ar > self.aspect_ratio_threshold  # Horizontal orientation (lying down)
            height_collapse = h < base_h * self.height_drop_ratio  # Significant height reduction
            low_horizontal_movement = abs(avg_vx) < self.max_horizontal_speed  # Minimal horizontal movement
            within_time = (self.frame_index - base_idx) <= self.fall_time_window
            
            # Check cooldown
            in_cd = (time.time() - tr.last_fall_time) < self.cooldown_secs
            
            # Fall detection logic (much stricter)
            potential_fall = False
            if not in_cd:
                # Require ALL of these conditions for a potential fall:
                # 1. Fast sustained downward movement
                # 2. Horizontal orientation (lying down)
                # 3. Significant height collapse
                # 4. Minimal horizontal movement (not walking)
                # 5. Within time window
                if (fast_drop and ar_flip and height_collapse and 
                    low_horizontal_movement and within_time):
                    potential_fall = True
            
            # Require person to be in "fallen" state for multiple consecutive frames
            if potential_fall:
                tr.fall_state_frames += 1
            else:
                tr.fall_state_frames = 0
            
            # Only trigger alert if person has been in fallen state for minimum frames
            is_fall = tr.fall_state_frames >= self.min_fall_frames
            
            if is_fall:
                tr.last_fall_time = time.time()
                fallen_tracks.append((tid, cur))
                
                # Trigger alert
                self._trigger_fall_alert(annotated_frame, tid, cur, t_now)
            
            # Draw bounding box
            color = (0, 0, 255) if is_fall else (0, 255, 0)
            thickness = 3 if is_fall else 2
            cv2.rectangle(annotated_frame, (cur[0], cur[1]), (cur[2], cur[3]), color, thickness)
            
            label = f"FALL ID:{tid}" if is_fall else f"ID:{tid}"
            cv2.putText(annotated_frame, label, (cur[0], cur[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        self.fall_detection_count = len(fallen_tracks)
        
        # Draw alert banner if falls detected
        if fallen_tracks:
            banner_height = 60
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], banner_height), (0, 0, 255), -1)
            text = f"FALL ALERT! {len(fallen_tracks)} person(s) fallen"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = (annotated_frame.shape[1] - text_size[0]) // 2
            cv2.putText(annotated_frame, text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # Draw statistics at bottom
        stats_y = annotated_frame.shape[0] - 60
        stats_bg = (50, 50, 50)
        cv2.rectangle(annotated_frame, (0, stats_y - 10), (450, annotated_frame.shape[0]), stats_bg, -1)
        cv2.putText(annotated_frame, f"Persons Tracked: {len(self.tracker.tracks)}", (10, stats_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Falls Detected: {len(fallen_tracks)}", (10, stats_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if fallen_tracks else (0, 255, 0), 2)
        
        # Send real-time updates
        if t_now - self.last_update_time >= 1.0:
            self._send_realtime_update()
            self.last_update_time = t_now
        
        # Return structured result format
        return {
            'frame': annotated_frame,
            'status': {
                'persons_tracked': len(self.tracker.tracks),
                'falls_detected': len(fallen_tracks),
                'total_alerts': self.total_alerts,
                'active_falls': len(fallen_tracks)
            },
            'metadata': {
                'frame_count': self.frame_count,
                'timestamp': t_now,
                'channel_id': self.channel_id
            }
        }
    
    def _trigger_fall_alert(self, frame, track_id, bbox, timestamp):
        """Trigger fall detection alert with snapshot"""
        self.total_alerts += 1
        
        # Generate filename
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        folder = self.snapshot_dir / f"fall_{track_id}_{timestamp_str}"
        folder.mkdir(parents=True, exist_ok=True)
        
        filename = f"trigger_frame.jpg"
        filepath = folder / filename
        
        # Save snapshot
        try:
            # Draw bounding box on snapshot
            snapshot = frame.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(snapshot, f"FALL ID:{track_id}",
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 2)
            
            cv2.imwrite(str(filepath), snapshot)
            file_size = os.path.getsize(filepath)
            
            logger.warning(f"[ALERT] FALL: Track {track_id} saved → {folder}")
            
            # Prepare alert data
            alert_message = f"Person fall detected - Track ID: {track_id}"
            alert_data = {
                'track_id': track_id,
                'bbox': list(bbox),
                'channel_id': self.channel_id
            }
            
            # Save to database
            if self.db_manager and self.app:
                try:
                    with self.app.app_context():
                        # Use relative path for database
                        relative_path = str(filepath.relative_to("static"))
                        snapshot_id = self.db_manager.save_fall_snapshot(
                            channel_id=self.channel_id,
                            snapshot_filename=filename,
                            snapshot_path=relative_path,
                            alert_message=alert_message,
                            alert_data=alert_data,
                            file_size=file_size,
                            fall_duration=0.0  # Not applicable with new logic
                        )
                        
                        logger.info(f"Fall snapshot saved to database: ID {snapshot_id}")
                        
                        # Emit real-time notification
                        self.socketio.emit('fall_detected', {
                            'snapshot_id': snapshot_id,
                            'channel_id': self.channel_id,
                            'track_id': track_id,
                            'snapshot_filename': filename,
                            'snapshot_url': f"/static/{relative_path}",
                            'alert_message': alert_message,
                            'timestamp': dt.isoformat()
                        })
                        
                except Exception as e:
                    logger.error(f"Error saving fall snapshot to database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Emit Socket.IO alert
            self.socketio.emit('fall_alert', {
                'channel_id': self.channel_id,
                'track_id': track_id,
                'alert_message': alert_message,
                'snapshot_filename': filename,
                'snapshot_url': f"/static/{filepath.relative_to('static')}",
                'timestamp': dt.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error saving fall snapshot: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _send_realtime_update(self):
        """Send real-time statistics update via Socket.IO"""
        if self.socketio:
            self.socketio.emit('fall_detection_update', {
                'channel_id': self.channel_id,
                'current_falls': self.fall_detection_count,
                'total_alerts': self.total_alerts,
                'frame_count': self.frame_count,
                'persons_tracked': len(self.tracker.tracks),
                'timestamp': datetime.now().isoformat()
            })
    
    def get_statistics(self):
        """Get current detection statistics"""
        return {
            'current_falls': self.fall_detection_count,
            'total_alerts': self.total_alerts,
            'frame_count': self.frame_count,
            'persons_tracked': len(self.tracker.tracks)
        }
    
    def get_current_status(self):
        """Get current module status"""
        return {
            'active': True,
            'current_falls': self.fall_detection_count,
            'total_alerts': self.total_alerts,
            'persons_tracked': len(self.tracker.tracks)
        }
    
    def update_config(self, config):
        """Update detection configuration"""
        if 'confidence_threshold' in config:
            self.conf_threshold = float(config['confidence_threshold'])
        if 'down_speed_threshold' in config:
            self.down_speed_threshold = float(config['down_speed_threshold'])
        if 'aspect_ratio_threshold' in config:
            self.aspect_ratio_threshold = float(config['aspect_ratio_threshold'])
        if 'height_drop_ratio' in config:
            self.height_drop_ratio = float(config['height_drop_ratio'])
        if 'cooldown_secs' in config:
            self.cooldown_secs = float(config['cooldown_secs'])
        if 'min_fall_frames' in config:
            self.min_fall_frames = int(config['min_fall_frames'])
        if 'max_horizontal_speed' in config:
            self.max_horizontal_speed = float(config['max_horizontal_speed'])
        if 'baseline_frames' in config:
            self.baseline_frames = int(config['baseline_frames'])
        if 'velocity_smoothing' in config:
            self.velocity_smoothing = int(config['velocity_smoothing'])
        
        logger.info(f"FallDetection config updated for channel {self.channel_id}")
