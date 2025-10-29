"""
People Counter Module
Implements line-crossing detection for counting people entering and exiting
"""
import cv2
import numpy as np
import logging
from datetime import datetime
import json
from .yolo_detector import YOLODetector, PersonTracker

logger = logging.getLogger(__name__)

class PeopleCounter:
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize People Counter
        
        Args:
            channel_id: Unique identifier for the video channel
            socketio: SocketIO instance for real-time updates
            db_manager: Database manager for storing counts
            app: Flask application instance for database contextKeeping only last 10 seconds of data     
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app
        
        # Initialize YOLO detector
        self.detector = YOLODetector(confidence_threshold=0.6)
        
        # Initialize person tracker
        self.tracker = PersonTracker(max_disappeared=30, max_distance=80)
        
        # Counting variables
        self.in_count = 0
        self.out_count = 0
        
        # Counting line configuration (default: vertical line at center)
        self.counting_line = {
            'start': {'x': 0.5, 'y': 0.0},  # Normalized coordinates (0-1)
            'end': {'x': 0.5, 'y': 1.0},
            'orientation': 'vertical'
        }
        
        # Tracking for line crossing detection
        self.tracked_positions = {}  # {person_id: [previous_positions]}
        self.crossed_persons = set()  # Prevent double counting
        
        # Performance metrics
        self.frame_count = 0
        self.last_reset_time = datetime.now()
        
        logger.info(f"People Counter initialized for channel {channel_id}")
    
    def set_counting_line(self, line_config):
        """
        Set the counting line configuration
        
        Args:
            line_config: Dictionary with line configuration
        """
        self.counting_line = line_config
        logger.info(f"Counting line updated for channel {self.channel_id}")
    
    def get_counting_line(self):
        """
        Get the current counting line configuration
        
        Returns:
            Dictionary with line configuration
        """
        return self.counting_line.copy()
    
    def get_counting_line_pixels(self, frame_width, frame_height):
        """Convert normalized line coordinates to pixel coordinates"""
        start_x = int(self.counting_line['start']['x'] * frame_width)
        start_y = int(self.counting_line['start']['y'] * frame_height)
        end_x = int(self.counting_line['end']['x'] * frame_width)
        end_y = int(self.counting_line['end']['y'] * frame_height)
        
        return (start_x, start_y), (end_x, end_y)
    
    def check_line_crossing(self, person_id, current_center, frame_width, frame_height):
        """
        Check if a person has crossed the counting line
        
        Args:
            person_id: Unique person identifier
            current_center: Current center position (x, y)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Direction of crossing ('in', 'out', or None)
        """
        if person_id in self.crossed_persons:
            return None
        
        # Get line coordinates in pixels
        line_start, line_end = self.get_counting_line_pixels(frame_width, frame_height)
        
        # Initialize tracking for new person
        if person_id not in self.tracked_positions:
            self.tracked_positions[person_id] = []
        
        # Add current position to tracking
        self.tracked_positions[person_id].append(current_center)
        
        # Keep only recent positions (last 10 frames)
        if len(self.tracked_positions[person_id]) > 10:
            self.tracked_positions[person_id] = self.tracked_positions[person_id][-10:]
        
        # Need at least 2 positions to detect crossing
        if len(self.tracked_positions[person_id]) < 2:
            return None
        
        previous_center = self.tracked_positions[person_id][-2]
        
        # Check line crossing based on orientation
        direction = None
        
        if self.counting_line['orientation'] == 'vertical':
            # Vertical line: check horizontal crossing
            line_x = line_start[0]  # Vertical line has same x for start and end
            
            # Right to Left = IN (crossing from right side to left side)
            if previous_center[0] >= line_x > current_center[0]:
                direction = 'in'  # Right to left
            # Left to Right = OUT (crossing from left side to right side)
            elif previous_center[0] <= line_x < current_center[0]:
                direction = 'out'  # Left to right
                
        elif self.counting_line['orientation'] == 'horizontal':
            # Horizontal line: check vertical crossing
            line_y = line_start[1]  # Horizontal line has same y for start and end
            
            if previous_center[1] <= line_y < current_center[1]:
                direction = 'in'  # Top to bottom
            elif previous_center[1] >= line_y > current_center[1]:
                direction = 'out'  # Bottom to top
        
        # If crossing detected, mark person as crossed to prevent double counting
        if direction:
            self.crossed_persons.add(person_id)
            
            # Remove from crossed set after some time (cleanup)
            # This is handled in the cleanup method
        
        return direction
    
    def cleanup_tracking(self):
        """Clean up old tracking data"""
        current_time = datetime.now()
        
        # Reset crossed persons periodically (every 30 seconds)
        if (current_time - self.last_reset_time).seconds > 30:
            self.crossed_persons.clear()
            self.last_reset_time = current_time
        
        # Clean up old position tracking for persons not seen recently
        active_person_ids = set(self.tracker.objects.keys())
        for person_id in list(self.tracked_positions.keys()):
            if person_id not in active_person_ids:
                del self.tracked_positions[person_id]
                self.crossed_persons.discard(person_id)
    
    def process_frame(self, frame):
        """
        Process a single frame for people counting
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with detections and counting line
        """
        self.frame_count += 1
        original_frame = frame.copy()
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Detect persons in frame
        detections = self.detector.detect_persons(frame)
        
        # Update tracker with detections
        tracked_objects = self.tracker.update(detections)
        
        # Check for line crossings
        for person_id, center in tracked_objects.items():
            direction = self.check_line_crossing(person_id, center, frame_width, frame_height)
            
            if direction:
                if direction == 'in':
                    self.in_count += 1
                    logger.info(f"Person entered - Channel {self.channel_id}, Total IN: {self.in_count}")
                elif direction == 'out':
                    self.out_count += 1
                    logger.info(f"Person exited - Channel {self.channel_id}, Total OUT: {self.out_count}")
                
                # Update database
                if self.db_manager and self.app:
                    try:
                        with self.app.app_context():
                            self.db_manager.update_footfall_count(self.channel_id, direction)
                            self.db_manager.log_detection_event(
                                self.channel_id,
                                'PeopleCounter',
                                f'person_{direction}',
                                {'center': center, 'direction': direction}
                            )
                    except Exception as e:
                        logger.error(f"Database update error: {e}")
                
                # Send real-time update
                self.socketio.emit('count_update', {
                    'channel_id': self.channel_id,
                    'in_count': self.in_count,
                    'out_count': self.out_count,
                    'direction': direction
                })
        
        # Clean up tracking data periodically
        if self.frame_count % 30 == 0:  # Every 30 frames
            self.cleanup_tracking()
        
        # Annotate frame
        annotated_frame = self.annotate_frame(original_frame, detections, tracked_objects)
        
        return annotated_frame
    
    def annotate_frame(self, frame, detections, tracked_objects):
        """
        Annotate frame with detections, tracking, and counting line
        
        Args:
            frame: Input frame
            detections: Person detections
            tracked_objects: Tracked person objects
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Draw person detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Person {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw tracking IDs
        for person_id, center in tracked_objects.items():
            cv2.circle(annotated_frame, center, 5, (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"ID:{person_id}", 
                       (center[0] + 10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw counting line
        line_start, line_end = self.get_counting_line_pixels(frame_width, frame_height)
        cv2.line(annotated_frame, line_start, line_end, (0, 255, 255), 3)
        
        # Draw direction labels
        if self.counting_line['orientation'] == 'vertical':
            # Right to Left = IN, Left to Right = OUT
            cv2.putText(annotated_frame, "RIGHT→IN", (frame_width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "OUT←LEFT", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            # Top: IN, Bottom: OUT
            cv2.putText(annotated_frame, "TOP→IN", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "OUT←BOTTOM", (10, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Draw counts
        cv2.putText(annotated_frame, f"IN: {self.in_count}", 
                   (10, frame_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"OUT: {self.out_count}", 
                   (10, frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        return annotated_frame
    
    def get_current_counts(self):
        """Get current in/out counts"""
        return {
            'in_count': self.in_count,
            'out_count': self.out_count,
            'net_count': self.in_count - self.out_count
        }
    
    def reset_counts(self):
        """Reset daily counts (typically called at midnight)"""
        self.in_count = 0
        self.out_count = 0
        self.crossed_persons.clear()
        self.tracked_positions.clear()
        logger.info(f"Counts reset for channel {self.channel_id}")
    
    def get_status(self):
        """Get module status information"""
        return {
            'module': 'PeopleCounter',
            'channel_id': self.channel_id,
            'status': 'active',
            'in_count': self.in_count,
            'out_count': self.out_count,
            'net_count': self.in_count - self.out_count,
            'counting_line': self.counting_line,
            'frame_count': self.frame_count,
            'tracked_objects': len(self.tracker.objects)
        }