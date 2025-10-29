"""
YOLO Detection Service for Person Detection
Uses YOLOv8 for real-time person detection
"""
import cv2
import numpy as np
import logging
import os
import sys

# Configure PyTorch environment before any imports
os.environ['PYTORCH_WEIGHTS_ONLY'] = 'False'
os.environ['TORCH_DISABLE_WEIGHTS_ONLY'] = '1'

# Set up PyTorch to accept YOLO models
import torch

# Disable weights_only globally by patching the default
if hasattr(torch.serialization, '_get_default_pickle_version'):
    original_get_default = torch.serialization._get_default_pickle_version
    def patched_get_default():
        return 2
    torch.serialization._get_default_pickle_version = patched_get_default

# Monkey patch torch.load to always set weights_only=False
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

# Import ultralytics after patching
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    # Add DetectionModel to safe globals for newer PyTorch versions
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except ImportError as e:
    print(f"Failed to import ultralytics: {e}")
    YOLO = None

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path='models/yolo11n.pt', confidence_threshold=0.5, device='auto'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        if YOLO is None:
            raise ImportError("Ultralytics YOLO could not be imported")
            
        self.confidence_threshold = confidence_threshold
        self.device = self._select_device(device)
        
        try:
            # Apply additional safety patches specifically for model loading
            with torch.serialization.safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.nn.modules.conv.Conv',
                'ultralytics.nn.modules.block.C2f',
                'ultralytics.nn.modules.block.SPPF',
                'ultralytics.nn.modules.head.Detect'
            ]):
                self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Safe context loading failed: {e}")
            try:
                # Force disable weights_only for this specific load
                original_env = os.environ.get('PYTORCH_DISABLE_WEIGHTS_ONLY', '0')
                os.environ['PYTORCH_DISABLE_WEIGHTS_ONLY'] = '1'
                
                # Temporary patch for this specific model load
                def force_unsafe_load(*args, **kwargs):
                    kwargs.pop('weights_only', None)  # Remove if present
                    return original_load(*args, weights_only=False, **kwargs)
                
                torch.load = force_unsafe_load
                self.model = YOLO(model_path)
                torch.load = safe_load  # Restore our patched version
                
                self.model.to(self.device)
                logger.info(f"YOLO model loaded with forced unsafe loading on {self.device}")
                
                # Restore environment
                os.environ['PYTORCH_DISABLE_WEIGHTS_ONLY'] = original_env
                
            except Exception as e2:
                logger.error(f"All YOLO loading attempts failed: {e2}")
                raise Exception(f"Could not load YOLO model. Error: {e2}")
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
    
    def _select_device(self, device):
        """Select the best available device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def detect_persons(self, frame):
        """
        Detect persons in frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Extract detection data
                    xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                    conf = boxes.conf.cpu().numpy()  # Confidence scores
                    cls = boxes.cls.cpu().numpy()   # Class IDs
                    
                    # Filter for person detections with sufficient confidence
                    for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                        if int(class_id) == self.person_class_id and confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Calculate center point and bottom center (for line crossing)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            bottom_center_x = center_x
                            bottom_center_y = y2
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'class_id': int(class_id),
                                'class_name': 'person',
                                'center': (center_x, center_y),
                                'bottom_center': (bottom_center_x, bottom_center_y),
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            return []
    
    def draw_detections(self, frame, detections, draw_confidence=True, color=(0, 255, 0)):
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            draw_confidence: Whether to draw confidence scores
            color: Color for bounding boxes (BGR)
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score
            if draw_confidence:
                label = f"Person {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw center point
            center_x, center_y = detection['center']
            cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        return annotated_frame
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8',
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'input_size': getattr(self.model, 'imgsz', 640)
        }

class PersonTracker:
    """Simple person tracker for counting applications"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        Initialize person tracker
        
        Args:
            max_disappeared: Maximum frames a person can be missing before removal
            max_distance: Maximum distance for person association
        """
        self.next_id = 0
        self.objects = {}  # {id: {'center': (x, y), 'disappeared': count}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register_person(self, center):
        """Register a new person"""
        self.objects[self.next_id] = {
            'center': center,
            'disappeared': 0
        }
        self.next_id += 1
        return self.next_id - 1
    
    def deregister_person(self, person_id):
        """Remove a person from tracking"""
        if person_id in self.objects:
            del self.objects[person_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary of tracked objects {id: center}
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for person_id in list(self.objects.keys()):
                self.objects[person_id]['disappeared'] += 1
                
                # Remove if disappeared for too long
                if self.objects[person_id]['disappeared'] > self.max_disappeared:
                    self.deregister_person(person_id)
            
            return {}
        
        # Extract centers from detections
        input_centers = [det['center'] for det in detections]
        
        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for center in input_centers:
                self.register_person(center)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centers = [self.objects[obj_id]['center'] for obj_id in object_ids]
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centers)[:, np.newaxis] - np.array(input_centers), axis=2)
            
            # Find the minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_idxs = set()
            used_col_idxs = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue
                
                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id]['center'] = input_centers[col]
                    self.objects[object_id]['disappeared'] = 0
                    
                    used_row_idxs.add(row)
                    used_col_idxs.add(col)
            
            # Handle unmatched detections and objects
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
            
            # Mark unmatched objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.objects[object_id]['disappeared'] += 1
                    
                    if self.objects[object_id]['disappeared'] > self.max_disappeared:
                        self.deregister_person(object_id)
            
            # Register new objects
            else:
                for col in unused_col_idxs:
                    self.register_person(input_centers[col])
        
        # Return current tracked objects
        return {obj_id: obj_data['center'] for obj_id, obj_data in self.objects.items()}