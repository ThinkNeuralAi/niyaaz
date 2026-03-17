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

from modules.model_manager import get_shared_model, release_shared_model

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path='models/best.engine', confidence_threshold=0.5, device='auto', img_size=640, person_class_id=None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file (TensorRT .engine or PyTorch .pt)
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            img_size: Input image size (must match TensorRT engine build size - default 640)
            person_class_id: Class ID for person detection (0 for COCO/YOLOv11, 12 for custom best.pt)
                            If None, auto-detects based on model path
        """
        if YOLO is None:
            raise ImportError("Ultralytics YOLO could not be imported")
            
        self.confidence_threshold = confidence_threshold
        self.device = self._select_device(device)
        self.img_size = img_size  # Must match TensorRT engine build size (640)
        self.model_path = model_path
        self._using_shared = False
        
        # Log which model is being loaded
        logger.info(f"Initializing YOLODetector with model: {model_path}")
        
        # Use shared model from model_manager to avoid duplicate GPU memory allocation
        try:
            self.model = get_shared_model(model_path, self.device)
            self._using_shared = True
            logger.info(f"YOLODetector using shared model: {model_path}")
        except Exception as e:
            logger.warning(f"Shared model load failed ({e}), falling back to direct load")
            self.model = YOLO(model_path, task='detect')
            self._using_shared = False
            logger.info(f"YOLODetector direct-loaded: {model_path}")
        
        self.is_engine = model_path.endswith('.engine') or model_path.endswith('.trt')

        # Validate best_labels.txt against model names (if available)
        self._validate_labels_file(model_path)
        
        if person_class_id is not None:
            self.person_class_id = person_class_id
            logger.info(f"YOLODetector explicitly set person_class_id={self.person_class_id}")
        else:
            # Try to infer person class id from model names when available
            self.person_class_id = None
            model_names = None
            try:
                model_names = getattr(self.model, 'names', None) or getattr(self.model, 'model', None) and getattr(self.model.model, 'names', None)
            except Exception:
                model_names = None

            if isinstance(model_names, dict):
                normalized_names = {int(c): str(n).lower() for c, n in model_names.items()}
                person_class_id_candidates = [c for c, n in normalized_names.items() if 'person' in n or n == 'person']
                if person_class_id_candidates:
                    self.person_class_id = person_class_id_candidates[0]
                    logger.info(f"Auto-detected person_class_id={self.person_class_id} from model names")
                else:
                    # Fallback heuristics
                    if 'yolov11' in model_path.lower() or 'yolov8' in model_path.lower() or 'yolov5' in model_path.lower():
                        self.person_class_id = 0
                        logger.info(f"Auto-detected person_class_id=0 for COCO model: {model_path}")
                    else:
                        self.person_class_id = 0
                        logger.warning(f"Could not infer person class from model names; defaulting person_class_id=0 for model: {model_path}")
            else:
                # Unknown model names mapping; fallback
                if 'yolov11' in model_path.lower() or 'yolov8' in model_path.lower() or 'yolov5' in model_path.lower():
                    self.person_class_id = 0
                    logger.info(f"Auto-detected person_class_id=0 for COCO model: {model_path}")
                else:
                    self.person_class_id = 0
                    logger.warning(f"Model class names unavailable; defaulting person_class_id=0 for model: {model_path}")
    
    def __del__(self):
        if getattr(self, '_using_shared', False):
            try:
                release_shared_model(self.model_path, self.device)
            except Exception:
                pass
    
    def _resolve_label_file(self, model_path):
        # Use model-specific labels by path.
        if 'yolo11n' in model_path.lower() or 'coco' in model_path.lower() or 'yolov8n' in model_path.lower():
            coco_path = os.path.join('config', 'coco_labels.txt')
            if os.path.exists(coco_path):
                return coco_path
        best_path = os.path.join('config', 'best_labels.txt')
        if os.path.exists(best_path):
            return best_path
        return os.path.join('config', 'labels.txt')

    def _validate_labels_file(self, model_path):
        """Validate detected labels against best/coco labels depending on model."""
        labels_path = self._resolve_label_file(model_path)
        if not os.path.exists(labels_path):
            logger.warning(f"Label file not found at {labels_path}; skipping label validation")
            return
        try:
            model_names = getattr(self.model, 'names', None) or getattr(self.model, 'model', None) and getattr(self.model.model, 'names', None)
            if not isinstance(model_names, dict):
                return
            model_labels = [str(v) for _, v in sorted(model_names.items())]
            file_labels = [line.strip() for line in open(labels_path, 'r').read().splitlines() if line.strip()]

            generic_names = all(str(n).startswith('class') for n in model_labels)
            if generic_names:
                logger.warning(f"Model names are generic class0..classN; using {labels_path} fallback where available")
            elif len(model_labels) != len(file_labels) or any(a.lower() != b.lower() for a, b in zip(model_labels, file_labels)):
                logger.warning(f"Label mismatch detected between model classes and {labels_path}")
                logger.warning(f" Model labels ({len(model_labels)}): {model_labels}")
                logger.warning(f" File labels ({len(file_labels)}): {file_labels}")
                logger.warning(f"Please update {labels_path} to match the model class labels.")
            else:
                logger.info(f"{labels_path} matches model class labels")
        except Exception as e:
            logger.warning(f"Could not validate {labels_path} vs model names: {e}")

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
        Detect persons in frame with optimized settings
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        try:
            # Check if frame is valid
            if frame is None or frame.size == 0:
                logger.warning("detect_persons: Frame is None or empty")
                return []

            original_h, original_w = frame.shape[:2]
            infer_frame = cv2.resize(frame, (self.img_size, self.img_size))

            # **DEBUG: First try without class filtering to see all detections**
            # This helps diagnose if the model is detecting anything at all
            results_all = self.model(
                infer_frame, 
                verbose=False,
                imgsz=self.img_size,
            )
            
            # Log all detections for debugging
            all_detections_count = 0
            all_classes_found = set()
            for result in results_all:
                boxes = result.boxes
                if boxes is not None:
                    cls = boxes.cls.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    all_detections_count += len(boxes)
                    for class_id, confidence in zip(cls, conf):
                        all_classes_found.add((int(class_id), float(confidence)))
            
            # Auto-correct person class id if the class is found by name in names mapping
            if self.person_class_id is not None:
                try:
                    model_names = getattr(self.model, 'names', None) or getattr(self.model, 'model', None) and getattr(self.model.model, 'names', None)
                except Exception:
                    model_names = None

                if isinstance(model_names, dict):
                    names_lower = {int(k): str(v).lower() for k, v in model_names.items()}
                    person_candidates = [k for k, v in names_lower.items() if 'person' in v]
                    if person_candidates and self.person_class_id not in person_candidates:
                        old_id = self.person_class_id
                        self.person_class_id = person_candidates[0]
                        logger.info(f"Updated person_class_id from {old_id} to {self.person_class_id} based on model class names")

            # Now run with person class filtering
            # Use the confidence threshold directly (no reduction) for yolov8n.pt
            # YOLOv8 models work well with standard thresholds
            results = self.model(
                infer_frame, 
                verbose=False,
                imgsz=self.img_size,      # Configurable image size
                device=self.device         # Explicit device
            )
            
            detections = []
            
            scale_x = original_w / self.img_size
            scale_y = original_h / self.img_size

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Extract detection data
                    xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                    conf = boxes.conf.cpu().numpy()  # Confidence scores
                    cls = boxes.cls.cpu().numpy()   # Class IDs
                    
                    # Filter for person detections with sufficient confidence
                    for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                        class_id_int = int(class_id)
                        if class_id_int == self.person_class_id and confidence >= self.confidence_threshold:
                            x1 = int(box[0] * scale_x)
                            y1 = int(box[1] * scale_y)
                            x2 = int(box[2] * scale_x)
                            y2 = int(box[3] * scale_y)
                            
                            # Calculate center point and bottom center (for line crossing)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            bottom_center_x = center_x
                            bottom_center_y = y2
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'class_id': class_id_int,
                                'class_name': 'person',
                                'center': (center_x, center_y),
                                'bottom_center': (bottom_center_x, bottom_center_y),
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            
                            detections.append(detection)
            
            # **DEBUG LOGGING** - Log every 30 calls to avoid spam, but always log first 10 calls
            if not hasattr(self, '_detect_call_count'):
                self._detect_call_count = 0
            self._detect_call_count += 1
            
            should_log = (self._detect_call_count <= 10) or (self._detect_call_count % 30 == 0)
            
            if should_log:
                logger.info(f"🔍 YOLO Detection Debug (call #{self._detect_call_count}):")
                logger.info(f"   - Model: {self.model_path}")
                logger.info(f"   - Person class ID expected: {self.person_class_id}")
                logger.info(f"   - Confidence threshold: {self.confidence_threshold}")
                logger.info(f"   - All detections (conf>=0.25): {all_detections_count}")
                if all_classes_found:
                    logger.info(f"   - All classes found: {sorted(all_classes_found)}")
                else:
                    logger.warning(f"   - ⚠️ No detections found at all (conf>=0.25)!")
                logger.info(f"   - Person detections (class={self.person_class_id}, conf>={self.confidence_threshold}): {len(detections)}")
                if len(detections) > 0:
                    logger.info(f"   - Sample person detection: bbox={detections[0]['bbox']}, conf={detections[0]['confidence']:.2f}, center={detections[0]['center']}")
                elif all_detections_count > 0:
                    logger.warning(f"   - ⚠️ Found {all_detections_count} detections but NONE are person class {self.person_class_id}!")
                else:
                    logger.warning(f"   - ⚠️ No detections found - model may not be detecting anything in this frame")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def detect_all_classes(self, frame, target_classes=None):
        """
        Detect all classes (or specific classes) in the frame
        Used by modules that need to detect uniforms and other classes beyond just 'person'
        
        Args:
            frame: Input frame (numpy array)
            target_classes: Optional list of class names to filter (e.g., ['person', 'uniform_beige'])
                          If None, returns all detected classes
        
        Returns:
            List of detection dictionaries with keys: bbox, confidence, class_id, class_name
        """
        if frame is None or frame.size == 0:
            logger.warning("detect_all_classes: Frame is None or empty")
            return []
        
        try:
            # Run YOLO detection without class filtering
            original_h, original_w = frame.shape[:2]
            infer_frame = cv2.resize(frame, (self.img_size, self.img_size))
            results = self.model.predict(
                infer_frame,
                verbose=False,             # Silent mode
                imgsz=self.img_size,      # Configurable image size
                conf=self.confidence_threshold,  # Filter at inference time
                iou=0.45,                  # NMS IoU threshold
                max_det=50,                # Max detections
                device=self.device         # Explicit device
                # No classes filter - detect all classes
            )
            scale_x = original_w / self.img_size
            scale_y = original_h / self.img_size
            
            detections = []
            total_before_filter = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    total_before_filter = len(boxes)
                    # Extract detection data
                    xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                    conf = boxes.conf.cpu().numpy()  # Confidence scores
                    cls = boxes.cls.cpu().numpy()   # Class IDs
                    
                    # Get class names from model
                    names = result.names  # Dict mapping class_id to class_name
                    
                    # Process each detection
                    for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                        if confidence >= self.confidence_threshold:
                            class_id_int = int(class_id)
                            class_name = names.get(class_id_int, None)
                            if class_name is None or (isinstance(class_name, str) and class_name.startswith('class') and class_name[5:].isdigit()):
                                labels_path = self._resolve_label_file(self.model_path)
                                if os.path.exists(labels_path):
                                    labels = [line.strip() for line in open(labels_path, 'r').read().splitlines() if line.strip()]
                                    if class_id_int < len(labels):
                                        class_name = labels[class_id_int]
                            if class_name is None:
                                class_name = f'class_{class_id_int}'
                            # Report mapping once per detection call for debugging
                            if not hasattr(self, '_detect_label_debug'):
                                self._detect_label_debug = 0
                            if self._detect_label_debug < 3:
                                logger.info(f"Mapped class_id {class_id_int} -> '{class_name}' (from {names.get(class_id_int, 'unknown')})")
                                self._detect_label_debug += 1
                            # Filter by target classes if specified
                            if target_classes and class_name not in target_classes:
                                # Log what class was detected but filtered out
                                if i == 0:  # Only log first filtered detection to avoid spam
                                    logger.info(f"🔍 Detected '{class_name}' (conf={confidence:.2f}) but not in target classes")
                                continue
                            
                            x1 = int(box[0] * scale_x)
                            y1 = int(box[1] * scale_y)
                            x2 = int(box[2] * scale_x)
                            y2 = int(box[3] * scale_y)
                            
                            # Calculate center point
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            bottom_center_x = center_x
                            bottom_center_y = y2
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'class_id': class_id_int,
                                'class_name': class_name,
                                'center': (center_x, center_y),
                                'bottom_center': (bottom_center_x, bottom_center_y),
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            
                            detections.append(detection)
            
            # Debug logging
            if total_before_filter > 0 and len(detections) == 0 and target_classes:
                logger.warning(f"🔍 detect_all_classes: {total_before_filter} raw detections, but 0 after filtering for classes: {target_classes}")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during all-class detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            'input_size': self.img_size,
            'optimizations': {
                'half_precision': torch.cuda.is_available(),
                'max_detections': 50,
                'person_only': True
            }
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