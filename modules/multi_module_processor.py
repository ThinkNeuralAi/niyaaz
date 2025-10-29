"""
Multi-Module Video Processor
Handles multiple analysis modules on the same video feed for efficiency
"""
import cv2
import threading
import time
import logging
from queue import Queue
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MultiModuleVideoProcessor:
    def __init__(self, video_source, channel_id, fps_limit=30):
        """
        Initialize Multi-Module Video Processor
        
        Args:
            video_source: Path to video file or camera index
            channel_id: Unique identifier for the video channel
            fps_limit: Maximum FPS for processing
        """
        self.video_source = video_source
        self.channel_id = channel_id
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        
        # Processing modules
        self.modules = {}  # {module_name: module_instance}
        self.module_results = {}  # {module_name: latest_result}
        
        # Threading and control
        self.is_running = False
        self.processing_thread = None
        
        # Video capture
        self.cap = None
        
        # Shared resources
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.frames_processed = 0
        self.start_time = None
        
        logger.info(f"Multi-module processor initialized for channel {channel_id}")
    
    def add_module(self, module_name: str, module_instance):
        """
        Add an analysis module to this video processor
        
        Args:
            module_name: Name of the module (e.g., 'PeopleCounter', 'QueueMonitor')
            module_instance: Instance of the analysis module
        """
        self.modules[module_name] = module_instance
        self.module_results[module_name] = None
        logger.info(f"Added module '{module_name}' to channel {self.channel_id}")
    
    def remove_module(self, module_name: str):
        """Remove an analysis module"""
        if module_name in self.modules:
            del self.modules[module_name]
            del self.module_results[module_name]
            logger.info(f"Removed module '{module_name}' from channel {self.channel_id}")
    
    def get_active_modules(self) -> List[str]:
        """Get list of active module names"""
        return list(self.modules.keys())
    
    def initialize_capture(self):
        """Initialize video capture"""
        try:
            # Determine source type
            if isinstance(self.video_source, str):
                if self.video_source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                    # RTSP/network stream
                    self.cap = cv2.VideoCapture(self.video_source)
                    # Configure for RTSP streams
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
                    logger.info(f"Initializing RTSP stream: {self.video_source}")
                    
                elif self.video_source.isdigit():
                    # Camera index
                    self.cap = cv2.VideoCapture(int(self.video_source))
                    logger.info(f"Initializing camera: {self.video_source}")
                    
                else:
                    # File path
                    self.cap = cv2.VideoCapture(self.video_source)
                    logger.info(f"Initializing video file: {self.video_source}")
            else:
                # Assume numeric camera index
                self.cap = cv2.VideoCapture(self.video_source)
                logger.info(f"Initializing camera index: {self.video_source}")
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.video_source}")
            
            # Additional RTSP optimization settings
            if isinstance(self.video_source, str) and self.video_source.startswith(('rtsp://', 'rtmp://')):
                # RTSP-specific optimizations
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                # Try to set a reasonable timeout
                self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Validate frame dimensions
            if self.frame_width <= 0 or self.frame_height <= 0:
                # Set default dimensions for RTSP streams that don't report correctly
                self.frame_width = 640
                self.frame_height = 480
                logger.warning(f"Invalid frame dimensions detected, using defaults: {self.frame_width}x{self.frame_height}")
            
            logger.info(f"Video capture initialized: {self.frame_width}x{self.frame_height} @ {self.video_fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize video capture: {e}")
            return False
    
    def start(self):
        """Start video processing"""
        if self.is_running:
            logger.warning("Multi-module processor is already running")
            return False
        
        if not self.initialize_capture():
            return False
        
        self.is_running = True
        self.start_time = time.time()
        
        # Load configuration for all modules
        for module in self.modules.values():
            if hasattr(module, 'load_configuration'):
                module.load_configuration()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info(f"Multi-module processing started for channel {self.channel_id}")
        return True
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        if self.cap:
            self.cap.release()
        
        logger.info(f"Multi-module processing stopped for channel {self.channel_id}")
    
    def _processing_loop(self):
        """Main processing loop"""
        last_frame_time = 0
        consecutive_failures = 0
        max_failures = 30  # Allow up to 30 consecutive failures before giving up
        
        # Determine if this is a live stream
        is_live_stream = isinstance(self.video_source, str) and (
            self.video_source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or
            self.video_source.isdigit()
        )
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Respect FPS limit
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive frame read failures. Stopping processing.")
                        break
                    
                    # For live streams, try to reconnect
                    if is_live_stream:
                        logger.info("Attempting to reconnect to stream...")
                        self.cap.release()
                        time.sleep(2)  # Wait before reconnecting
                        if not self.initialize_capture():
                            logger.error("Failed to reconnect to stream")
                            break
                        continue
                    
                    # For video files, loop back to beginning
                    elif isinstance(self.video_source, str) and not self.video_source.isdigit():
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.error("Failed to read frame from video source")
                        break
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                # Validate frame
                if frame is None or frame.size == 0:
                    logger.warning("Received empty frame")
                    continue
                
                # Resize frame if too large for better performance
                if frame.shape[1] > 1280:
                    scale_factor = 1280 / frame.shape[1]
                    new_width = 1280
                    new_height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Store raw frame
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
                
                # Process frame through all modules
                annotated_frame = self._process_through_modules(frame)
                
                # Update latest annotated frame
                with self.frame_lock:
                    self.latest_annotated_frame = annotated_frame
                
                self.frames_processed += 1
                last_frame_time = current_time
                
                # Log performance statistics periodically
                if self.frames_processed % 300 == 0:  # Every 300 frames
                    elapsed_time = current_time - self.start_time
                    avg_fps = self.frames_processed / elapsed_time
                    logger.info(f"Multi-module processing stats - Channel {self.channel_id}: "
                              f"{self.frames_processed} frames, avg FPS: {avg_fps:.2f}, "
                              f"active modules: {list(self.modules.keys())}, "
                              f"source: {'LIVE' if is_live_stream else 'FILE'}")
                
            except Exception as e:
                logger.error(f"Error in multi-module processing loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive processing errors. Stopping.")
                    break
                if not self.is_running:
                    break
                time.sleep(0.1)
        
        logger.info(f"Multi-module processing loop ended for channel {self.channel_id}")
        
        # Cleanup
        if self.cap:
            self.cap.release()
    
    def _process_through_modules(self, frame):
        """
        Process frame through all active modules and combine annotations
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with all module outputs combined
        """
        if not self.modules:
            return frame
        
        # Start with original frame
        combined_frame = frame.copy()
        
        # Process through each module
        for module_name, module in self.modules.items():
            try:
                # Each module processes the original frame
                module_result = module.process_frame(frame.copy())
                
                # Store module result
                self.module_results[module_name] = {
                    'frame': module_result,
                    'timestamp': time.time(),
                    'status': module.get_status() if hasattr(module, 'get_status') else {}
                }
                
            except Exception as e:
                logger.error(f"Error processing frame through module {module_name}: {e}")
                self.module_results[module_name] = {
                    'frame': frame.copy(),
                    'timestamp': time.time(),
                    'status': {'error': str(e)}
                }
        
        # Combine annotations from all modules
        combined_frame = self._combine_module_annotations(frame, self.module_results)
        
        return combined_frame
    
    def _combine_module_annotations(self, original_frame, module_results):
        """
        Combine visual annotations from multiple modules
        
        Args:
            original_frame: Original video frame
            module_results: Dictionary of module processing results
            
        Returns:
            Frame with combined annotations from all modules
        """
        combined_frame = original_frame.copy()
        frame_height, frame_width = combined_frame.shape[:2]
        
        # Draw a header showing active modules
        header_height = 40
        header_frame = np.zeros((header_height, frame_width, 3), dtype=np.uint8)
        
        # List active modules
        module_names = list(module_results.keys())
        if module_names:
            header_text = f"Active: {', '.join(module_names)}"
            cv2.putText(header_frame, header_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # For each module, extract key visual elements
        y_offset = 60  # Start below header
        
        for i, (module_name, result) in enumerate(module_results.items()):
            if result and 'frame' in result:
                module_frame = result['frame']
                
                # Extract and overlay key information from each module
                if module_name == 'PeopleCounter':
                    self._overlay_people_counter_info(combined_frame, result, y_offset + i * 30)
                elif module_name == 'QueueMonitor':
                    self._overlay_queue_monitor_info(combined_frame, result, y_offset + i * 30)
                
                # You can add more module-specific overlays here
        
        # Combine header with main frame
        final_frame = np.vstack([header_frame, combined_frame])
        
        return final_frame
    
    def _overlay_people_counter_info(self, frame, result, y_pos):
        """Overlay people counter information"""
        if 'status' in result:
            status = result['status']
            in_count = status.get('in_count', 0)
            out_count = status.get('out_count', 0)
            
            # Draw counts
            text = f"People Counter - IN: {in_count} | OUT: {out_count}"
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _overlay_queue_monitor_info(self, frame, result, y_pos):
        """Overlay queue monitor information"""
        if 'status' in result:
            status = result['status']
            queue_count = status.get('queue_count', 0)
            counter_count = status.get('counter_count', 0)
            
            # Draw counts
            text = f"Queue Monitor - Queue: {queue_count} | Counter: {counter_count}"
            color = (0, 255, 255) if queue_count <= 2 else (0, 0, 255)  # Yellow if normal, red if crowded
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def get_latest_frame(self, module_name=None):
        """
        Get the latest processed frame
        
        Args:
            module_name: If specified, get frame from specific module, otherwise combined frame
            
        Returns:
            Latest frame (combined or module-specific)
        """
        with self.frame_lock:
            if module_name and module_name in self.module_results:
                result = self.module_results[module_name]
                if result and 'frame' in result:
                    return result['frame'].copy()
            
            # Return combined frame if available
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            elif self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            else:
                # Return blank frame if nothing available
                return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def get_module_result(self, module_name):
        """Get the latest result from a specific module"""
        return self.module_results.get(module_name)
    
    def get_all_module_results(self):
        """Get results from all modules"""
        return self.module_results.copy()
    
    def get_status(self):
        """Get processor status including all modules"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        module_statuses = {}
        for module_name, module in self.modules.items():
            if hasattr(module, 'get_status'):
                module_statuses[module_name] = module.get_status()
        
        return {
            'channel_id': self.channel_id,
            'is_running': self.is_running,
            'video_source': self.video_source,
            'frames_processed': self.frames_processed,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'target_fps': self.fps_limit,
            'frame_size': f"{getattr(self, 'frame_width', 0)}x{getattr(self, 'frame_height', 0)}",
            'active_modules': list(self.modules.keys()),
            'module_statuses': module_statuses
        }