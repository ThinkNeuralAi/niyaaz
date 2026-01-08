"""
DeepStream SDK Processor
Handles RTSP ingestion, NVDEC decode, preprocessing, inference, and tracking using DeepStream SDK
"""
import gi
import sys
import logging
import numpy as np
import cv2
from typing import Callable, Optional, List, Dict, Any
import threading
import time

# Attempt to import GStreamer and DeepStream
try:
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import GObject, Gst
    
    # Try importing DeepStream bindings
    sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
    import pyds
    
    DEEPSTREAM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ DeepStream SDK bindings loaded successfully")
except (ImportError, ValueError) as e:
    DEEPSTREAM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ DeepStream SDK not available: {e}")
    pyds = None

def check_deepstream_availability() -> bool:
    """Check if DeepStream SDK is available"""
    return DEEPSTREAM_AVAILABLE

class DeepStreamProcessor:
    """
    DeepStream-based video processor with hardware acceleration
    
    Pipeline: RTSP Source → NVDEC → Preprocessing → Inference → Tracker → Output
    """
    
    def __init__(
        self,
        rtsp_url: str,
        channel_id: str,
        model_path: str,
        config_file: Optional[str] = None,
        tracker_config: Optional[str] = None,
        input_width: int = 1920,
        input_height: int = 1080
    ):
        """
        Initialize DeepStream processor
        
        Args:
            rtsp_url: RTSP stream URL
            channel_id: Unique channel identifier
            model_path: Path to YOLO model (ONNX or TensorRT engine)
            config_file: Path to nvinfer config file
            tracker_config: Path to tracker config file
            input_width: Input frame width
            input_height: Input frame height
        """
        if not DEEPSTREAM_AVAILABLE:
            raise RuntimeError("DeepStream SDK is not available")
        
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.model_path = model_path
        self.config_file = config_file
        self.tracker_config = tracker_config
        self.input_width = input_width
        self.input_height = input_height
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Pipeline components
        self.pipeline = None
        self.bus = None
        self.loop = None
        
        # Callbacks
        self.frame_callback: Optional[Callable] = None
        self.detection_callback: Optional[Callable] = None
        
        # Status
        self.is_running = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_count': 0,
            'avg_inference_time': 0,
            'decode_errors': 0
        }
        
        logger.info(f"DeepStream processor initialized for channel {channel_id}")
    
    def register_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Register callback for processed frames"""
        self.frame_callback = callback
    
    def register_detection_callback(self, callback: Callable[[List[Dict]], None]):
        """Register callback for detections"""
        self.detection_callback = callback
    
    def _create_pipeline(self) -> bool:
        """Create DeepStream GStreamer pipeline"""
        try:
            logger.info(f"Creating DeepStream pipeline for {self.rtsp_url}")
            
            # Create pipeline
            self.pipeline = Gst.Pipeline()
            if not self.pipeline:
                logger.error("Failed to create pipeline")
                return False
            
            # Create elements
            # 1. RTSP Source with hardware acceleration
            source = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
            source.set_property('location', self.rtsp_url)
            source.set_property('latency', 200)
            source.set_property('drop-on-latency', True)
            source.set_property('protocols', 'tcp')
            
            # 2. RTP Depayloader
            depay = Gst.ElementFactory.make("rtph264depay", "depay")
            
            # 3. H264 Parser
            h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
            
            # 4. NVDEC - Hardware decoder
            decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
            # Note: enable-max-performance property doesn't exist in nvv4l2decoder
            decoder.set_property('drop-frame-interval', 0)
            decoder.set_property('num-extra-surfaces', 1)
            
            # 5. Stream Muxer
            streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
            streammux.set_property('width', self.input_width)
            streammux.set_property('height', self.input_height)
            streammux.set_property('batch-size', 1)
            streammux.set_property('batched-push-timeout', 40000)
            streammux.set_property('live-source', 1)
            
            # 6. Primary Inference Engine (YOLO)
            pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
            if self.config_file:
                pgie.set_property('config-file-path', self.config_file)
            else:
                # Use default YOLO config
                pgie.set_property('config-file-path', 'config/deepstream_yolo_config.txt')
            
            # 7. Tracker (Disabled - will use CPU tracking in Python)
            # tracker = Gst.ElementFactory.make("nvtracker", "tracker")
            
            # 8. Video Converter
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
            
            # 9. Caps filter
            caps = Gst.ElementFactory.make("capsfilter", "filter")
            caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
            
            # 10. OSD for drawing bboxes
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            
            # 11. Final video converter
            nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
            
            # 12. Caps filter for output
            caps2 = Gst.ElementFactory.make("capsfilter", "filter2")
            caps2.set_property("caps", Gst.Caps.from_string("video/x-raw, format=BGRx"))
            
            # 13. App Sink for frame extraction
            appsink = Gst.ElementFactory.make("appsink", "appsink")
            appsink.set_property('emit-signals', True)
            appsink.set_property('sync', False)
            appsink.set_property('max-buffers', 2)
            appsink.set_property('drop', True)
            
            # Check all elements created
            elements = [source, depay, h264parse, decoder, streammux, pgie,
                       nvvidconv, caps, nvosd, nvvidconv2, caps2, appsink]
            
            if not all(elements):
                logger.error("Not all elements could be created")
                return False
            
            # Add elements to pipeline
            for element in elements:
                self.pipeline.add(element)
            
            # Link static elements
            # Note: rtspsrc has dynamic pads, so we connect them in pad-added callback
            if not h264parse.link(decoder):
                logger.error("Failed to link h264parse to decoder")
                return False
            
            # Link decoder to streammux (using request pad)
            sinkpad = streammux.get_request_pad("sink_0")
            if not sinkpad:
                logger.error("Failed to get sink pad from streammux")
                return False
            
            srcpad = decoder.get_static_pad("src")
            if not srcpad:
                logger.error("Failed to get src pad from decoder")
                return False
            
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                logger.error("Failed to link decoder to streammux")
                return False
            
            # Link rest of the pipeline
            if not streammux.link(pgie):
                logger.error("Failed to link streammux to pgie")
                return False
            
            if not pgie.link(nvvidconv):
                logger.error("Failed to link pgie to nvvidconv")
                return False
            
            if not nvvidconv.link(caps):
                logger.error("Failed to link nvvidconv to caps")
                return False
            
            if not caps.link(nvosd):
                logger.error("Failed to link caps to nvosd")
                return False
            
            if not nvosd.link(nvvidconv2):
                logger.error("Failed to link nvosd to nvvidconv2")
                return False
            
            if not nvvidconv2.link(caps2):
                logger.error("Failed to link nvvidconv2 to caps2")
                return False
            
            if not caps2.link(appsink):
                logger.error("Failed to link caps2 to appsink")
                return False
            
            # Connect callbacks
            source.connect("pad-added", self._on_pad_added, depay)
            appsink.connect("new-sample", self._on_new_sample)
            
            # Add probe on tracker output to extract detections
            # Tracker disabled - using CPU tracking in Python
            # tracker_src_pad = tracker.get_static_pad("src")
            # if tracker_src_pad:
            #     tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._tracker_src_pad_buffer_probe, 0)
            
            # Add probe on pgie (inference) output to extract detections
            pgie_src_pad = pgie.get_static_pad("src")
            if pgie_src_pad:
                pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._tracker_src_pad_buffer_probe, 0)
            
            # Create bus and add watch
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()

            self.bus.connect("message", self._bus_call)
            
            logger.info("✅ DeepStream pipeline created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating DeepStream pipeline: {e}", exc_info=True)
            return False
    
    def _on_pad_added(self, src, new_pad, depay):
        """Callback when new pad is added to rtspsrc"""
        caps = new_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        
        # Only link video pads
        if gstname.startswith("application/x-rtp"):
            sink_pad = depay.get_static_pad("sink")
            if not sink_pad.is_linked():
                new_pad.link(sink_pad)
                logger.info("Linked rtspsrc pad to depayloader")
    
    def _tracker_src_pad_buffer_probe(self, pad, info, u_data):
        """Probe to extract detections from tracker output"""
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK
            
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            l_frame = batch_meta.frame_meta_list
            
            detections = []
            
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                except StopIteration:
                    break
                
                l_obj = frame_meta.obj_meta_list
                
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration:
                        break
                    
                    # Extract detection info
                    detection = {
                        'class_id': obj_meta.class_id,
                        'confidence': obj_meta.confidence,
                        'bbox': {
                            'left': obj_meta.rect_params.left,
                            'top': obj_meta.rect_params.top,
                            'width': obj_meta.rect_params.width,
                            'height': obj_meta.rect_params.height
                        },
                        'tracker_id': obj_meta.object_id,
                        'class_name': obj_meta.obj_label if hasattr(obj_meta, 'obj_label') else f"class_{obj_meta.class_id}"
                    }
                    
                    detections.append(detection)
                    
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
            
            # Call detection callback
            if detections and self.detection_callback:
                self.detection_callback(detections)
            
            self.stats['detections_count'] += len(detections)
            
        except Exception as e:
            logger.error(f"Error in tracker probe: {e}")
        
        return Gst.PadProbeReturn.OK
    
    def _on_new_sample(self, appsink):
        """Callback when new frame is available"""
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            
            # Get buffer
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            
            # Map buffer to numpy array
            success, map_info = buf.map(Gst.MapFlags.READ)
            if success:
                # Convert to numpy array
                frame = np.ndarray(
                    shape=(height, width, 4),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                
                # Convert BGRx to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Call frame callback
                if self.frame_callback:
                    self.frame_callback(frame_bgr)
                
                # Update stats
                self.frame_count += 1
                self.stats['frames_processed'] += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                buf.unmap(map_info)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        return Gst.FlowReturn.OK
    
    def _bus_call(self, bus, message):
        """Handle bus messages"""
        t = message.type
        
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream")
            self.stop()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"Warning: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Error: {err}: {debug}")
            self.stats['decode_errors'] += 1
            self.stop()
        
        return True
    
    def start(self) -> bool:
        """Start the DeepStream pipeline"""
        try:
            if self.is_running:
                logger.warning("Pipeline is already running")
                return True
            
            # Create pipeline
            if not self._create_pipeline():
                return False
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Unable to set pipeline to PLAYING state")
                return False
            
            self.is_running = True
            logger.info(f"✅ DeepStream pipeline started for channel {self.channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting DeepStream pipeline: {e}", exc_info=True)
            return False
    
    def stop(self):
        """Stop the DeepStream pipeline"""
        try:
            if not self.is_running:
                return
            
            self.is_running = False
            
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            
            logger.info(f"DeepStream pipeline stopped for channel {self.channel_id}")
            
        except Exception as e:
            logger.error(f"Error stopping DeepStream pipeline: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'fps': self.fps,
            'frames_processed': self.stats['frames_processed'],
            'detections_count': self.stats['detections_count'],
            'decode_errors': self.stats['decode_errors'],
            'is_running': self.is_running
        }
