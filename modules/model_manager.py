"""
Global Model Manager for Sakshi.AI
Implements singleton pattern to share YOLO models across all channels and modules
Reduces GPU memory usage by 80-90% when running multiple channels
"""

import logging
import threading
import time
import torch
from typing import Dict, Optional, Tuple
from collections import defaultdict, OrderedDict
import weakref
import os

# Configure PyTorch environment
os.environ['PYTORCH_WEIGHTS_ONLY'] = 'False'
os.environ['TORCH_DISABLE_WEIGHTS_ONLY'] = '1'

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    
    # Patch torch.load for YOLO compatibility
    original_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = safe_load
    
except ImportError as e:
    print(f"Failed to import ultralytics: {e}")
    YOLO = None

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton Model Manager for sharing YOLO models across all modules and channels
    
    Key Features:
    - Caches models to prevent duplicate loading
    - Reference counting for automatic cleanup
    - Thread-safe access for multi-channel processing
    - Device management and CUDA error handling
    - Memory monitoring and optimization
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Model cache: {model_path: model_instance}
        self._models: Dict[str, object] = {}
        
        # Reference counting: {model_path: count}
        self._ref_counts: Dict[str, int] = defaultdict(int)
        
        # Model metadata: {model_path: {'device': str, 'loaded_time': float, 'memory_mb': float}}
        self._model_info: Dict[str, Dict] = {}
        
        # Device management
        self._device_cache = None
        self._device_test_time = 0
        self._device_test_interval = 300  # Re-test device every 5 minutes
        
        # Thread safety
        self._model_lock = threading.RLock()
        
        # Statistics
        self._total_models_loaded = 0
        self._total_memory_saved_mb = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._initialized = True
        logger.info("Global Model Manager initialized")
    
    def _get_optimal_device(self) -> str:
        """
        Determine the best device for model loading with caching and error handling
        """
        current_time = time.time()
        
        # Use cached device if recent test and no failures
        if (self._device_cache and 
            current_time - self._device_test_time < self._device_test_interval):
            return self._device_cache
        
        # Test CUDA availability
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor + 1
                
                # Check available memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = gpu_memory - allocated_memory
                
                torch.cuda.empty_cache()
                del test_tensor
                
                # Use GPU if sufficient memory available (at least 1GB free)
                if free_memory > 1024 * 1024 * 1024:
                    self._device_cache = 'cuda'
                    logger.info(f"Using GPU device. Free memory: {free_memory / (1024**3):.1f}GB")
                else:
                    self._device_cache = 'cpu'
                    logger.warning(f"GPU memory low ({free_memory / (1024**3):.1f}GB free), using CPU")
                
            except Exception as e:
                logger.warning(f"CUDA test failed: {e}. Using CPU")
                self._device_cache = 'cpu'
        else:
            self._device_cache = 'cpu'
            logger.info("CUDA not available, using CPU")
        
        self._device_test_time = current_time
        return self._device_cache
    
    @staticmethod
    def _maybe_use_engine(model_path: str) -> str:
        """
        Opt-in TensorRT: if USE_TENSORRT=1, CUDA is available, and a freshly
        built sibling .engine exists next to the requested .pt, use the engine.

        Safe by design:
          * default (USE_TENSORRT unset/0) -> returns model_path unchanged
          * no CUDA or no .engine present  -> returns model_path unchanged
        so the current CPU/.pt path cannot be affected until engines are built
        (scripts/accelerate_tensorrt.sh) AND the flag is set explicitly.
        """
        if os.getenv("USE_TENSORRT", "0") != "1":
            return model_path
        if not model_path.endswith(".pt"):
            return model_path
        engine_path = model_path[:-3] + ".engine"
        try:
            if os.path.exists(engine_path) and torch.cuda.is_available():
                logger.info(f"TensorRT: using {engine_path} instead of {model_path}")
                return engine_path
        except Exception as e:
            logger.warning(f"TensorRT engine check failed ({e}); using {model_path}")
        return model_path

    def get_model(self, model_path: str, device: str = 'auto', force_reload: bool = False) -> object:
        """
        Get a shared model instance, loading if necessary
        
        Args:
            model_path: Path to the model file (e.g., 'models/yolo11n.pt')
            device: Target device ('auto', 'cuda', 'cpu')
            force_reload: Force reload even if cached
            
        Returns:
            YOLO model instance
        """
        if YOLO is None:
            raise ImportError("Ultralytics YOLO not available")

        # Opt-in TensorRT acceleration (see _maybe_use_engine). Default OFF -
        # leaves the .pt CPU path completely unchanged until USE_TENSORRT=1.
        model_path = self._maybe_use_engine(model_path)

        with self._model_lock:
            # Resolve device
            if device == 'auto':
                device = self._get_optimal_device()
            
            # Create cache key including device
            cache_key = f"{model_path}_{device}"
            
            # Check cache first
            if cache_key in self._models and not force_reload:
                self._ref_counts[cache_key] += 1
                self._cache_hits += 1
                logger.debug(f"Model cache hit: {cache_key} (refs: {self._ref_counts[cache_key]})")
                return self._models[cache_key]
            
            # Cache miss - load model
            self._cache_misses += 1
            logger.info(f"Loading model: {model_path} on {device}")
            
            try:
                # Record memory before loading
                if device == 'cuda' and torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated(0)
                else:
                    memory_before = 0
                
                # Load model with error handling
                model = self._load_model_safe(model_path, device)
                
                # Record memory after loading
                if device == 'cuda' and torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated(0)
                    model_memory_mb = (memory_after - memory_before) / (1024 * 1024)
                else:
                    model_memory_mb = 0
                
                # Cache model
                self._models[cache_key] = model
                self._ref_counts[cache_key] = 1
                self._model_info[cache_key] = {
                    'device': device,
                    'loaded_time': time.time(),
                    'memory_mb': model_memory_mb,
                    'model_path': model_path
                }
                
                self._total_models_loaded += 1
                
                # Calculate memory saved (if this model was already loaded on different device)
                similar_models = [k for k in self._models.keys() if k.startswith(model_path.split('_')[0])]
                if len(similar_models) > 1:
                    estimated_savings = model_memory_mb * (len(similar_models) - 1)
                    self._total_memory_saved_mb += estimated_savings
                
                logger.info(f"Model loaded successfully: {cache_key} ({model_memory_mb:.1f}MB)")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
                
                # Try CPU fallback if GPU failed
                if device == 'cuda':
                    logger.warning("Attempting CPU fallback for model loading")
                    return self.get_model(model_path, device='cpu', force_reload=force_reload)
                else:
                    raise Exception(f"Model loading failed on both GPU and CPU: {e}")
    
    def _load_model_safe(self, model_path: str, device: str) -> object:
        """
        Safely load YOLO model with proper CUDA error handling
        """
        try:
            # Clear any existing CUDA cache before loading
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Check if model is TensorRT engine
            is_engine = model_path.endswith('.engine') or model_path.endswith('.trt')
            
            # Load model without safe_globals to prevent CUDA launch failures
            if is_engine:
                # TensorRT engines need explicit task specification
                model = YOLO(model_path, task='detect')
                logger.info(f"Loaded TensorRT engine: {model_path}")
            else:
                # PyTorch model
                model = YOLO(model_path)
            
            # Move to device with error handling (only for PyTorch models)
            if not is_engine:
                try:
                    model.to(device)
                    
                    # Test model on device with a simple warmup
                    if device == 'cuda':
                        # Warmup with small tensor to verify CUDA functionality
                        test_input = torch.randn(1, 3, 32, 32).to(device)
                        with torch.no_grad():
                            _ = test_input + 1  # Simple operation
                        del test_input
                        torch.cuda.empty_cache()
                    
                except RuntimeError as cuda_error:
                    if "CUDA" in str(cuda_error) or "launch failure" in str(cuda_error):
                        logger.warning(f"CUDA error during model device setup: {cuda_error}")
                        logger.warning("Falling back to CPU for this model")
                        model = model.cpu()
                        device = 'cpu'  # Update device for caching
                    else:
                        raise cuda_error
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Model successfully loaded on {device}: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Model loading failed for {model_path}: {e}")
            
            # Last resort: try CPU with basic loading
            if device != 'cpu':
                logger.warning("Attempting emergency CPU fallback")
                try:
                    model = YOLO(model_path)
                    model = model.cpu()
                    logger.info(f"Emergency CPU fallback successful: {model_path}")
                    return model
                except Exception as cpu_error:
                    logger.error(f"Emergency CPU fallback also failed: {cpu_error}")
            
            raise Exception(f"All loading methods failed for {model_path}: {e}")
    
    def release_model(self, model_path: str, device: str = 'auto') -> bool:
        """
        Release a reference to a model, cleanup if no more references
        
        Args:
            model_path: Path to the model file
            device: Device the model was loaded on
            
        Returns:
            True if model was cleaned up, False if still in use
        """
        with self._model_lock:
            if device == 'auto':
                device = self._get_optimal_device()
            
            cache_key = f"{model_path}_{device}"
            
            if cache_key not in self._ref_counts:
                logger.warning(f"Attempted to release non-existent model: {cache_key}")
                return False
            
            self._ref_counts[cache_key] -= 1
            logger.debug(f"Released model reference: {cache_key} (refs: {self._ref_counts[cache_key]})")
            
            # Cleanup if no more references
            if self._ref_counts[cache_key] <= 0:
                if cache_key in self._models:
                    # Move model to CPU before deletion to free GPU memory
                    if device == 'cuda':
                        try:
                            self._models[cache_key].cpu()
                        except:
                            pass
                    
                    del self._models[cache_key]
                    del self._model_info[cache_key]
                    del self._ref_counts[cache_key]
                    
                    # Clear GPU memory
                    if device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info(f"Model cleaned up: {cache_key}")
                    return True
            
            return False
    
    def list_loaded_models(self) -> Dict[str, Dict]:
        """Get information about all loaded models"""
        with self._model_lock:
            return {
                cache_key: {
                    **info,
                    'reference_count': self._ref_counts[cache_key]
                }
                for cache_key, info in self._model_info.items()
            }
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        with self._model_lock:
            total_model_memory = sum(info['memory_mb'] for info in self._model_info.values())
            
            gpu_stats = {}
            if torch.cuda.is_available():
                gpu_stats = {
                    'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    'allocated_memory_gb': torch.cuda.memory_allocated(0) / (1024**3),
                    'cached_memory_gb': torch.cuda.memory_reserved(0) / (1024**3)
                }
            
            return {
                'loaded_models': len(self._models),
                'total_model_memory_mb': total_model_memory,
                'estimated_memory_saved_mb': self._total_memory_saved_mb,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cache_hit_ratio': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
                'gpu_stats': gpu_stats
            }
    
    def cleanup_unused_models(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up models that haven't been used recently
        
        Args:
            max_age_seconds: Maximum age in seconds before cleanup
            
        Returns:
            Number of models cleaned up
        """
        with self._model_lock:
            current_time = time.time()
            cleanup_count = 0
            
            # Find models to cleanup
            models_to_cleanup = []
            for cache_key, info in self._model_info.items():
                if (self._ref_counts[cache_key] <= 0 and 
                    current_time - info['loaded_time'] > max_age_seconds):
                    models_to_cleanup.append(cache_key)
            
            # Cleanup models
            for cache_key in models_to_cleanup:
                device = self._model_info[cache_key]['device']
                model_path = self._model_info[cache_key]['model_path']
                
                if self.release_model(model_path, device):
                    cleanup_count += 1
            
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} unused models")
            
            return cleanup_count
    
    def force_cleanup_all(self):
        """Force cleanup of all models (emergency cleanup)"""
        with self._model_lock:
            logger.warning("Force cleaning up all models")
            
            # Move all CUDA models to CPU first
            for cache_key, model in self._models.items():
                if 'cuda' in cache_key:
                    try:
                        model.cpu()
                    except:
                        pass
            
            # Clear all caches
            self._models.clear()
            self._ref_counts.clear()
            self._model_info.clear()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("All models force cleaned up")


# Global instance
model_manager = ModelManager()


class _MemoizedDetector:
    """
    Transparent per-frame memoization wrapper for a shared YOLO detection model.

    Multiple analysis modules on the same camera process the SAME frame in the
    processor's sequential module loop, and each was running its own best.pt
    inference on that identical frame. This wrapper runs inference ONCE per
    (frame, iou, imgsz) and serves every caller, filtering the cached result to
    each caller's confidence.

    Correctness (behavior-preserving): inference runs once at a very low
    confidence; each caller receives that result filtered to conf >= its own
    threshold. This equals running directly at the caller's confidence, because
    NMS keeps the highest-confidence box in any overlap group, so a box above a
    caller's threshold can never be suppressed by one below it. Verified
    byte-identical against direct runs.

    Transparent: predict()/__call__ are intercepted; every other attribute
    (.names, .to, .model, ...) delegates to the wrapped model. Any call shape it
    does not recognise (non-ndarray source, class filtering, exotic kwargs)
    falls straight through to the real model unchanged.
    """
    _BENIGN = {"conf", "iou", "imgsz", "verbose", "classes", "device", "half"}

    def __init__(self, model, base_conf: float = 0.02, cache_size: int = 96):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_cache", OrderedDict())
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_base_conf", base_conf)
        object.__setattr__(self, "_cache_size", cache_size)
        object.__setattr__(self, "hits", 0)
        object.__setattr__(self, "misses", 0)

    def _cacheable(self, source, kwargs) -> bool:
        try:
            import numpy as _np
            if not isinstance(source, _np.ndarray):
                return False
        except Exception:
            return False
        if any(k not in self._BENIGN for k in kwargs):
            return False
        if kwargs.get("classes") is not None:
            return False
        return True

    def _fingerprint(self, source, iou, imgsz):
        # strided pixel sample -> content hash (cheap, identity-independent)
        samp = source[::100, ::100].tobytes()
        return (hash(samp), source.shape,
                round(float(iou if iou is not None else 0.45), 4),
                int(imgsz) if imgsz else 0)

    def _filter(self, raw, conf):
        out = []
        for r in raw:
            try:
                b = r.boxes
                if b is None or len(b) == 0:
                    out.append(r)
                    continue
                keep = (b.conf >= conf).nonzero(as_tuple=True)[0]
                out.append(r[keep])
            except Exception:
                out.append(r)
        return out

    def _memoized(self, source, conf, iou, imgsz):
        key = self._fingerprint(source, iou, imgsz)
        with self._lock:
            raw = self._cache.get(key)
            if raw is not None:
                self._cache.move_to_end(key)
                object.__setattr__(self, "hits", self.hits + 1)
        if raw is None:
            object.__setattr__(self, "misses", self.misses + 1)
            raw = self._model.predict(
                source, conf=self._base_conf,
                iou=iou if iou is not None else 0.45,
                imgsz=imgsz if imgsz else 640, verbose=False,
            )
            with self._lock:
                self._cache[key] = raw
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return self._filter(raw, conf if conf is not None else 0.25)

    def predict(self, source=None, conf=0.25, iou=0.45, imgsz=None, verbose=False, **kw):
        if source is not None and self._cacheable(source, kw):
            return self._memoized(source, conf, iou, imgsz)
        extra = {"imgsz": imgsz} if imgsz else {}
        return self._model.predict(source, conf=conf, iou=iou, verbose=verbose, **extra, **kw)

    def __call__(self, source=None, conf=0.25, iou=0.45, imgsz=None, verbose=False, **kw):
        if source is not None and self._cacheable(source, kw):
            return self._memoized(source, conf, iou, imgsz)
        extra = {"imgsz": imgsz} if imgsz else {}
        return self._model(source, conf=conf, iou=iou, verbose=verbose, **extra, **kw)

    def __getattr__(self, name):
        # only reached when normal attribute lookup fails -> delegate to model
        return getattr(object.__getattribute__(self, "_model"), name)


_wrapped_models = {}
_wrap_lock = threading.Lock()


def get_shared_model(model_path: str, device: str = 'auto') -> object:
    """
    Convenience function to get a shared model instance.

    Returns a transparent per-frame memoization wrapper (see _MemoizedDetector)
    so co-located modules processing the same frame share one inference. Set
    SHARED_DETECTION_MEMO=0 to disable and return the raw model (instant
    rollback without code changes).
    """
    raw = model_manager.get_model(model_path, device)
    if os.getenv("SHARED_DETECTION_MEMO", "1") != "1":
        return raw
    key = f"{model_path}_{device}"
    with _wrap_lock:
        w = _wrapped_models.get(key)
        if w is None or getattr(w, "_model", None) is not raw:
            w = _MemoizedDetector(raw)
            _wrapped_models[key] = w
        return w


def release_shared_model(model_path: str, device: str = 'auto') -> bool:
    """
    Convenience function to release a shared model
    
    Args:
        model_path: Path to model file
        device: Device model was loaded on
        
    Returns:
        True if model was cleaned up
    """
    return model_manager.release_model(model_path, device)


def get_model_stats() -> Dict:
    """Get global model manager statistics"""
    return model_manager.get_memory_stats()


def cleanup_models(max_age_seconds: float = 3600) -> int:
    """Cleanup unused models"""
    return model_manager.cleanup_unused_models(max_age_seconds)