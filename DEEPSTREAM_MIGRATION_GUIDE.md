# SAKSHI.AI — DeepStream Migration Guide

## Target Server Specifications

| Component | Details |
|-----------|---------|
| **OS** | Ubuntu 24.04.4 LTS (noble) |
| **Kernel** | 6.17.0-14-generic |
| **CPU** | Intel i5-12400F (6C/12T, up to 5.6 GHz) |
| **RAM** | 32 GB DDR5 |
| **GPU** | NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) |
| **CUDA** | 13.0 |
| **Driver** | 580.126.09 |
| **Storage** | ~853 GB free on NVMe |

---

## Table of Contents

1. [Why DeepStream?](#1-why-deepstream)
2. [Architecture Comparison: Current vs DeepStream](#2-architecture-comparison)
3. [Compatibility Assessment & Prerequisites](#3-compatibility-assessment)
4. [Step-by-Step Installation](#4-step-by-step-installation)
5. [DeepStream Pipeline Design](#5-deepstream-pipeline-design)
6. [Code Migration Plan](#6-code-migration-plan)
7. [Module-by-Module Migration](#7-module-by-module-migration)
8. [Configuration Migration](#8-configuration-migration)
9. [TensorRT Model Conversion](#9-tensorrt-model-conversion)
10. [Testing & Validation](#10-testing--validation)
11. [Performance Tuning](#11-performance-tuning)
12. [Rollback Plan](#12-rollback-plan)

---

## 1. Why DeepStream?

### Current Architecture Bottlenecks

Your current pipeline does:
```
RTSP → OpenCV (CPU decode) → Python frame loop → Ultralytics YOLO (GPU infer) → Python post-process
```

**Problems:**
- **CPU-bound decoding**: OpenCV uses FFmpeg on CPU to decode H.264/H.265 — this bottlenecks at ~6-8 1080p streams on your i5-12400F
- **Python GIL**: Frame capture + inference + post-processing all fight for the GIL
- **Memory copies**: Frame data bounces CPU→GPU→CPU→GPU (decode on CPU, upload to GPU for YOLO, download results to CPU, re-upload for next frame)
- **No batching at decode level**: Each camera has its own decode thread — no hardware multiplexing
- **Ultralytics overhead**: High-level API adds ~2-5ms per frame in Python wrapper overhead

### What DeepStream Gives You

```
RTSP → NVDEC (GPU decode) → nvstreammux (GPU batch) → nvinfer (TensorRT) → Python probe (metadata only)
```

**Gains:**
| Metric | Current (OpenCV) | DeepStream | Improvement |
|--------|-----------------|------------|-------------|
| Decode | CPU (FFmpeg) | GPU (NVDEC) | ~5x faster, frees CPU |
| Streams at 1080p/15fps | ~6-8 | ~16-20+ | 2-3x more streams |
| Inference | Ultralytics wrapper | Native TensorRT | ~30-40% faster |
| Memory copies | 3-4 per frame | 0 (stays on GPU) | Near-zero copy |
| Batched inference | Manual (4 frames) | Automatic (configurable) | Higher throughput |
| CPU usage | 60-80% | 15-25% | Frees CPU for logic |

Your RTX 5060 Ti has **dedicated NVDEC hardware** — it decodes video streams with zero impact on CUDA cores. This is essentially "free" decoding.

---

## 2. Architecture Comparison

### Current Architecture
```
┌──────────────────────────────────────────────────────┐
│                    Flask App (app.py)                  │
│                                                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  Camera 1  │  │  Camera 2  │  │  Camera N  │       │
│  │  Thread    │  │  Thread    │  │  Thread    │       │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘       │
│        │               │               │               │
│  ┌─────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐       │
│  │  OpenCV    │  │  OpenCV    │  │  OpenCV    │       │
│  │  Decode    │  │  Decode    │  │  Decode    │       │
│  │  (CPU)     │  │  (CPU)     │  │  (CPU)     │       │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘       │
│        │               │               │               │
│  ┌─────▼──────────────────────────────────────┐        │
│  │        MultiModuleVideoProcessor           │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │        │
│  │  │  YOLO    │ │  Module  │ │  Module  │   │        │
│  │  │ (shared) │ │  Queue   │ │  Dress   │   │        │
│  │  └──────────┘ └──────────┘ └──────────┘   │        │
│  └────────────────────────────────────────────┘        │
│                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────┐      │
│  │  Database   │  │  SocketIO   │  │  Telegram  │      │
│  │  (Postgres) │  │  (Alerts)   │  │  Notifier  │      │
│  └─────────────┘  └─────────────┘  └───────────┘      │
└──────────────────────────────────────────────────────┘
```

### Target DeepStream Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                       Flask App (app.py)                      │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              DeepStream Pipeline (C/GStreamer)         │      │
│  │                                                        │      │
│  │  ┌──────┐ ┌──────┐ ┌──────┐                           │      │
│  │  │RTSP 1│ │RTSP 2│ │RTSP N│  (rtspsrc elements)      │      │
│  │  └──┬───┘ └──┬───┘ └──┬───┘                           │      │
│  │     │        │        │                                │      │
│  │  ┌──▼────────▼────────▼──┐                             │      │
│  │  │    NVDEC (GPU decode) │  Hardware decoder            │      │
│  │  └──────────┬────────────┘                             │      │
│  │             │                                          │      │
│  │  ┌──────────▼────────────┐                             │      │
│  │  │  nvstreammux (batch)  │  Batches frames from all    │      │
│  │  │  batch-size=N         │  streams into single tensor │      │
│  │  └──────────┬────────────┘                             │      │
│  │             │                                          │      │
│  │  ┌──────────▼────────────┐                             │      │
│  │  │  nvinfer (primary)    │  YOLO TensorRT engine       │      │
│  │  │  (Person detection)   │  Runs on batched frames     │      │
│  │  └──────────┬────────────┘                             │      │
│  │             │                                          │      │
│  │  ┌──────────▼────────────┐                             │      │
│  │  │  nvinfer (secondary)  │  Specialized models         │      │
│  │  │  (Cash/PPE/Dress)     │  (optional per stream)      │      │
│  │  └──────────┬────────────┘                             │      │
│  │             │                                          │      │
│  │  ┌──────────▼────────────┐                             │      │
│  │  │  nvtracker            │  Hardware-accelerated       │      │
│  │  │  (DeepSORT/NvDCF)     │  multi-object tracking      │      │
│  │  └──────────┬────────────┘                             │      │
│  │             │                                          │      │
│  │  ┌──────────▼────────────┐     ┌──────────────────┐   │      │
│  │  │  Python Probe         │────▶│ Detection Modules │   │      │
│  │  │  (buffer probe)       │     │ (Queue, Dress,    │   │      │
│  │  │                       │     │  Fall, Cash, etc.) │   │      │
│  │  └──────────┬────────────┘     └──────────────────┘   │      │
│  │             │                                          │      │
│  │  ┌──────────▼──────┐  ┌───────────────┐               │      │
│  │  │  nvvideoconvert │  │  appsink      │               │      │
│  │  │  + nvdsosd      │  │  (frame grab) │               │      │
│  │  └─────────────────┘  └───────────────┘               │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐          │
│  │  Database   │  │  SocketIO   │  │  Telegram     │          │
│  │  (Postgres) │  │  (Alerts)   │  │  Notifier     │          │
│  └─────────────┘  └─────────────┘  └───────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

### Key Differences

| Aspect | Current | DeepStream |
|--------|---------|------------|
| Decode | Per-camera OpenCV thread (CPU) | Single NVDEC pipeline (GPU) |
| Batching | Manual in batch_processor.py | nvstreammux handles automatically |
| Inference | Ultralytics YOLO wrapper | nvinfer with native TensorRT |
| Tracking | deep-sort-realtime (Python) | nvtracker (GPU-accelerated NvDCF) |
| Frame flow | Python numpy arrays | GstBuffer (GPU memory) |
| Post-process | Full frame access in Python | Metadata probes (efficient) |
| OSD/Annotation | OpenCV cv2.rectangle/putText | nvdsosd (GPU-rendered) |

---

## 3. Compatibility Assessment

### CRITICAL: CUDA 13.0 + RTX 5060 Ti Compatibility

> **⚠️ Important**: Your RTX 5060 Ti is a **Blackwell-architecture** GPU with **CUDA 13.0**. As of March 2026, you need to verify DeepStream SDK compatibility.

**Check the latest DeepStream release:**

```bash
# Check NVIDIA's DeepStream compatibility matrix
# https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_supported_platforms.html

# DeepStream 7.1+ should support CUDA 13.0 and Blackwell GPUs
# Verify before proceeding:
apt-cache search deepstream
```

### Required Software Stack

| Component | Required Version | Notes |
|-----------|-----------------|-------|
| **NVIDIA Driver** | 580.x ✅ (already installed) | Compatible |
| **CUDA Toolkit** | 13.0 ✅ (already installed via driver) | Need dev tools |
| **cuDNN** | 9.x+ (match CUDA 13.0) | Required for nvinfer |
| **TensorRT** | 10.x+ (match CUDA 13.0) | Core inference engine |
| **GStreamer** | 1.24+ | Ubuntu 24.04 ships 1.24 |
| **DeepStream SDK** | 7.1+ | Must support CUDA 13.0 |
| **Python bindings** | pyds (from DeepStream) | Python metadata access |
| **librdkafka** | Optional | For message broker |

### Pre-Installation Verification

Run these commands on your Ubuntu server to verify readiness:

```bash
# 1. Verify NVIDIA driver and CUDA
nvidia-smi
nvcc --version  # If CUDA toolkit is installed

# 2. Check GStreamer
gst-inspect-1.0 --version
gst-inspect-1.0 nvv4l2decoder  # Should show NVDEC plugin

# 3. Check available NVDEC sessions
nvidia-smi dmon -s u  # Monitor decoder utilization

# 4. Check if GStreamer NVIDIA plugins are available
gst-inspect-1.0 | grep nv

# 5. Python version (DeepStream 7.x needs Python 3.10-3.12)
python3 --version
```

---

## 4. Step-by-Step Installation

### Phase 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    unzip \
    software-properties-common

# Install GStreamer and plugins
sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-rtsp \
    libgstrtspserver-1.0-0 \
    libgstrtspserver-1.0-dev

# Install Python GStreamer bindings
sudo apt install -y \
    python3-gi \
    python3-gst-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gstreamer-1.0

# Install multimedia libraries
sudo apt install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev

# Install JSON/config libraries
sudo apt install -y \
    libjson-glib-dev \
    libssl-dev \
    libcurl4-openssl-dev
```

### Phase 2: CUDA Toolkit Installation

```bash
# Check if CUDA toolkit (dev tools) is installed
nvcc --version

# If not installed, install CUDA toolkit matching your driver
# For CUDA 13.0 on Ubuntu 24.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0

# Set environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### Phase 3: cuDNN Installation

```bash
# Install cuDNN (match CUDA 13.0)
# Download from: https://developer.nvidia.com/cudnn-downloads
# Select: Linux > x86_64 > Ubuntu > 24.04 > deb (network)

sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13

# Verify
dpkg -l | grep cudnn
```

### Phase 4: TensorRT Installation

```bash
# Install TensorRT (match CUDA 13.0)
# Download from: https://developer.nvidia.com/tensorrt

sudo apt install -y \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvparsers-dev \
    libnvonnxparsers-dev \
    python3-libnvinfer

# Or install via pip for Python bindings
pip3 install tensorrt

# Verify
python3 -c "import tensorrt; print(tensorrt.__version__)"
dpkg -l | grep nvinfer
```

### Phase 5: DeepStream SDK Installation

```bash
# Option A: APT package (recommended for Ubuntu 24.04)
sudo apt install -y deepstream-7.1  # or latest available version

# Option B: Download .deb from NVIDIA
# https://developer.nvidia.com/deepstream-getting-started
# Download the deb package for your platform
wget <deepstream-deb-url>  # Get URL from NVIDIA developer portal
sudo apt install -y ./deepstream-*.deb

# Option C: tar.gz manual install
# Download tar from NVIDIA
tar -xvf deepstream_sdk_*.tbz2 -C /opt/nvidia/deepstream/
cd /opt/nvidia/deepstream/deepstream-7.1/
sudo ./install.sh

# Post-installation
sudo ldconfig

# Verify DeepStream installation
deepstream-app --version-all

# Check DeepStream GStreamer plugins
gst-inspect-1.0 nvinfer
gst-inspect-1.0 nvstreammux
gst-inspect-1.0 nvvideoconvert
gst-inspect-1.0 nvdsosd
gst-inspect-1.0 nvtracker
```

### Phase 6: DeepStream Python Bindings (pyds)

```bash
# The Python bindings are critical for your application
# They should come with the DeepStream SDK

# Check if pyds is available
python3 -c "import pyds; print('pyds version:', pyds.__version__)"

# If not available, build from source:
cd /opt/nvidia/deepstream/deepstream-7.1/sources/deepstream_python_apps/
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
cd deepstream_python_apps/bindings/
mkdir build && cd build
cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=12
make -j$(nproc)
pip3 install ./pyds-*.whl

# Verify
python3 -c "import pyds; print('pyds loaded successfully')"
```

### Phase 7: Python Environment Setup

```bash
# Create virtual environment on the Ubuntu server
cd /path/to/sakshiai
python3 -m venv .venv
source .venv/bin/activate

# Install core Python dependencies
pip install flask==2.3.3
pip install flask-socketio==5.3.6
pip install flask-sqlalchemy==3.1.1
pip install sqlalchemy==2.0.23
pip install psycopg2-binary==2.9.7
pip install eventlet==0.33.3
pip install Pillow
pip install requests

# Install AI/ML dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics>=8.0.0
pip install opencv-python==4.8.1.78
pip install deep-sort-realtime==1.3.2
pip install numpy

# Make pyds available in venv (symlink from system)
PYDS_PATH=$(python3 -c "import pyds; print(pyds.__file__)" 2>/dev/null)
if [ -n "$PYDS_PATH" ]; then
    VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    ln -sf $(dirname $PYDS_PATH)/pyds* $VENV_SITE/
fi

# Also make gi (GObject Introspection) available in venv
GI_PATH=$(python3 -c "import gi; print(gi.__path__[0])" 2>/dev/null)
if [ -n "$GI_PATH" ]; then
    ln -sf $GI_PATH $VENV_SITE/gi
fi
```

---

## 5. DeepStream Pipeline Design

### 5.1 Pipeline for Your Application

Your application needs a pipeline that:
1. Ingests multiple RTSP streams (currently 4 cameras, expandable to 16+)
2. Decodes with NVDEC hardware
3. Batches frames from all streams
4. Runs primary YOLO inference (person detection — shared across all modules)
5. Optionally runs secondary inference (cash, PPE, dress code models)
6. Provides Python access to detection metadata AND raw frames (for GIF recording)
7. Supports per-stream ROI filtering

### 5.2 GStreamer Pipeline Diagram

```
Camera 1 ─┐
Camera 2 ─┤─► nvstreammux ─► nvinfer ─► nvtracker ─► tee
Camera 3 ─┤       │          (YOLO)     (NvDCF)      │
Camera 4 ─┘       │                                    ├─► probe (Python metadata callback)
                   │                                    │
                   │                                    ├─► nvvideoconvert ─► appsink
                   │                                    │   (for frame grab + GIF recording)
                   │                                    │
                   │                                    └─► nvdsosd ─► fakesink
                   │                                        (optional display)
```

### 5.3 nvinfer Configuration Files

You need configuration files for each TensorRT model used in the pipeline.

**Primary Detector — YOLO (Person/Object detection):**

Create `config/ds_yolo_primary.txt`:
```ini
[property]
gpu-id=0
net-scale-factor=0.00392156862   # 1/255
model-engine-file=models/yolo11n.engine
# If engine doesn't exist, build from ONNX:
# onnx-file=models/yolo11n.onnx
labelfile-path=config/labels.txt
batch-size=4
network-mode=2                    # 0=FP32, 1=FP16, 2=INT8 (if calibrated)
num-detected-classes=80
interval=0                        # Infer every frame (0 = every frame, 1 = every 2nd, etc.)
process-mode=1                    # 1=primary detector
network-type=0                    # 0=detector, 1=classifier, 2=segmentation
cluster-mode=2                    # 2=NMS
maintain-aspect-ratio=1
symmetric-padding=1

# Input dimensions (must match your YOLO model)
infer-dims=3;640;640

# Output parsing for YOLO
parse-bbox-func-name=NvDsInferParseCustomYolo
custom-lib-path=lib/libnvdsinfer_custom_yolo.so
output-blob-names=output0

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

**Secondary Detector — Cash Detection:**

Create `config/ds_cash_secondary.txt`:
```ini
[property]
gpu-id=0
net-scale-factor=0.00392156862
model-engine-file=models/cash.engine
labelfile-path=config/cash_labels.txt
batch-size=4
network-mode=1                    # FP16
num-detected-classes=2            # cash, no_cash (adjust to your model)
interval=2                        # Every 3rd frame (less frequent for secondary)
process-mode=2                    # 2=secondary detector
operate-on-class-ids=0            # Operate on person detections from primary
network-type=0

infer-dims=3;640;640

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.5
```

**Tracker Configuration:**

Create `config/ds_tracker.txt`:
```ini
[tracker]
tracker-width=640
tracker-height=384
gpu-id=0
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=config/ds_tracker_NvDCF.yml
enable-batch-process=1
enable-past-frame=1
display-tracking-id=1
```

Create `config/ds_tracker_NvDCF.yml`:
```yaml
%YAML:1.0

NvDCF:
  # Anthropic note: this is a standard NvDCF tracker config
  maxTargetsPerStream: 99
  filterLr: 0.11
  gaussianSigma: 0.75
  minDetectorConfidence: 0.2
  minTrackerConfidence: 0.5
  minTargetHits: 3
  minFramesBeforeInactive: 30
  featureImgSizeLevel: 3
  SearchRegionPaddingScale: 3
  maxShadowTrackingAge: 30
  useUniqueID: 0
```

### 5.4 nvstreammux Configuration

```ini
# Key parameters for nvstreammux
batch-size=4              # Number of cameras (match your camera count)
batched-push-timeout=40000  # Timeout in microseconds (40ms for ~25fps)
width=1280                # Muxer output width
height=720                # Muxer output height
enable-padding=1          # Maintain aspect ratio
gpu-id=0
nvbuf-memory-type=0       # 0=default, 1=cuda-pinned, 2=cuda-device, 3=cuda-unified
```

---

## 6. Code Migration Plan

### 6.1 What Changes vs What Stays the Same

| Component | Action | Details |
|-----------|--------|---------|
| **app.py** | MODIFY | Replace per-camera OpenCV threads with single DeepStream pipeline |
| **modules/deepstream_processor.py** | REWRITE | New robust DeepStream pipeline manager |
| **modules/multi_module_processor.py** | MODIFY | Receives metadata from DS probes instead of running YOLO |
| **modules/model_manager.py** | KEEP (reduced role) | Still manages any secondary Python-based models |
| **modules/batch_processor.py** | REMOVE | nvstreammux handles batching |
| **modules/rtsp_connection_pool.py** | REMOVE | DeepStream handles RTSP connections |
| **modules/video_processor.py** | KEEP (fallback) | Fallback for non-RTSP sources |
| **modules/queue_monitor.py** | MODIFY (minor) | Accept DS metadata instead of raw YOLO results |
| **modules/dress_code_monitoring.py** | MODIFY (minor) | Same — metadata adapter |
| **modules/fall_detection.py** | MODIFY (minor) | Uses nvtracker IDs instead of custom tracking |
| **modules/service_discipline_monitor.py** | MODIFY (minor) | Replace deep-sort with nvtracker |
| **All other detection modules** | MODIFY (minor) | Accept detection metadata from DS probe |
| **modules/database.py** | KEEP | No changes needed |
| **modules/telegram_notifier.py** | KEEP | No changes needed |
| **modules/gif_recorder.py** | MODIFY (minor) | Get frames from appsink instead of OpenCV |
| **config/channels.json** | KEEP | Same structure, pipeline reads it |
| **config/default.json** | ADD fields | Add DeepStream-specific settings |
| **templates/** | KEEP | No changes needed |
| **static/** | KEEP | No changes needed |

### 6.2 New Files to Create

| File | Purpose |
|------|---------|
| `modules/ds_pipeline.py` | Main DeepStream pipeline manager (replaces deepstream_processor.py) |
| `modules/ds_probe_handler.py` | Python buffer probe callbacks for metadata extraction |
| `modules/ds_frame_grabber.py` | Extracts numpy frames from GstBuffer for GIF recording + MJPEG feed |
| `modules/ds_module_adapter.py` | Adapter layer so existing modules work with DS metadata |
| `config/ds_yolo_primary.txt` | nvinfer primary detector config |
| `config/ds_cash_secondary.txt` | nvinfer secondary detector config (optional) |
| `config/ds_tracker.txt` | nvtracker config |
| `config/ds_tracker_NvDCF.yml` | NvDCF tracker parameters |
| `lib/libnvdsinfer_custom_yolo.so` | Custom YOLO output parser (compiled C++) |
| `scripts/convert_yolo_to_onnx.py` | Convert YOLO .pt models to ONNX for TensorRT |
| `scripts/build_yolo_parser.sh` | Build custom YOLO parser library |

### 6.3 Implementation: Core DeepStream Pipeline

Here is the complete implementation of the new DeepStream pipeline manager:

```python
# modules/ds_pipeline.py
"""
DeepStream Pipeline Manager for Sakshi.AI
Replaces OpenCV-based per-camera processing with a single GPU-accelerated pipeline.
"""

import sys
import os
import logging
import threading
import time
import json
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import defaultdict

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError:
    PYDS_AVAILABLE = False
    print("WARNING: pyds not available. DeepStream pipeline will not work.")

logger = logging.getLogger(__name__)

# DeepStream metadata constants
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


class DeepStreamPipeline:
    """
    Manages a single DeepStream pipeline that processes multiple RTSP streams.
    
    Features:
    - Multi-stream NVDEC hardware decoding
    - Batched TensorRT inference (nvinfer)
    - GPU-accelerated tracking (nvtracker)
    - Python probe callbacks for detection metadata
    - Frame extraction for GIF recording and MJPEG streaming
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Application config dict including:
                - channels: List of channel configs from channels.json
                - models: Model paths for primary/secondary detectors
                - pipeline: Pipeline parameters (batch_size, resolution, etc.)
        """
        if not PYDS_AVAILABLE:
            raise RuntimeError("pyds not installed. Cannot create DeepStream pipeline.")
        
        Gst.init(None)
        
        self.config = config
        self.pipeline = None
        self.loop = None
        self.loop_thread = None
        self.is_running = False
        
        # Stream management
        self.source_bins: Dict[int, Gst.Element] = {}  # stream_id -> source bin
        self.channel_map: Dict[int, str] = {}  # stream_id -> channel_id
        self.channel_to_stream: Dict[str, int] = {}  # channel_id -> stream_id
        
        # Callbacks
        self._detection_callbacks: List[Callable] = []
        self._frame_callbacks: List[Callable] = []
        
        # Frame storage for MJPEG streaming and GIF recording
        self._latest_frames: Dict[str, np.ndarray] = {}
        self._frame_lock = threading.Lock()
        
        # Detection metadata storage
        self._latest_detections: Dict[str, List[dict]] = defaultdict(list)
        self._detection_lock = threading.Lock()
        
        # Statistics
        self._fps_counters: Dict[str, dict] = defaultdict(lambda: {
            'frame_count': 0,
            'start_time': time.time(),
            'fps': 0.0
        })
        
        logger.info("DeepStreamPipeline initialized")
    
    def build_pipeline(self, channels: List[dict]) -> bool:
        """
        Build the complete DeepStream pipeline for all channels.
        
        Args:
            channels: List of channel configs from channels.json
            
        Returns:
            True if pipeline built successfully
        """
        try:
            self.pipeline = Gst.Pipeline()
            if not self.pipeline:
                logger.error("Failed to create Gst.Pipeline")
                return False
            
            # Create streammux
            streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
            if not streammux:
                logger.error("Failed to create nvstreammux")
                return False
            
            batch_size = len(channels)
            streammux.set_property('batch-size', batch_size)
            streammux.set_property('width', self.config.get('muxer_width', 1280))
            streammux.set_property('height', self.config.get('muxer_height', 720))
            streammux.set_property('batched-push-timeout', 40000)
            streammux.set_property('enable-padding', True)
            streammux.set_property('gpu-id', 0)
            streammux.set_property('nvbuf-memory-type', 0)
            streammux.set_property('live-source', 1)
            self.pipeline.add(streammux)
            
            # Create source bins for each camera
            for idx, channel in enumerate(channels):
                if not channel.get('enabled', True):
                    continue
                
                rtsp_url = channel.get('rtsp_url', '')
                channel_id = channel.get('channel_id', f'camera_{idx}')
                
                if not rtsp_url:
                    logger.warning(f"Skipping {channel_id}: no RTSP URL")
                    continue
                
                source_bin = self._create_source_bin(idx, rtsp_url, channel_id)
                if not source_bin:
                    logger.error(f"Failed to create source bin for {channel_id}")
                    continue
                
                self.pipeline.add(source_bin)
                
                # Link source bin to streammux
                srcpad = source_bin.get_static_pad("src")
                sinkpad = streammux.request_pad_simple(f"sink_{idx}")
                if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                    logger.error(f"Failed to link source {channel_id} to muxer")
                    continue
                
                self.source_bins[idx] = source_bin
                self.channel_map[idx] = channel_id
                self.channel_to_stream[channel_id] = idx
                
                logger.info(f"Added stream {idx}: {channel_id} ({rtsp_url})")
            
            if not self.source_bins:
                logger.error("No source bins created. Pipeline cannot run.")
                return False
            
            # Primary inference (YOLO)
            pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
            pgie.set_property('config-file-path', 'config/ds_yolo_primary.txt')
            pgie.set_property('batch-size', batch_size)
            self.pipeline.add(pgie)
            
            # Tracker
            tracker = Gst.ElementFactory.make("nvtracker", "tracker")
            tracker_config = self.config.get('tracker_config', 'config/ds_tracker.txt')
            if os.path.exists(tracker_config):
                # Parse tracker config file
                self._apply_tracker_config(tracker, tracker_config)
            else:
                # Default tracker settings
                tracker.set_property('tracker-width', 640)
                tracker.set_property('tracker-height', 384)
                tracker.set_property('gpu-id', 0)
                tracker.set_property('ll-lib-file',
                    '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
            self.pipeline.add(tracker)
            
            # Video converter (for frame extraction)
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
            nvvidconv.set_property('gpu-id', 0)
            self.pipeline.add(nvvidconv)
            
            # OSD (On-Screen Display) — renders bboxes on GPU
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            nvosd.set_property('process-mode', 1)  # GPU mode
            self.pipeline.add(nvosd)
            
            # Tee to split into: (1) frame grabber, (2) display/sink
            tee = Gst.ElementFactory.make("tee", "splitter")
            self.pipeline.add(tee)
            
            # Queue for probe path
            queue_probe = Gst.ElementFactory.make("queue", "queue-probe")
            self.pipeline.add(queue_probe)
            
            # Queue for frame grab path
            queue_frames = Gst.ElementFactory.make("queue", "queue-frames")
            self.pipeline.add(queue_frames)
            
            # Video convert for frame extraction (GPU → CPU)
            nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
            self.pipeline.add(nvvidconv2)
            
            # Caps filter to get BGR frames for OpenCV
            capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
            caps = Gst.Caps.from_string("video/x-raw, format=BGR")
            capsfilter.set_property('caps', caps)
            self.pipeline.add(capsfilter)
            
            # Appsink for frame extraction (GIF recording, MJPEG streaming)
            appsink = Gst.ElementFactory.make("appsink", "frame-sink")
            appsink.set_property('emit-signals', True)
            appsink.set_property('max-buffers', 2)
            appsink.set_property('drop', True)  # Drop old frames if not consumed
            appsink.connect("new-sample", self._on_new_sample)
            self.pipeline.add(appsink)
            
            # Fakesink for probe path (we only need the probe, not the output)
            fakesink = Gst.ElementFactory.make("fakesink", "fake-sink")
            self.pipeline.add(fakesink)
            
            # Link everything
            # streammux → pgie → tracker → nvvidconv → nvosd → tee
            if not streammux.link(pgie):
                logger.error("Failed to link streammux → pgie")
                return False
            if not pgie.link(tracker):
                logger.error("Failed to link pgie → tracker")
                return False
            if not tracker.link(nvvidconv):
                logger.error("Failed to link tracker → nvvidconv")
                return False
            if not nvvidconv.link(nvosd):
                logger.error("Failed to link nvvidconv → nvosd")
                return False
            if not nvosd.link(tee):
                logger.error("Failed to link nvosd → tee")
                return False
            
            # tee → queue_probe → fakesink (metadata probe path)
            tee_src1 = tee.request_pad_simple("src_%u")
            queue_probe_sink = queue_probe.get_static_pad("sink")
            tee_src1.link(queue_probe_sink)
            queue_probe.link(fakesink)
            
            # tee → queue_frames → nvvidconv2 → capsfilter → appsink (frame grab path)
            tee_src2 = tee.request_pad_simple("src_%u")
            queue_frames_sink = queue_frames.get_static_pad("sink")
            tee_src2.link(queue_frames_sink)
            queue_frames.link(nvvidconv2)
            nvvidconv2.link(capsfilter)
            capsfilter.link(appsink)
            
            # Add buffer probe on nvosd sink pad for metadata extraction
            osdsinkpad = nvosd.get_static_pad("sink")
            if osdsinkpad:
                osdsinkpad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self._osd_sink_pad_buffer_probe,
                    0
                )
            else:
                logger.error("Failed to get OSD sink pad for probe")
                return False
            
            logger.info(f"Pipeline built successfully with {len(self.source_bins)} streams")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline build failed: {e}", exc_info=True)
            return False
    
    def _create_source_bin(self, stream_id: int, rtsp_url: str, channel_id: str) -> Optional[Gst.Bin]:
        """Create a source bin for an RTSP stream with NVDEC hardware decoding."""
        bin_name = f"source-bin-{stream_id:02d}"
        source_bin = Gst.Bin.new(bin_name)
        
        # RTSP source
        rtspsrc = Gst.ElementFactory.make("rtspsrc", f"rtspsrc-{stream_id}")
        rtspsrc.set_property('location', rtsp_url)
        rtspsrc.set_property('latency', 200)
        rtspsrc.set_property('drop-on-latency', True)
        rtspsrc.set_property('protocols', 0x4)  # TCP only for reliability
        # Timeout: 10 seconds
        rtspsrc.set_property('timeout', 10000000)
        
        # Depayloader (will be linked dynamically)
        depay = Gst.ElementFactory.make("rtph264depay", f"depay-{stream_id}")
        
        # H.264 parser
        h264parse = Gst.ElementFactory.make("h264parse", f"parse-{stream_id}")
        
        # NVDEC hardware decoder
        decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{stream_id}")
        decoder.set_property('gpu-id', 0)
        # Enable low-latency mode
        decoder.set_property('enable-max-performance', True)
        
        # Add elements to bin
        for elem in [rtspsrc, depay, h264parse, decoder]:
            source_bin.add(elem)
        
        # Link static elements: depay → h264parse → decoder
        depay.link(h264parse)
        h264parse.link(decoder)
        
        # rtspsrc has dynamic pads — connect on pad-added
        rtspsrc.connect("pad-added", self._on_rtspsrc_pad_added, depay)
        
        # Create ghost pad from decoder src
        decoder_srcpad = decoder.get_static_pad("src")
        ghost_pad = Gst.GhostPad.new("src", decoder_srcpad)
        source_bin.add_pad(ghost_pad)
        
        return source_bin
    
    def _on_rtspsrc_pad_added(self, rtspsrc, pad, depay):
        """Callback when rtspsrc creates a new pad (dynamic linking)."""
        caps = pad.get_current_caps()
        if caps is None:
            return
        
        struct = caps.get_structure(0)
        media_type = struct.get_name()
        
        # Only link video pads (ignore audio)
        if media_type.startswith("application/x-rtp"):
            encoding = struct.get_string("encoding-name")
            if encoding and encoding.upper() in ("H264", "H265"):
                sink_pad = depay.get_static_pad("sink")
                if not sink_pad.is_linked():
                    pad.link(sink_pad)
                    logger.debug(f"Linked rtspsrc pad to depayloader ({encoding})")
    
    def _apply_tracker_config(self, tracker, config_path: str):
        """Parse tracker config file and apply properties."""
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or '=' not in line or line.startswith('['):
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        tracker.set_property(key, value)
                    except Exception:
                        pass  # Skip properties that don't apply
        except FileNotFoundError:
            logger.warning(f"Tracker config not found: {config_path}")
    
    def _osd_sink_pad_buffer_probe(self, pad, info, u_data):
        """
        Buffer probe callback — extracts detection metadata from each batch.
        This is the core integration point between DeepStream and your Python modules.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            stream_id = frame_meta.source_id
            channel_id = self.channel_map.get(stream_id, f"unknown_{stream_id}")
            frame_number = frame_meta.frame_num
            
            # Extract all detections for this frame
            detections = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                # Extract bounding box
                rect = obj_meta.rect_params
                detection = {
                    'class_id': obj_meta.class_id,
                    'class_name': obj_meta.obj_label,
                    'confidence': obj_meta.confidence,
                    'tracker_id': obj_meta.object_id if obj_meta.object_id != UNTRACKED_OBJECT_ID else -1,
                    'bbox': {
                        'x': rect.left,
                        'y': rect.top,
                        'w': rect.width,
                        'h': rect.height,
                    },
                    'frame_number': frame_number,
                    'stream_id': stream_id,
                    'channel_id': channel_id,
                }
                
                detections.append(detection)
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            # Update FPS counter
            fps_data = self._fps_counters[channel_id]
            fps_data['frame_count'] += 1
            elapsed = time.time() - fps_data['start_time']
            if elapsed >= 5.0:
                fps_data['fps'] = fps_data['frame_count'] / elapsed
                fps_data['frame_count'] = 0
                fps_data['start_time'] = time.time()
            
            # Store detections
            with self._detection_lock:
                self._latest_detections[channel_id] = detections
            
            # Fire detection callbacks
            for callback in self._detection_callbacks:
                try:
                    callback(channel_id, detections, frame_number)
                except Exception as e:
                    logger.error(f"Detection callback error: {e}")
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK
    
    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        """
        Appsink callback — extracts raw video frames as numpy arrays.
        Used for MJPEG streaming and GIF recording.
        """
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # Get frame dimensions
        struct = caps.get_structure(0)
        width = struct.get_int("width").value
        height = struct.get_int("height").value
        
        # Map buffer to numpy
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK
        
        try:
            # This is the batched frame — need to get batch metadata
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if batch_meta:
                # For each frame in batch, extract and store
                l_frame = batch_meta.frame_meta_list
                while l_frame is not None:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                        stream_id = frame_meta.source_id
                        channel_id = self.channel_map.get(stream_id)
                        
                        if channel_id:
                            # Get frame from batch
                            frame = np.ndarray(
                                shape=(height, width, 3),
                                dtype=np.uint8,
                                buffer=map_info.data
                            ).copy()
                            
                            with self._frame_lock:
                                self._latest_frames[channel_id] = frame
                            
                            # Fire frame callbacks
                            for callback in self._frame_callbacks:
                                try:
                                    callback(channel_id, frame)
                                except Exception as e:
                                    logger.error(f"Frame callback error: {e}")
                        
                        l_frame = l_frame.next
                    except StopIteration:
                        break
        finally:
            buf.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def start(self) -> bool:
        """Start the DeepStream pipeline."""
        if not self.pipeline:
            logger.error("Pipeline not built. Call build_pipeline() first.")
            return False
        
        # Set up bus message handling
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        # Start GLib main loop in a separate thread
        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
        
        # Set pipeline to PLAYING
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to set pipeline to PLAYING")
            self.stop()
            return False
        
        self.is_running = True
        logger.info("DeepStream pipeline started")
        return True
    
    def stop(self):
        """Stop the DeepStream pipeline gracefully."""
        self.is_running = False
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        if self.loop and self.loop.is_running():
            self.loop.quit()
        
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5)
        
        logger.info("DeepStream pipeline stopped")
    
    def _run_loop(self):
        """Run the GLib main loop."""
        try:
            self.loop.run()
        except Exception as e:
            logger.error(f"GLib main loop error: {e}")
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages."""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.EOS:
            logger.info("Pipeline reached end of stream")
            self.stop()
        
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Pipeline error: {err.message}")
            logger.debug(f"Debug: {debug}")
            # Attempt recovery for RTSP errors
            source = message.src.get_name()
            if 'rtspsrc' in source:
                logger.info(f"RTSP error on {source}, will attempt reconnection...")
                # DeepStream handles reconnection automatically with rtspsrc
        
        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"Pipeline warning: {err.message}")
        
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                logger.debug(f"Pipeline state: {old.value_nick} → {new.value_nick}")
    
    # ─── Public API for Integration with Flask App ────────────────────
    
    def register_detection_callback(self, callback: Callable):
        """Register a callback for detection events: callback(channel_id, detections, frame_num)"""
        self._detection_callbacks.append(callback)
    
    def register_frame_callback(self, callback: Callable):
        """Register a callback for new frames: callback(channel_id, frame_ndarray)"""
        self._frame_callbacks.append(callback)
    
    def get_latest_frame(self, channel_id: str) -> Optional[np.ndarray]:
        """Get the latest decoded frame for a channel (for MJPEG streaming)."""
        with self._frame_lock:
            return self._latest_frames.get(channel_id)
    
    def get_latest_detections(self, channel_id: str) -> List[dict]:
        """Get the latest detections for a channel."""
        with self._detection_lock:
            return self._latest_detections.get(channel_id, [])
    
    def get_fps(self, channel_id: str) -> float:
        """Get current FPS for a channel."""
        return self._fps_counters.get(channel_id, {}).get('fps', 0.0)
    
    def get_statistics(self) -> dict:
        """Get pipeline statistics for all streams."""
        stats = {}
        for stream_id, channel_id in self.channel_map.items():
            stats[channel_id] = {
                'stream_id': stream_id,
                'fps': self.get_fps(channel_id),
                'has_frame': channel_id in self._latest_frames,
                'detection_count': len(self._latest_detections.get(channel_id, [])),
            }
        return stats
    
    def add_stream(self, channel_id: str, rtsp_url: str) -> bool:
        """Dynamically add a new RTSP stream to the running pipeline."""
        if not self.is_running:
            logger.error("Pipeline not running")
            return False
        
        stream_id = max(self.source_bins.keys(), default=-1) + 1
        
        # Pause pipeline for modification
        self.pipeline.set_state(Gst.State.PAUSED)
        
        source_bin = self._create_source_bin(stream_id, rtsp_url, channel_id)
        if not source_bin:
            self.pipeline.set_state(Gst.State.PLAYING)
            return False
        
        self.pipeline.add(source_bin)
        
        streammux = self.pipeline.get_by_name("stream-muxer")
        srcpad = source_bin.get_static_pad("src")
        sinkpad = streammux.request_pad_simple(f"sink_{stream_id}")
        
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            logger.error(f"Failed to link new source {channel_id}")
            self.pipeline.set_state(Gst.State.PLAYING)
            return False
        
        self.source_bins[stream_id] = source_bin
        self.channel_map[stream_id] = channel_id
        self.channel_to_stream[channel_id] = stream_id
        
        # Update batch size
        streammux.set_property('batch-size', len(self.source_bins))
        
        source_bin.set_state(Gst.State.PLAYING)
        self.pipeline.set_state(Gst.State.PLAYING)
        
        logger.info(f"Dynamically added stream: {channel_id}")
        return True
    
    def remove_stream(self, channel_id: str) -> bool:
        """Dynamically remove an RTSP stream from the running pipeline."""
        stream_id = self.channel_to_stream.get(channel_id)
        if stream_id is None:
            logger.warning(f"Stream not found: {channel_id}")
            return False
        
        source_bin = self.source_bins.get(stream_id)
        if not source_bin:
            return False
        
        # Set source bin to NULL
        source_bin.set_state(Gst.State.NULL)
        
        # Remove from streammux
        streammux = self.pipeline.get_by_name("stream-muxer")
        sinkpad = streammux.get_static_pad(f"sink_{stream_id}")
        if sinkpad:
            streammux.release_request_pad(sinkpad)
        
        self.pipeline.remove(source_bin)
        
        # Cleanup mappings
        del self.source_bins[stream_id]
        del self.channel_map[stream_id]
        del self.channel_to_stream[channel_id]
        
        with self._frame_lock:
            self._latest_frames.pop(channel_id, None)
        with self._detection_lock:
            self._latest_detections.pop(channel_id, None)
        
        logger.info(f"Removed stream: {channel_id}")
        return True
```

### 6.4 Implementation: Module Adapter

This adapter translates DeepStream metadata into the format your existing detection modules expect:

```python
# modules/ds_module_adapter.py
"""
Adapter layer that converts DeepStream detection metadata into the format
expected by existing Sakshi.AI detection modules (QueueMonitor, DressCode, etc.)

This allows existing modules to work with minimal changes.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class DSDetectionResult:
    """
    Mimics the Ultralytics Results object so existing modules 
    can use ds_result.boxes.xyxy, ds_result.boxes.conf, etc.
    """
    
    def __init__(self, detections: List[dict], frame_shape: tuple):
        self.orig_shape = frame_shape[:2]  # (height, width)
        self.boxes = DSBoxes(detections, frame_shape)
        self.names = self._build_names(detections)
    
    def _build_names(self, detections: List[dict]) -> Dict[int, str]:
        names = {}
        for det in detections:
            names[det['class_id']] = det.get('class_name', str(det['class_id']))
        return names


class DSBoxes:
    """Mimics Ultralytics Boxes object."""
    
    def __init__(self, detections: List[dict], frame_shape: tuple):
        if not detections:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)
            self.id = None
            self.data = np.empty((0, 6), dtype=np.float32)
            return
        
        h, w = frame_shape[:2]
        
        xyxy_list = []
        conf_list = []
        cls_list = []
        id_list = []
        
        for det in detections:
            bbox = det['bbox']
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = x1 + bbox['w']
            y2 = y1 + bbox['h']
            
            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(det['confidence'])
            cls_list.append(det['class_id'])
            
            tracker_id = det.get('tracker_id', -1)
            if tracker_id >= 0:
                id_list.append(tracker_id)
        
        self.xyxy = np.array(xyxy_list, dtype=np.float32)
        self.conf = np.array(conf_list, dtype=np.float32)
        self.cls = np.array(cls_list, dtype=np.float32)
        self.id = np.array(id_list, dtype=np.float32) if id_list else None
        
        # data: [x1, y1, x2, y2, conf, cls]
        self.data = np.column_stack([
            self.xyxy,
            self.conf.reshape(-1, 1),
            self.cls.reshape(-1, 1)
        ])
    
    @property
    def xywh(self):
        """Convert xyxy to xywh format."""
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        xywh = self.xyxy.copy()
        xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]  # width
        xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]  # height
        xywh[:, 0] = self.xyxy[:, 0] + xywh[:, 2] / 2   # cx
        xywh[:, 1] = self.xyxy[:, 1] + xywh[:, 3] / 2   # cy
        return xywh
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self


class DeepStreamModuleAdapter:
    """
    Bridges DeepStream pipeline output to existing detection modules.
    
    Instead of modules calling model.predict(frame), they call:
        adapter.get_results(channel_id) → returns DSDetectionResult
    
    This is a drop-in replacement for the shared detection caching
    currently done in MultiModuleVideoProcessor.
    """
    
    def __init__(self, ds_pipeline):
        """
        Args:
            ds_pipeline: DeepStreamPipeline instance
        """
        self.pipeline = ds_pipeline
        
        # Register for detection callbacks
        self.pipeline.register_detection_callback(self._on_detections)
        
        # Per-channel detection cache
        self._results_cache: Dict[str, DSDetectionResult] = {}
        self._frame_shapes: Dict[str, tuple] = {}
    
    def _on_detections(self, channel_id: str, detections: List[dict], frame_num: int):
        """Called by DeepStream probe for each frame's detections."""
        frame = self.pipeline.get_latest_frame(channel_id)
        shape = frame.shape if frame is not None else (720, 1280, 3)
        self._frame_shapes[channel_id] = shape
        self._results_cache[channel_id] = DSDetectionResult(detections, shape)
    
    def get_results(self, channel_id: str) -> Optional[DSDetectionResult]:
        """
        Get the latest detection results for a channel.
        Drop-in replacement for YOLO model.predict() results.
        """
        return self._results_cache.get(channel_id)
    
    def get_person_detections(self, channel_id: str, class_id: int = 0) -> List[dict]:
        """
        Get person detections filtered by class ID.
        Most modules need person bounding boxes.
        """
        result = self._results_cache.get(channel_id)
        if result is None:
            return []
        
        persons = []
        mask = result.boxes.cls == class_id
        for i in range(len(result.boxes.xyxy)):
            if mask[i]:
                bbox = result.boxes.xyxy[i]
                persons.append({
                    'bbox': bbox,
                    'confidence': float(result.boxes.conf[i]),
                    'tracker_id': int(result.boxes.id[i]) if result.boxes.id is not None and i < len(result.boxes.id) else -1,
                })
        return persons
    
    def get_tracked_objects(self, channel_id: str) -> Dict[int, dict]:
        """
        Get all tracked objects with their tracker IDs.
        Replacement for deep-sort-realtime in ServiceDisciplineMonitor.
        """
        result = self._results_cache.get(channel_id)
        if result is None or result.boxes.id is None:
            return {}
        
        tracked = {}
        for i in range(len(result.boxes.xyxy)):
            if i < len(result.boxes.id):
                track_id = int(result.boxes.id[i])
                if track_id >= 0:
                    tracked[track_id] = {
                        'bbox': result.boxes.xyxy[i],
                        'class_id': int(result.boxes.cls[i]),
                        'class_name': result.names.get(int(result.boxes.cls[i]), ''),
                        'confidence': float(result.boxes.conf[i]),
                    }
        return tracked
```

### 6.5 Integration with app.py

Here's how to modify the main application to use DeepStream:

```python
# Key changes in app.py — replace the per-camera processing with DeepStream

# ─── BEFORE (current code) ─────────────────────────
# For each channel:
#   processor = MultiModuleVideoProcessor(channel_id, rtsp_url, modules, ...)
#   processor.start()
#   # Each processor has its own OpenCV capture thread + YOLO model

# ─── AFTER (DeepStream) ────────────────────────────
import json
from modules.ds_pipeline import DeepStreamPipeline
from modules.ds_module_adapter import DeepStreamModuleAdapter

# Load channels config
with open('config/channels.json') as f:
    channels_config = json.load(f)

channels = [ch for ch in channels_config['channels'] if ch.get('enabled', True)]

# Create single DeepStream pipeline for ALL cameras
ds_config = {
    'muxer_width': 1280,
    'muxer_height': 720,
    'tracker_config': 'config/ds_tracker.txt',
}

ds_pipeline = DeepStreamPipeline(ds_config)
ds_pipeline.build_pipeline(channels)

# Create adapter for existing modules
ds_adapter = DeepStreamModuleAdapter(ds_pipeline)

# Set up detection callback to dispatch to modules
def on_detections(channel_id, detections, frame_num):
    """Route detections to the appropriate modules for this channel."""
    channel_cfg = next((ch for ch in channels if ch['channel_id'] == channel_id), None)
    if not channel_cfg:
        return
    
    frame = ds_pipeline.get_latest_frame(channel_id)
    if frame is None:
        return
    
    # Get the cached results in Ultralytics-compatible format
    results = ds_adapter.get_results(channel_id)
    if results is None:
        return
    
    # Run each module's logic (same as current process_frame)
    for module_info in active_modules.get(channel_id, []):
        module = module_info['instance']
        try:
            # Modules receive frame + pre-computed detections
            module.process_frame(frame, shared_detections=results)
        except Exception as e:
            logger.error(f"Module error on {channel_id}: {e}")

ds_pipeline.register_detection_callback(on_detections)

# Start the pipeline
ds_pipeline.start()

# Video feed endpoint — now uses DeepStream frames
@app.route('/video_feed/<app_name>/<channel_id>')
def video_feed(app_name, channel_id):
    def generate():
        while True:
            frame = ds_pipeline.get_latest_frame(channel_id)
            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(1.0 / 15)  # 15 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
```

---

## 7. Module-by-Module Migration

### 7.1 Changes Required Per Module

Each detection module currently does:
```python
def process_frame(self, frame, shared_detections=None):
    if shared_detections:
        results = shared_detections  # Use cached YOLO results
    else:
        results = self.model.predict(frame)  # Run YOLO
    
    # Process detections...
    for det in results[0].boxes:
        # Module-specific logic
```

**After migration**, modules receive `DSDetectionResult` which has the same interface (`boxes.xyxy`, `boxes.conf`, `boxes.cls`, `boxes.id`). The key changes per module:

| Module | Change Required | Details |
|--------|----------------|---------|
| **QueueMonitor** | Minimal | `shared_detections` already used. DSDetectionResult is compatible. ROI filtering stays the same. |
| **DressCodeMonitoring** | Minimal | Classification detections come from secondary nvinfer or remain in Python. ROI filtering stays same. |
| **CashDetection** | Minimal | Can use secondary nvinfer for cash model, or keep Python inference for specialized model. |
| **SmokingDetection** | Minimal | Same — detections from nvinfer or secondary. |
| **FallDetection** | **Moderate** | Replace custom tracking history with `nvtracker` IDs. The tracker provides persistent object IDs across frames. Remove custom fall heuristics that depend on frame-by-frame position arrays; instead use tracker ID → position history map. |
| **ServiceDisciplineMonitor** | **Moderate** | Replace `deep-sort-realtime` with nvtracker IDs from DeepStream. The `boxes.id` field provides tracker IDs. Remove Deep-SORT initialization code. |
| **TableServiceMonitor** | Minimal | Uses person positions + ROI — compatible. |
| **IdleTimeMonitor** | Minimal | Person positions + time tracking — compatible. |
| **CrowdDetection** | Minimal | Person count in area — compatible. |
| **PPEMonitoring** | **Moderate** | If PPE model is a separate model, use secondary nvinfer. Otherwise keep Python inference on cropped person images. |
| **MaterialTheftMonitor** | None | Uses OpenCV background subtraction, not YOLO. Receives raw frames — works as-is via frame callback. |
| **PeopleCounter** | Minimal | Uses counting line + detections — compatible. |
| **UnauthorizedEntryMonitor** | Minimal | Person positions + restricted areas — compatible. |
| **GIFRecorder** | **Moderate** | Frame source changes from OpenCV to appsink. frames are delivered via `ds_pipeline.get_latest_frame()`. |

### 7.2 Example: Migrating QueueMonitor

**Before** (current code):
```python
class QueueMonitor:
    def process_frame(self, frame, shared_detections=None):
        if shared_detections and len(shared_detections) > 0:
            results = shared_detections
        else:
            results = self.model(frame, conf=0.5, verbose=False)
        
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls == 0:  # person
                    if self._point_in_roi((x1+x2)/2, (y1+y2)/2, self.queue_roi):
                        queue_count += 1
```

**After** (DeepStream — almost identical):
```python
class QueueMonitor:
    def process_frame(self, frame, shared_detections=None):
        # shared_detections is now a DSDetectionResult — same interface!
        if shared_detections is not None:
            results = shared_detections
        else:
            return  # No fallback needed — DeepStream always provides detections
        
        # This code is UNCHANGED — DSDetectionResult has same .boxes.xyxy etc.
        if results and results.boxes is not None:
            for i in range(len(results.boxes.xyxy)):
                x1, y1, x2, y2 = results.boxes.xyxy[i]
                conf = float(results.boxes.conf[i])
                cls = int(results.boxes.cls[i])
                if cls == 0:  # person
                    if self._point_in_roi((x1+x2)/2, (y1+y2)/2, self.queue_roi):
                        queue_count += 1
```

### 7.3 Example: Migrating FallDetection (requires tracking changes)

**Before**:
```python
class FallDetection:
    def __init__(self):
        self.tracked_persons = {}  # Manual tracking with position history
    
    def process_frame(self, frame, shared_detections=None):
        # Manual tracking logic to associate person across frames
        results = shared_detections or self.model(frame)
        # ... complex association code ...
```

**After**:
```python
class FallDetection:
    def __init__(self):
        self.tracked_persons = {}  # Now uses nvtracker IDs
    
    def process_frame(self, frame, shared_detections=None):
        results = shared_detections
        if results is None or results.boxes.id is None:
            return
        
        for i in range(len(results.boxes.xyxy)):
            cls = int(results.boxes.cls[i])
            if cls != 0:  # Only track persons
                continue
            
            tracker_id = int(results.boxes.id[i]) if results.boxes.id is not None else -1
            if tracker_id < 0:
                continue
            
            bbox = results.boxes.xyxy[i]
            x1, y1, x2, y2 = bbox
            h = y2 - y1
            w = x2 - x1
            aspect_ratio = w / max(h, 1)
            center_y = (y1 + y2) / 2
            
            # Get or create tracking history for this person
            if tracker_id not in self.tracked_persons:
                self.tracked_persons[tracker_id] = {
                    'positions': [],
                    'aspect_ratios': [],
                    'first_seen': time.time()
                }
            
            person = self.tracked_persons[tracker_id]
            person['positions'].append(center_y)
            person['aspect_ratios'].append(aspect_ratio)
            
            # Keep last 30 frames of history
            person['positions'] = person['positions'][-30:]
            person['aspect_ratios'] = person['aspect_ratios'][-30:]
            
            # Fall detection logic (same as before)
            if len(person['positions']) >= 5:
                height_drop = person['positions'][-1] - min(person['positions'][-10:])
                if (aspect_ratio > self.aspect_ratio_threshold and
                    height_drop > self.height_drop_ratio * h):
                    self._trigger_fall_alert(frame, tracker_id, bbox)
```

---

## 8. Configuration Migration

### 8.1 Updated default.json

Add these DeepStream-specific settings to your `config/default.json`:

```json
{
  "deepstream": {
    "enabled": true,
    "muxer_width": 1280,
    "muxer_height": 720,
    "muxer_batch_timeout_usec": 40000,
    "gpu_id": 0,
    "primary_model": {
      "config_file": "config/ds_yolo_primary.txt",
      "engine_file": "models/yolo11n.engine",
      "labels_file": "config/labels.txt"
    },
    "secondary_models": {
      "cash": {
        "config_file": "config/ds_cash_secondary.txt",
        "engine_file": "models/cash.engine"
      }
    },
    "tracker": {
      "config_file": "config/ds_tracker.txt",
      "tracker_lib": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    },
    "rtsp": {
      "latency_ms": 200,
      "protocols": "tcp",
      "reconnect_interval_sec": 5
    },
    "frame_extraction": {
      "enabled": true,
      "max_buffers": 2,
      "drop_old": true
    },
    "fallback_to_opencv": true
  }
}
```

### 8.2 channels.json — No Changes Needed

Your existing `channels.json` format works perfectly. The DeepStream pipeline reads `rtsp_url`, `channel_id`, `enabled`, and `modules` exactly as configured.

---

## 9. TensorRT Model Conversion

### 9.1 Convert YOLO .pt to ONNX

DeepStream's `nvinfer` needs TensorRT engines, which are built from ONNX models. Your RTX 5060 Ti TensorRT engines must be built ON the target GPU.

```bash
# On your Ubuntu server:
source .venv/bin/activate

# Step 1: Export YOLO to ONNX
python3 -c "
from ultralytics import YOLO

# Primary detector
model = YOLO('models/yolo11n.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=17)
print('Exported yolo11n.onnx')

# Best model
model = YOLO('models/best.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=17)
print('Exported best.onnx')
"

# Step 2: Build TensorRT engine (automatically done by nvinfer on first run)
# OR manually build with trtexec for more control:
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolo11n.onnx \
    --saveEngine=models/yolo11n.engine \
    --fp16 \
    --workspace=4096 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:4x3x640x640 \
    --maxShapes=images:8x3x640x640

# For cash detection model
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/cash.onnx \
    --saveEngine=models/cash.engine \
    --fp16 \
    --workspace=2048
```

> **IMPORTANT**: TensorRT engines are GPU-architecture specific. Engines built on your Windows development machine (if any) will NOT work on the RTX 5060 Ti. Always build engines on the deployment server.

### 9.2 Custom YOLO Output Parser

DeepStream's `nvinfer` needs a custom output parser library for YOLO models because YOLO output format differs from standard SSD/FasterRCNN.

**Option A: Use NVIDIA's DeepStream-YOLO repository (recommended)**

```bash
cd /opt/nvidia/deepstream/deepstream/sources/
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo

# Build for YOLOv8/v11
# Edit Makefile to match your CUDA version
make -j$(nproc)

# Copy the library
cp libnvdsinfer_custom_impl_Yolo.so /path/to/sakshiai/lib/libnvdsinfer_custom_yolo.so
```

**Option B: Use Ultralytics' built-in TensorRT export**

If you export with `model.export(format='engine')`, Ultralytics creates a TensorRT engine that can be loaded directly in Python (bypassing nvinfer). You could then:
1. Use DeepStream only for decode + mux
2. Run inference in Python using the Ultralytics engine
3. This is a hybrid approach — simpler but slightly less performant

```python
# Hybrid approach: DeepStream decode + Python TensorRT inference
from ultralytics import YOLO

# Load TensorRT engine
model = YOLO('models/yolo11n.engine')

def on_frame(channel_id, frame):
    """Called by DeepStream for each decoded frame."""
    results = model.predict(frame, conf=0.5, verbose=False)
    # Process results with existing modules...
```

### 9.3 Choosing: Full nvinfer vs Hybrid

| Approach | Pros | Cons |
|----------|------|------|
| **Full nvinfer** | Maximum performance, zero-copy GPU inference, batched across all streams | Requires custom parser library, more complex setup |
| **Hybrid (DS decode + Python infer)** | Simpler migration, reuses existing model code, faster to implement | GPU→CPU→GPU copy for inference, Python GIL for inference |

**Recommendation**: Start with the **Hybrid approach** for faster migration, then optimize to full nvinfer once everything is stable.

---

## 10. Testing & Validation

### 10.1 Phase 1: Verify DeepStream Installation

```bash
# Test 1: Basic GStreamer NVDEC pipeline
gst-launch-1.0 rtspsrc location="rtsp://admin:admin@132.154.208.136:555/cam/realmonitor?channel=1&subtype=1" ! \
    rtph264depay ! h264parse ! nvv4l2decoder ! \
    nvvideoconvert ! 'video/x-raw, format=BGR' ! \
    appsink

# Test 2: DeepStream test app (comes with SDK)
cd /opt/nvidia/deepstream/deepstream/samples
deepstream-app -c source1_csi_dec_infer_resnet_int8.txt

# Test 3: Python bindings
python3 -c "
import pyds
print('pyds version:', pyds.__version__)
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
print('GStreamer version:', Gst.version_string())
"
```

### 10.2 Phase 2: Single Camera Test

```python
# test_single_camera.py
"""Test DeepStream pipeline with a single camera."""
import sys
sys.path.insert(0, '.')

from modules.ds_pipeline import DeepStreamPipeline
import time
import cv2

config = {
    'muxer_width': 1280,
    'muxer_height': 720,
}

channels = [{
    'channel_id': 'camera_1',
    'channel_name': 'Test Camera',
    'rtsp_url': 'rtsp://admin:admin@132.154.208.136:555/cam/realmonitor?channel=1&subtype=1',
    'enabled': True,
}]

pipeline = DeepStreamPipeline(config)

def on_detection(channel_id, detections, frame_num):
    if frame_num % 30 == 0:  # Print every 30th frame
        print(f"[{channel_id}] Frame {frame_num}: {len(detections)} detections")
        for det in detections[:3]:
            print(f"  - {det['class_name']} ({det['confidence']:.2f}) "
                  f"tracker_id={det['tracker_id']}")

pipeline.register_detection_callback(on_detection)

if pipeline.build_pipeline(channels):
    pipeline.start()
    print("Pipeline running. Press Ctrl+C to stop.")
    try:
        while True:
            frame = pipeline.get_latest_frame('camera_1')
            if frame is not None:
                cv2.imshow('DeepStream Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(0.033)  # ~30fps
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
else:
    print("Failed to build pipeline!")
```

### 10.3 Phase 3: Multi-Camera Test

```python
# test_multi_camera.py
"""Test DeepStream pipeline with all cameras from channels.json."""
import sys, json, time
sys.path.insert(0, '.')

from modules.ds_pipeline import DeepStreamPipeline

with open('config/channels.json') as f:
    channels_config = json.load(f)

channels = [ch for ch in channels_config['channels'] if ch.get('enabled', True)]

config = {
    'muxer_width': 1280,
    'muxer_height': 720,
}

pipeline = DeepStreamPipeline(config)

def on_detection(channel_id, detections, frame_num):
    if frame_num % 100 == 0:
        fps = pipeline.get_fps(channel_id)
        person_count = sum(1 for d in detections if d['class_id'] == 0)
        print(f"[{channel_id}] Frame {frame_num} | FPS: {fps:.1f} | Persons: {person_count}")

pipeline.register_detection_callback(on_detection)

if pipeline.build_pipeline(channels):
    pipeline.start()
    print(f"Pipeline running with {len(channels)} cameras. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(5)
            stats = pipeline.get_statistics()
            print("\n--- Pipeline Statistics ---")
            for ch_id, st in stats.items():
                print(f"  {ch_id}: FPS={st['fps']:.1f}, "
                      f"Detections={st['detection_count']}, "
                      f"Frame={'Yes' if st['has_frame'] else 'No'}")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
else:
    print("Failed to build pipeline!")
```

### 10.4 Phase 4: Module Integration Test

```python
# test_modules_with_ds.py
"""Test that existing modules work with DeepStream adapter."""
import sys, json, time
sys.path.insert(0, '.')

from modules.ds_pipeline import DeepStreamPipeline
from modules.ds_module_adapter import DeepStreamModuleAdapter
from modules.queue_monitor import QueueMonitor

with open('config/channels.json') as f:
    channels_config = json.load(f)

channels = [ch for ch in channels_config['channels'] if ch.get('enabled', True)]

config = {'muxer_width': 1280, 'muxer_height': 720}
pipeline = DeepStreamPipeline(config)
adapter = DeepStreamModuleAdapter(pipeline)

# Initialize QueueMonitor for camera_1
cam1_config = next(ch for ch in channels if ch['channel_id'] == 'camera_1')
queue_cfg = next(m for m in cam1_config['modules'] if m['type'] == 'QueueMonitor')
queue_monitor = QueueMonitor(config=queue_cfg['config'])

def on_detection(channel_id, detections, frame_num):
    if channel_id != 'camera_1':
        return
    
    frame = pipeline.get_latest_frame(channel_id)
    results = adapter.get_results(channel_id)
    
    if frame is not None and results is not None:
        annotated = queue_monitor.process_frame(frame, shared_detections=[results])
        if frame_num % 60 == 0:
            print(f"QueueMonitor processed frame {frame_num}")

pipeline.register_detection_callback(on_detection)

if pipeline.build_pipeline(channels):
    pipeline.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
```

### 10.5 Validation Checklist

- [ ] DeepStream SDK installed and `deepstream-app --version-all` works
- [ ] GStreamer NVDEC decoder works: `gst-inspect-1.0 nvv4l2decoder`
- [ ] pyds importable in Python: `import pyds`
- [ ] Single RTSP stream decoded and displayed via DeepStream
- [ ] Multi-stream pipeline runs without crash
- [ ] Detections extracted correctly via buffer probe
- [ ] Frames extracted via appsink in numpy format
- [ ] MJPEG video feed works in browser
- [ ] QueueMonitor produces correct alerts
- [ ] DressCodeMonitoring produces correct alerts
- [ ] FallDetection works with nvtracker IDs
- [ ] GIF recording captures alert clips
- [ ] Telegram notifications send correctly
- [ ] Database alerts are logged
- [ ] Pipeline gracefully handles camera disconnection
- [ ] Pipeline auto-reconnects on camera reconnection
- [ ] Dynamic add/remove camera works
- [ ] Memory usage is stable over 24+ hours
- [ ] GPU utilization is within expected range

---

## 11. Performance Tuning

### 11.1 Expected Performance on RTX 5060 Ti

| Streams | Resolution | FPS/stream | GPU Decode | GPU Infer | GPU Mem |
|---------|-----------|------------|-----------|-----------|---------|
| 4 | 720p | 30 | ~5% | ~15% | ~2 GB |
| 8 | 720p | 30 | ~10% | ~30% | ~3 GB |
| 12 | 720p | 20 | ~15% | ~45% | ~4 GB |
| 16 | 720p | 15 | ~20% | ~60% | ~5 GB |
| 4 | 1080p | 30 | ~10% | ~25% | ~3 GB |
| 8 | 1080p | 25 | ~20% | ~50% | ~5 GB |

Your RTX 5060 Ti (16GB VRAM) should comfortably handle **12-16 cameras at 720p/15fps** with YOLO11n.

### 11.2 Optimization Knobs

**Reduce GPU inference load:**
```ini
# In ds_yolo_primary.txt:
interval=1        # Skip every other frame (infer on every 2nd frame)
network-mode=1    # FP16 (vs FP32) — ~2x faster, minimal accuracy loss
batch-size=4      # Match your camera count
```

**Reduce decode overhead:**
```python
# Use substream (lower resolution) from cameras
# In channels.json, change RTSP URL:
"rtsp_url": "rtsp://admin:admin@IP:PORT/cam/realmonitor?channel=1&subtype=1"
#                                                                   ^^^^^^^^
#                                                         subtype=0 (main), subtype=1 (sub)
```

**Reduce memory usage:**
```python
# Muxer output resolution
streammux.set_property('width', 960)    # Reduce from 1280
streammux.set_property('height', 540)   # Reduce from 720
```

**Reduce CPU usage (from frame extraction):**
```python
# Only extract frames when needed (e.g., every 3rd frame for MJPEG)
appsink.set_property('max-buffers', 1)
appsink.set_property('drop', True)
```

### 11.3 Monitor Performance

```bash
# GPU monitoring
watch -n 1 nvidia-smi

# DeepStream FPS
# Built into the pipeline's get_statistics() method

# System monitoring
htop   # CPU usage
iotop  # Disk I/O (for GIF writing)

# GStreamer debug (if issues)
export GST_DEBUG=3
export GST_DEBUG_DUMP_DOT_DIR=/tmp/gst-dot/
# Then convert: dot -Tpng /tmp/gst-dot/*.dot -o pipeline.png
```

---

## 12. Rollback Plan

If DeepStream migration encounters issues, the existing OpenCV pipeline is your fallback:

### 12.1 Feature Flag Approach

```python
# In app.py — use feature flag for easy rollback
USE_DEEPSTREAM = os.getenv('USE_DEEPSTREAM', 'true').lower() == 'true'

if USE_DEEPSTREAM:
    try:
        from modules.ds_pipeline import DeepStreamPipeline
        from modules.ds_module_adapter import DeepStreamModuleAdapter
        # ... DeepStream initialization ...
        logger.info("Using DeepStream pipeline")
    except Exception as e:
        logger.warning(f"DeepStream failed: {e}. Falling back to OpenCV.")
        USE_DEEPSTREAM = False

if not USE_DEEPSTREAM:
    from modules.multi_module_processor import MultiModuleVideoProcessor
    # ... existing OpenCV pipeline ...
    logger.info("Using OpenCV pipeline (fallback)")
```

### 12.2 Keep Old Code

Don't delete any existing modules. The new DeepStream code goes in new files:
- `modules/ds_pipeline.py` (NEW)
- `modules/ds_module_adapter.py` (NEW)
- `modules/ds_frame_grabber.py` (NEW)

Old code stays intact for rollback:
- `modules/multi_module_processor.py` (KEEP)
- `modules/rtsp_connection_pool.py` (KEEP)
- `modules/batch_processor.py` (KEEP)

---

## Migration Timeline Summary

| Phase | Task | Priority |
|-------|------|----------|
| **Phase 1** | Install DeepStream SDK + dependencies on Ubuntu server | HIGH |
| **Phase 2** | Build TensorRT engines for YOLO models on RTX 5060 Ti | HIGH |
| **Phase 3** | Implement `ds_pipeline.py` and `ds_module_adapter.py` | HIGH |
| **Phase 4** | Test single camera with DeepStream | HIGH |
| **Phase 5** | Test multi-camera pipeline | HIGH |
| **Phase 6** | Integrate with app.py using feature flag | MEDIUM |
| **Phase 7** | Migrate each detection module (minor interface changes) | MEDIUM |
| **Phase 8** | Update GIF recording to use DeepStream frames | MEDIUM |
| **Phase 9** | Test all modules end-to-end | HIGH |
| **Phase 10** | Performance tuning and 24-hour stability test | MEDIUM |
| **Phase 11** | Remove old OpenCV fallback (optional) | LOW |

---

## Quick Reference: Key Commands

```bash
# Check DeepStream version
deepstream-app --version-all

# Test NVDEC with RTSP
gst-launch-1.0 rtspsrc location=rtsp://... ! rtph264depay ! h264parse ! nvv4l2decoder ! fakesink

# Build TensorRT engine
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# Monitor GPU
nvidia-smi dmon -s pucvmet -d 1

# GStreamer debug
GST_DEBUG=3 python3 app.py

# Run app with DeepStream enabled
USE_DEEPSTREAM=true python3 app.py

# Run app with fallback (no DeepStream)
USE_DEEPSTREAM=false python3 app.py
```
