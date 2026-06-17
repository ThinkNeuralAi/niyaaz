# ============================================================================
# Sakshi.AI - Intelligent Video Analytics Platform
# Base: NVIDIA CUDA 12.1 + cuDNN 8 runtime on Ubuntu 22.04
# Requires: NVIDIA Container Toolkit on the host and driver >= 525
# ============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # RTSP over TCP, 5-second timeout (mirrors .env setting)
    OPENCV_FFMPEG_CAPTURE_OPTIONS="timeout;5000000|rtsp_transport;tcp"

WORKDIR /app

# --------------------------------------------------------------------------
# System dependencies
#   libgl1-mesa-glx  – required by opencv-python (libGL.so.1)
#   libglib2.0-0     – required by OpenCV threading
#   libsm6 / libxext6 / libxrender1 – required by OpenCV display subsystem
#   ffmpeg           – required for RTSP stream decoding via FFmpeg backend
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# --------------------------------------------------------------------------
# Install PyTorch with CUDA 12.1 wheels FIRST so the requirement
# "torch>=2.0.0" in requirements.txt is already satisfied when we run
# pip install -r requirements.txt, preventing the CPU-only wheel from
# being pulled from PyPI.
# --------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
        torch==2.1.2 \
        torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

# --------------------------------------------------------------------------
# Install remaining Python dependencies.
# torch / torchvision requirements are already met by the step above.
# --------------------------------------------------------------------------
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# Copy application source code
# Files/folders excluded by .dockerignore are not sent to the build context.
# --------------------------------------------------------------------------
COPY app.py .
COPY modules/ modules/
COPY templates/ templates/
COPY static/ static/
COPY config/ config/

# --------------------------------------------------------------------------
# Create runtime directories.
# data/, models/, logs/ and static/ are mounted as volumes at runtime,
# but the directories must exist inside the image as mount-points.
# --------------------------------------------------------------------------
RUN mkdir -p \
        data \
        logs \
        models \
        static/alerts \
        static/ppe_snapshots \
        static/cash_snapshots \
        static/fall_snapshots \
        static/smoking_snapshots \
        static/dresscode_snapshots \
        static/queue_violations \
        static/heatmaps \
        static/table_service_violations \
        static/service_discipline \
        static/material_theft_snapshots \
        static/idle_time_snapshots \
        static/unauthorized_entry_snapshots \
        static/person_smoking_snapshots

EXPOSE 5000

CMD ["python3", "app.py"]
