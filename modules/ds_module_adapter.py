"""
Sakshi.AI — DeepStream Module Adapter

Bridges DeepStream detection metadata to the Ultralytics-compatible format
that existing detection modules expect (boxes.xyxy, boxes.conf, boxes.cls, boxes.id).

Existing modules call:
    result.boxes.xyxy, result.boxes.conf, result.boxes.cls, result.boxes.id
This adapter provides DSDetectionResult that matches that interface.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DSBoxes:
    """Mimics Ultralytics Boxes object for compatibility with existing modules."""

    def __init__(self, detections: list, frame_shape: tuple):
        if not detections:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)
            self.id = None
            self.data = np.empty((0, 6), dtype=np.float32)
            return

        xyxy_list = []
        conf_list = []
        cls_list = []
        id_list = []
        has_tracker = False

        for det in detections:
            bbox = det["bbox"]
            x1 = bbox["x"]
            y1 = bbox["y"]
            x2 = x1 + bbox["w"]
            y2 = y1 + bbox["h"]

            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(det["confidence"])
            cls_list.append(det["class_id"])

            tracker_id = det.get("tracker_id", -1)
            if tracker_id >= 0:
                has_tracker = True
                id_list.append(tracker_id)
            else:
                id_list.append(-1)

        self.xyxy = np.array(xyxy_list, dtype=np.float32)
        self.conf = np.array(conf_list, dtype=np.float32)
        self.cls = np.array(cls_list, dtype=np.float32)
        self.id = np.array([x for x in id_list if x >= 0], dtype=np.float32) if has_tracker else None

        self.data = np.column_stack([
            self.xyxy,
            self.conf.reshape(-1, 1),
            self.cls.reshape(-1, 1),
        ])

    @property
    def xywh(self):
        """Convert xyxy to xywh (center-x, center-y, width, height)."""
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        xywh = np.empty_like(self.xyxy)
        xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]  # w
        xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]  # h
        xywh[:, 0] = self.xyxy[:, 0] + xywh[:, 2] / 2   # cx
        xywh[:, 1] = self.xyxy[:, 1] + xywh[:, 3] / 2   # cy
        return xywh

    @property
    def xyxyn(self):
        """Normalized xyxy (not available without frame shape)."""
        return self.xyxy

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __len__(self):
        return len(self.xyxy)


class DSDetectionResult:
    """
    Mimics Ultralytics Results object so existing modules work unchanged.
    Modules access: result.boxes.xyxy, result.boxes.conf, result.boxes.cls
    """

    def __init__(self, detections: list, frame_shape: tuple):
        self.orig_shape = frame_shape[:2]  # (height, width)
        self.boxes = DSBoxes(detections, frame_shape)
        self.names = self._build_names(detections)

    @staticmethod
    def _build_names(detections: list) -> dict:
        names = {}
        for det in detections:
            names[det["class_id"]] = det.get("class_name", str(det["class_id"]))
        return names

    def __len__(self):
        return len(self.boxes)

    def __bool__(self):
        return len(self.boxes) > 0


class DeepStreamModuleAdapter:
    """
    Bridges DeepStream pipeline output to existing detection modules.

    Modules that used to call model.predict(frame) now call:
        adapter.get_results(channel_id) → DSDetectionResult
    which has the same interface as Ultralytics Results.

    Also provides:
        adapter.get_person_detections(channel_id) — filtered person bboxes
        adapter.get_tracked_objects(channel_id) — tracker ID → detection map
    """

    def __init__(self, ds_pipeline):
        """
        Args:
            ds_pipeline: DeepStreamPipeline instance
        """
        self.pipeline = ds_pipeline
        self.pipeline.register_detection_callback(self._on_detections)

        # Per-channel detection cache
        self._results_cache: Dict[str, DSDetectionResult] = {}
        self._raw_detections: Dict[str, list] = defaultdict(list)

    def _on_detections(self, channel_id: str, detections: list, frame_num: int):
        """Called by DeepStream probe for each frame's detections."""
        frame = self.pipeline.get_latest_frame(channel_id)
        shape = frame.shape if frame is not None else (720, 1280, 3)
        self._raw_detections[channel_id] = detections
        self._results_cache[channel_id] = DSDetectionResult(detections, shape)

    def get_results(self, channel_id: str) -> Optional[DSDetectionResult]:
        """
        Get the latest detection results for a channel.
        Drop-in for Ultralytics model.predict() results.
        """
        return self._results_cache.get(channel_id)

    def get_person_detections(self, channel_id: str, person_class_id: int = 0) -> list:
        """Get person detections filtered by class ID."""
        result = self._results_cache.get(channel_id)
        if result is None:
            return []

        persons = []
        mask = result.boxes.cls == person_class_id
        for i in range(len(result.boxes.xyxy)):
            if mask[i]:
                persons.append({
                    "bbox": result.boxes.xyxy[i],
                    "confidence": float(result.boxes.conf[i]),
                    "tracker_id": int(result.boxes.id[i]) if result.boxes.id is not None and i < len(result.boxes.id) else -1,
                })
        return persons

    def get_tracked_objects(self, channel_id: str) -> Dict[int, dict]:
        """
        Get all tracked objects keyed by tracker ID.
        Replacement for deep-sort-realtime.
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
                        "bbox": result.boxes.xyxy[i],
                        "class_id": int(result.boxes.cls[i]),
                        "class_name": result.names.get(int(result.boxes.cls[i]), ""),
                        "confidence": float(result.boxes.conf[i]),
                    }
        return tracked

    def get_raw_detections(self, channel_id: str) -> list:
        """Get raw detection dicts from DeepStream metadata."""
        return list(self._raw_detections.get(channel_id, []))
