"""
Service Discipline Monitor — CLEANED UP VERSION
================================================

Event-based service monitoring:
    T_seated:        Customer sits at a table (after dwell-time confirmation)
    T_order_start:   Waiter first interacts with that customer
    T_order_end:     Waiter leaves after taking the order
    T_food_served:   Waiter brings food to that table

Metrics:
    Order wait time   = T_order_start - T_seated
    Service wait time = T_food_served - T_order_start

import cv2
import math
import logging
import numpy as np
import json
import time
import os
import torch
from datetime import datetime
from pathlib import Path

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logging.warning("DeepSort not available. Install with: pip install deep-sort-realtime")

from .model_manager import get_shared_model
from .yolo_detector import YOLODetector
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)


class ServiceDisciplineMonitor:
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app

        # Model configuration
        self.person_model_path = "models/yolo11n.pt"
        self.uniform_model_path = "models/best.pt"
        self.conf_threshold = 0.5
        self.nms_iou = 0.45

        # Uniform / staff indicator classes
        self.server_uniform_classes = {
            "Uniform_black", "Uniform_grey", "Uniform_cream",
            "Uniform_blue", "Uniform_brown",
        }
        self.server_hairnet_class = "Hairnet"
        self.hairnet_conf_threshold = 0.4

        # Table cleanliness classes
        self.table_clean_class = "Table_clean"
        self.table_unclean_class = "Table_unclean"
        self.unclean_conf_threshold = 0.6

        # Person class IDs
        self.person_class_id_yolo11n = 0

        # Uniform-detection filtering
        self.uniform_iou_threshold = 0.15
        self.uniform_proximity_px = 150
        self.uniform_memory_frames = 90       # ~3s @ 30 fps
        self.uniform_crop_recheck = True
        self.uniform_crop_conf_threshold = 0.3
        self._track_uniform_memory = {}       # {track_id: last_frame_with_uniform}

        # Load detectors
        self.person_detector = YOLODetector(
            model_path=self.person_model_path,
            confidence_threshold=0.25,
            img_size=640,
            person_class_id=self.person_class_id_yolo11n,
        )
        self.uniform_model = get_shared_model(self.uniform_model_path)
        logger.info(f"[{self.channel_id}] Loaded dual models: persons + uniforms")

        # DeepSORT
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=30, n_init=3,
                max_iou_distance=0.7, max_cosine_distance=0.3,
                nn_budget=50, embedder="mobilenet",
                embedder_gpu=torch.cuda.is_available(),
            )
            self.tracking_enabled = True
        else:
            self.tracker = None
            self.tracking_enabled = False

        # Table ROIs
        self.table_rois = {}
        self.table_display_names = {}

        # Settings
        self.settings = {
            "order_wait_threshold": 300.0,        # 5 min
            "service_wait_threshold": 600.0,      # 10 min
            "alert_cooldown": 1800.0,             # 30 min per table per violation type
            "track_timeout": 15.0,
            # Interaction is NOW scaled to person height (multiplier).
            # A waiter is "near" if distance < interaction_distance_factor * customer_height.
            "interaction_distance_factor": 1.5,
            "interaction_distance_min_px": 80,    # absolute floor for tiny bboxes
            "interaction_duration": 2.0,
            "food_served_gap": 10.0,
            # New: customer must dwell inside ROI this long before T_seated is recorded.
            "seated_dwell_seconds": 8.0,
            # New: waiter must be GONE this long before interaction-end is committed
            # (smooths over 1-frame tracker dropouts).
            "interaction_grace_seconds": 1.5,
            # New: cleanliness state expires if not re-detected within this window.
            "cleanliness_timeout_seconds": 60.0,
        }

        # Per-track state
        # {track_id: {
        #   "type": "customer"|"waiter",
        #   "table_id": str,
        #   "center": [x,y], "bbox": [x1,y1,x2,y2],
        #   "first_seen_in_roi": ts | None,   # for dwell-time gating
        #   "T_seated": ts | None,
        #   "T_order_start": ts | None,
        #   "T_order_end": ts | None,
        #   "T_food_served": ts | None,
        #   "last_seen": ts,
        #   "order_wait_time": float | None,
        #   "service_wait_time": float | None,
        #   "interaction_history": [(waiter_id, ts, type)],
        #   "waiter_at_table": bool,
        #   "waiter_visit_count": int,        # how many distinct visits a waiter has made
        #   "order_alert_fired": bool,        # one-shot, prevents repeat
        #   "service_alert_fired": bool,
        # }}
        self.person_tracks = {}

        # Per-table state
        # {table_id: {
        #   "customer_track_ids": [...],
        #   "waiter_track_ids": [...],
        #   "last_order_alert_time": ts | None,
        #   "last_service_alert_time": ts | None,
        # }}
        self.table_tracking = {}

        # Persistent interaction state — keyed (customer_id, waiter_id)
        # value: {"start_ts": ts, "last_near_ts": ts}
        self.ongoing_interactions = {}

        # Cleanliness state
        # {table_id: {"is_unclean": bool, "last_detected": ts, "consecutive_frames": int}}
        self.table_cleanliness = {}
        self.unclean_frames_required = 3

        # Bookkeeping
        self.frame_count = 0
        self.total_alerts = 0
        self.last_update_time = time.time()

        # GIF recorder
        self.gif_recorder = AlertGifRecorder(buffer_size=90, gif_duration=3.0, fps=5)
        self._was_recording_alert = False
        self._pending_alert_info = None

        self.load_configuration()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _get_table_display_name(self, table_id):
        if table_id in self.table_display_names:
            return self.table_display_names[table_id]
        import re
        match = re.search(r'(\d+)', str(table_id))
        return f"Table {match.group(1)}" if match else str(table_id)

    def _get_table_number(self, table_id):
        import re
        if table_id in self.table_display_names:
            return self.table_display_names[table_id]
        match = re.search(r'(\d+)', str(table_id))
        return match.group(1) if match else str(table_id)

    # ------------------------------------------------------------------
    # Configuration loading / saving (unchanged from original)
    # ------------------------------------------------------------------
    def load_configuration(self):
        try:
            self._load_table_rois_from_config()
            self._load_settings_from_config()
            logger.info(
                f"[{self.channel_id}] Service discipline config loaded: "
                f"{len(self.table_rois)} tables configured"
            )
        except Exception as e:
            logger.error(f"Failed to load service discipline config: {e}", exc_info=True)

    def _load_table_rois_from_config(self):
        config_path = Path("config/channels.json")
        if not config_path.exists():
            return
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for channel in config.get('channels', []):
            if channel.get('channel_id') != self.channel_id:
                continue
            for module in channel.get('modules', []):
                if module.get('type') != 'ServiceDisciplineMonitor':
                    continue
                table_rois_config = module.get('config', {}).get('table_rois', {})
                for table_id, roi_data in table_rois_config.items():
                    if not isinstance(roi_data, dict) or 'points' not in roi_data:
                        continue
                    polygon = []
                    for p in roi_data['points']:
                        if isinstance(p, dict) and 'x' in p and 'y' in p:
                            polygon.append((float(p['x']), float(p['y'])))
                        elif isinstance(p, (list, tuple)) and len(p) >= 2:
                            polygon.append((float(p[0]), float(p[1])))
                    if len(polygon) >= 3:
                        self.table_rois[table_id] = {
                            "polygon": polygon,
                            "bbox": (
                                min(p[0] for p in polygon),
                                min(p[1] for p in polygon),
                                max(p[0] for p in polygon),
                                max(p[1] for p in polygon),
                            ),
                        }
                        if roi_data.get('label'):
                            self.table_display_names[table_id] = roi_data['label']

    def _load_settings_from_config(self):
        config_path = Path("config/channels.json")
        if not config_path.exists():
            return
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for channel in config.get('channels', []):
            if channel.get('channel_id') != self.channel_id:
                continue
            for module in channel.get('modules', []):
                if module.get('type') != 'ServiceDisciplineMonitor':
                    continue
                self.settings.update(module.get('config', {}).get('settings', {}))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _point_in_polygon(self, point, polygon, bbox):
        x, y = point
        min_x, min_y, max_x, max_y = bbox
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, len(polygon) + 1):
            p2x, p2y = polygon[i % len(polygon)]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _bbox_iou(self, box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Uniform detection (unchanged from original)
    # ------------------------------------------------------------------
    def _check_uniform_on_person(self, person_bbox, uniform_detections):
        px1, py1, px2, py2 = person_bbox
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        person_h = max(1, py2 - py1)
        for uni in uniform_detections:
            ub = uni.get("bbox", [])
            if len(ub) != 4:
                continue
            if self._bbox_iou(person_bbox, ub) >= self.uniform_iou_threshold:
                return True
            ucx, ucy = (ub[0] + ub[2]) / 2, (ub[1] + ub[3]) / 2
            dist = math.hypot(pcx - ucx, pcy - ucy)
            if dist < self.uniform_proximity_px and dist < person_h * 0.8:
                return True
        return False

    def _crop_uniform_recheck(self, frame, person_bbox):
        try:
            h, w = frame.shape[:2]
            x1 = max(0, int(person_bbox[0])); y1 = max(0, int(person_bbox[1]))
            x2 = min(w, int(person_bbox[2])); y2 = min(h, int(person_bbox[3]))
            if x2 - x1 < 20 or y2 - y1 < 20:
                return False
            crop = frame[y1:y2, x1:x2]
            results = self.uniform_model(
                crop, conf=self.uniform_crop_conf_threshold,
                iou=self.nms_iou, verbose=False,
            )
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_name = results[0].names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name in self.server_uniform_classes:
                        return True
                    if cls_name == self.server_hairnet_class and conf >= self.hairnet_conf_threshold:
                        return True
            return False
        except Exception as e:
            logger.debug(f"[{self.channel_id}] Uniform crop recheck failed: {e}")
            return False

    def _is_person_in_uniform(self, track_id, person_bbox, uniform_detections, frame=None):
        if self._check_uniform_on_person(person_bbox, uniform_detections):
            self._track_uniform_memory[track_id] = self.frame_count
            return True
        last = self._track_uniform_memory.get(track_id)
        if last is not None:
            if (self.frame_count - last) <= self.uniform_memory_frames:
                return True
            del self._track_uniform_memory[track_id]
        if self.uniform_crop_recheck and frame is not None:
            if self._crop_uniform_recheck(frame, person_bbox):
                self._track_uniform_memory[track_id] = self.frame_count
                return True
        return False

    # ------------------------------------------------------------------
    # Table-tracking helper
    # ------------------------------------------------------------------
    def _ensure_table_tracking(self, table_id):
        if table_id not in self.table_tracking:
            self.table_tracking[table_id] = {}
        d = self.table_tracking[table_id]
        d.setdefault("customer_track_ids", [])
        d.setdefault("waiter_track_ids", [])
        d.setdefault("last_order_alert_time", None)
        d.setdefault("last_service_alert_time", None)

    def set_table_roi(self, table_id, polygon_points):
        if not polygon_points or len(polygon_points) < 3:
            return
        normalized = []
        for p in polygon_points:
            if isinstance(p, dict) and 'x' in p and 'y' in p:
                normalized.append((float(p['x']), float(p['y'])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                normalized.append((float(p[0]), float(p[1])))
        if len(normalized) < 3:
            return
        self.table_rois[table_id] = {
            "polygon": normalized,
            "bbox": (
                min(p[0] for p in normalized), min(p[1] for p in normalized),
                max(p[0] for p in normalized), max(p[1] for p in normalized),
            ),
        }
        self._ensure_table_tracking(table_id)
        logger.info(f"[{self.channel_id}] Set ROI for table {table_id}")

    # ==================================================================
    # MAIN PROCESSING PIPELINE
    # ==================================================================
    def process_frame(self, frame):
        self.frame_count += 1
        current_time = datetime.now()
        now_ts = current_time.timestamp()

        if frame is None or frame.size == 0:
            return frame

        try:
            self.gif_recorder.add_frame(frame)
            clean_frame = frame.copy()
            h, w = frame.shape[:2]

            # --- 1. Detect persons + uniforms ---
            person_detections = self.person_detector.detect_persons(frame)
            uniform_results = self.uniform_model(
                frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False,
            )

            ds_inputs = []
            uniform_detections_list = []

            for det in person_detections:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    ds_inputs.append((
                        [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                        det.get("confidence", 0.5), "person",
                    ))

            if len(uniform_results) > 0 and uniform_results[0].boxes is not None:
                names = uniform_results[0].names
                for box in uniform_results[0].boxes:
                    cls_name = names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if cls_name in self.server_uniform_classes:
                        ds_inputs.append((
                            [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            conf, "person",
                        ))
                        uniform_detections_list.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf, "class_name": cls_name,
                        })
                    elif cls_name == self.server_hairnet_class and conf >= self.hairnet_conf_threshold:
                        uniform_detections_list.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf, "class_name": cls_name,
                        })

            # --- 2. Run tracker ---
            tracks = []
            if self.tracking_enabled and self.tracker and ds_inputs:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tracks = self.tracker.update_tracks(ds_inputs, frame=frame)
                except Exception as e:
                    logger.error(f"DeepSORT error: {e}")

            # --- 3. Cleanliness ---
            self._update_table_cleanliness(uniform_results, h, w, now_ts)

            # --- 4. Classify tracks + record T_seated (with dwell time) ---
            self._process_tracked_persons(
                tracks, uniform_detections_list, current_time, frame,
            )

            # --- 5. Detect interactions ---
            self._detect_interactions(current_time)

            # --- 6. Check violations ---
            self._check_violations(current_time, clean_frame)

            # --- 7. Cleanup ---
            self._cleanup_stale_tracks(now_ts)

            if now_ts - self.last_update_time >= 1.0:
                self._send_realtime_update()
                self.last_update_time = now_ts

            # --- 8. GIF recording ---
            if self.gif_recorder.is_recording_alert:
                gif_frame = self._annotate_frame_for_gif(frame)
                self.gif_recorder.add_alert_frame(gif_frame)

            if self._was_recording_alert and not self.gif_recorder.is_recording_alert:
                self._finalize_gif_alert()

            self._was_recording_alert = self.gif_recorder.is_recording_alert

            return self._draw_annotations(frame, current_time)

        except Exception as e:
            logger.error(f"ServiceDiscipline error: {e}", exc_info=True)
            return frame

    # ------------------------------------------------------------------
    # Cleanliness (with timeout decay)
    # ------------------------------------------------------------------
    def _update_table_cleanliness(self, uniform_results, frame_h, frame_w, now_ts):
        if len(uniform_results) == 0 or uniform_results[0].boxes is None:
            self._decay_cleanliness(now_ts)
            return

        names = uniform_results[0].names
        clean_centers, unclean_centers = [], []
        for box in uniform_results[0].boxes:
            cls_name = names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2 / frame_w, (y1 + y2) / 2 / frame_h
            if cls_name == self.table_unclean_class and conf >= self.unclean_conf_threshold:
                unclean_centers.append((cx, cy))
            elif cls_name == self.table_clean_class and conf >= self.conf_threshold:
                clean_centers.append((cx, cy))

        for table_id, roi_info in self.table_rois.items():
            polygon, bbox_norm = roi_info.get("polygon"), roi_info.get("bbox")
            if not polygon or not bbox_norm:
                continue
            if table_id not in self.table_cleanliness:
                self.table_cleanliness[table_id] = {
                    "is_unclean": False, "last_detected": None, "consecutive_frames": 0,
                }
            state = self.table_cleanliness[table_id]
            found_unclean = any(
                self._point_in_polygon((cx, cy), polygon, bbox_norm)
                for cx, cy in unclean_centers
            )
            found_clean = any(
                self._point_in_polygon((cx, cy), polygon, bbox_norm)
                for cx, cy in clean_centers
            )
            if found_unclean:
                state["consecutive_frames"] += 1
                state["last_detected"] = now_ts
                if state["consecutive_frames"] >= self.unclean_frames_required:
                    state["is_unclean"] = True
            elif found_clean:
                state["is_unclean"] = False
                state["consecutive_frames"] = 0
                state["last_detected"] = now_ts

        self._decay_cleanliness(now_ts)

    def _decay_cleanliness(self, now_ts):
        """Expire 'unclean' state if not re-detected within timeout."""
        timeout = self.settings["cleanliness_timeout_seconds"]
        for tid, state in self.table_cleanliness.items():
            if state["is_unclean"] and state["last_detected"] is not None:
                if (now_ts - state["last_detected"]) > timeout:
                    state["is_unclean"] = False
                    state["consecutive_frames"] = 0
                    logger.info(f"[{self.channel_id}] Table {tid}: cleanliness state expired")

    def _is_table_unclean(self, table_id):
        st = self.table_cleanliness.get(table_id)
        return bool(st and st["is_unclean"])

    # ------------------------------------------------------------------
    # Track classification + T_seated (with dwell time)
    # ------------------------------------------------------------------
    def _process_tracked_persons(self, tracks, uniform_detections, current_time, frame):
        h, w = frame.shape[:2]
        now_ts = current_time.timestamp()
        dwell_required = self.settings["seated_dwell_seconds"]

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_tlbr()
            tbbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            tcenter = [(tbbox[0] + tbbox[2]) / 2, (tbbox[1] + tbbox[3]) / 2]

            # ROI test using feet (bottom-center, normalized)
            feet = (tcenter[0] / w, tbbox[3] / h)
            table_id = None
            for tid, roi_info in self.table_rois.items():
                polygon, bbox_norm = roi_info.get("polygon"), roi_info.get("bbox")
                if polygon and bbox_norm and self._point_in_polygon(feet, polygon, bbox_norm):
                    table_id = tid
                    break

            if table_id:
                self._ensure_table_tracking(table_id)

            is_waiter = self._is_person_in_uniform(
                track_id, tbbox, uniform_detections,
                frame=frame if table_id else None,
            )

            # ---- New track ----
            if track_id not in self.person_tracks:
                ptype = "waiter" if is_waiter else "customer"
                self.person_tracks[track_id] = {
                    "type": ptype, "table_id": table_id,
                    "center": tcenter, "bbox": tbbox,
                    "first_seen_in_roi": now_ts if table_id else None,
                    "T_seated": None,
                    "T_order_start": None, "T_order_end": None, "T_food_served": None,
                    "last_seen": now_ts,
                    "order_wait_time": None, "service_wait_time": None,
                    "interaction_history": [],
                    "waiter_at_table": False,
                    "waiter_visit_count": 0,
                    "order_alert_fired": False,
                    "service_alert_fired": False,
                }
                if ptype == "waiter" and table_id:
                    if track_id not in self.table_tracking[table_id]["waiter_track_ids"]:
                        self.table_tracking[table_id]["waiter_track_ids"].append(track_id)
                # NOTE: do NOT set T_seated yet — wait for dwell time.
                continue

            # ---- Existing track update ----
            p = self.person_tracks[track_id]
            p["center"] = tcenter
            p["bbox"] = tbbox
            p["last_seen"] = now_ts

            # Track ROI dwell time
            if table_id:
                if p.get("first_seen_in_roi") is None or p.get("table_id") != table_id:
                    p["first_seen_in_roi"] = now_ts
            else:
                p["first_seen_in_roi"] = None

            # ---- Reclassification logic (does NOT wipe customer events) ----
            if is_waiter and p["type"] == "customer":
                # Promote to waiter, but keep timestamps so a brief uniform
                # flicker on a real customer doesn't reset their wait timer.
                p["type"] = "waiter"
                old_table = p.get("table_id")
                if old_table and old_table in self.table_tracking:
                    if track_id in self.table_tracking[old_table]["customer_track_ids"]:
                        self.table_tracking[old_table]["customer_track_ids"].remove(track_id)
                if table_id:
                    if track_id not in self.table_tracking[table_id]["waiter_track_ids"]:
                        self.table_tracking[table_id]["waiter_track_ids"].append(track_id)

            elif not is_waiter and p["type"] == "waiter" \
                    and track_id not in self._track_uniform_memory:
                # Memory expired → demote to customer
                p["type"] = "customer"
                old_table = p.get("table_id")
                if old_table and old_table in self.table_tracking:
                    if track_id in self.table_tracking[old_table]["waiter_track_ids"]:
                        self.table_tracking[old_table]["waiter_track_ids"].remove(track_id)
                    if track_id not in self.table_tracking[old_table]["customer_track_ids"]:
                        self.table_tracking[old_table]["customer_track_ids"].append(track_id)

            # Update table_id if changed
            if table_id and p["table_id"] != table_id:
                old_table = p["table_id"]
                if old_table and old_table in self.table_tracking:
                    if p["type"] == "customer":
                        if track_id in self.table_tracking[old_table]["customer_track_ids"]:
                            self.table_tracking[old_table]["customer_track_ids"].remove(track_id)
                    else:
                        if track_id in self.table_tracking[old_table]["waiter_track_ids"]:
                            self.table_tracking[old_table]["waiter_track_ids"].remove(track_id)
                p["table_id"] = table_id

            # ---- T_seated: only after dwell-time confirmation ----
            if (p["type"] == "customer" and table_id and p["T_seated"] is None
                    and p.get("first_seen_in_roi") is not None
                    and (now_ts - p["first_seen_in_roi"]) >= dwell_required):
                p["T_seated"] = now_ts
                if track_id not in self.table_tracking[table_id]["customer_track_ids"]:
                    self.table_tracking[table_id]["customer_track_ids"].append(track_id)
                logger.info(
                    f"[{self.channel_id}] T_seated: customer {track_id} at table {table_id} "
                    f"(after {dwell_required:.1f}s dwell)"
                )

    # ------------------------------------------------------------------
    # Interaction detection
    # ------------------------------------------------------------------
    def _detect_interactions(self, current_time):
        now_ts = current_time.timestamp()
        duration_required = self.settings["interaction_duration"]
        food_gap = self.settings["food_served_gap"]
        grace = self.settings["interaction_grace_seconds"]
        dist_factor = self.settings["interaction_distance_factor"]
        dist_min = self.settings["interaction_distance_min_px"]

        active_keys = set()

        for table_id, info in self.table_tracking.items():
            customer_ids = info.get("customer_track_ids", [])
            waiter_ids = info.get("waiter_track_ids", [])

            for cid in customer_ids:
                cust = self.person_tracks.get(cid)
                if not cust or cust["type"] != "customer" or cust["T_seated"] is None:
                    continue

                cc = cust["center"]
                cbbox = cust.get("bbox", [])
                ch = max(40, cbbox[3] - cbbox[1]) if len(cbbox) == 4 else 100
                near_threshold = max(dist_min, dist_factor * ch)

                waiter_currently_near = False

                for wid in waiter_ids:
                    waiter = self.person_tracks.get(wid)
                    if not waiter or waiter["type"] != "waiter":
                        continue
                    wc = waiter["center"]
                    distance = math.hypot(cc[0] - wc[0], cc[1] - wc[1])
                    key = (cid, wid)

                    if distance < near_threshold:
                        waiter_currently_near = True
                        active_keys.add(key)
                        if key not in self.ongoing_interactions:
                            self.ongoing_interactions[key] = {
                                "start_ts": now_ts, "last_near_ts": now_ts,
                            }
                        else:
                            self.ongoing_interactions[key]["last_near_ts"] = now_ts

                        elapsed = now_ts - self.ongoing_interactions[key]["start_ts"]
                        if elapsed < duration_required:
                            continue

                        # ---- T_order_start: first valid interaction ----
                        if cust["T_order_start"] is None:
                            cust["T_order_start"] = now_ts
                            cust["order_wait_time"] = now_ts - cust["T_seated"]
                            cust["waiter_at_table"] = True
                            cust["waiter_visit_count"] = 1
                            cust["interaction_history"].append((wid, now_ts, "order"))
                            logger.info(
                                f"[{self.channel_id}] T_order_start: table {table_id} "
                                f"(order wait {cust['order_wait_time']:.1f}s)"
                            )

                        # ---- T_food_served: subsequent visit, after gap ----
                        # FIX: dropped requirement that T_order_end be set, and
                        # changed the broken `or` to `and` in the gating.
                        elif (cust["T_food_served"] is None
                              and cust["waiter_visit_count"] >= 2
                              and (now_ts - cust["T_order_start"]) >= food_gap):
                            cust["T_food_served"] = now_ts
                            cust["service_wait_time"] = now_ts - cust["T_order_start"]
                            cust["interaction_history"].append((wid, now_ts, "food"))
                            logger.info(
                                f"[{self.channel_id}] T_food_served: table {table_id} "
                                f"(service wait {cust['service_wait_time']:.1f}s)"
                            )

                # ---- Detect waiter-leaving (with grace period) ----
                # Only treat as "left" if waiter has been gone longer than the grace window
                if cust["T_order_start"] is not None and cust["T_order_end"] is None:
                    # Find any waiter who was near recently
                    any_recent = False
                    for wid in waiter_ids:
                        key = (cid, wid)
                        last_near = self.ongoing_interactions.get(key, {}).get("last_near_ts")
                        if last_near is not None and (now_ts - last_near) <= grace:
                            any_recent = True
                            break
                    if not any_recent and cust.get("waiter_at_table", False):
                        cust["T_order_end"] = now_ts
                        cust["waiter_at_table"] = False
                        logger.info(
                            f"[{self.channel_id}] T_order_end: waiter left table {table_id}"
                        )

                # ---- Track repeat visits (for service-served gating) ----
                if waiter_currently_near and not cust.get("waiter_at_table", False):
                    cust["waiter_at_table"] = True
                    cust["waiter_visit_count"] = cust.get("waiter_visit_count", 0) + 1

        # Sweep stale interactions (with grace period)
        stale = []
        for key, st in self.ongoing_interactions.items():
            if key not in active_keys and (now_ts - st["last_near_ts"]) > grace:
                stale.append(key)
        for k in stale:
            del self.ongoing_interactions[k]

    # ------------------------------------------------------------------
    # Violation checks (TIME-BASED + EVENT-BASED, separate cooldowns)
    # ------------------------------------------------------------------
    def _check_violations(self, current_time, frame=None):
        now_ts = current_time.timestamp()
        order_th = self.settings["order_wait_threshold"]
        service_th = self.settings["service_wait_threshold"]
        cooldown = self.settings["alert_cooldown"]

        for table_id, info in self.table_tracking.items():
            if self._is_table_unclean(table_id):
                continue

            for cid in info.get("customer_track_ids", []):
                cust = self.person_tracks.get(cid)
                if not cust or cust["T_seated"] is None:
                    continue

                # ===== ORDER WAIT VIOLATION =====
                # Fires if order hasn't been taken AND elapsed > threshold,
                # OR when order is finally taken but already over threshold.
                if not cust.get("order_alert_fired", False):
                    if cust["T_order_start"] is None:
                        elapsed = now_ts - cust["T_seated"]
                    else:
                        elapsed = cust["T_order_start"] - cust["T_seated"]

                    if elapsed > order_th:
                        last = info.get("last_order_alert_time")
                        if last is None or (now_ts - last) > cooldown:
                            self._trigger_violation_alert(
                                table_id, cid, "order_wait", elapsed, current_time, frame,
                            )
                            info["last_order_alert_time"] = now_ts
                            cust["order_alert_fired"] = True

                # ===== SERVICE WAIT VIOLATION =====
                # Only meaningful once order is taken.
                if (cust["T_order_start"] is not None
                        and not cust.get("service_alert_fired", False)):
                    if cust["T_food_served"] is None:
                        elapsed = now_ts - cust["T_order_start"]
                    else:
                        elapsed = cust["T_food_served"] - cust["T_order_start"]

                    if elapsed > service_th:
                        last = info.get("last_service_alert_time")
                        if last is None or (now_ts - last) > cooldown:
                            self._trigger_violation_alert(
                                table_id, cid, "service_wait", elapsed, current_time, frame,
                            )
                            info["last_service_alert_time"] = now_ts
                            cust["service_alert_fired"] = True

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def _trigger_violation_alert(self, table_id, cid, vtype, wait_time, current_time, frame=None):
        if self.db_manager and self.app:
            try:
                with self.app.app_context():
                    if not self.db_manager.is_within_operation_hours(self.channel_id):
                        return
            except Exception:
                pass

        cust = self.person_tracks.get(cid)
        if not cust:
            return

        table_name = self._get_table_display_name(table_id)
        table_number = self._get_table_number(table_id)

        alert_info = {
            "type": "service_discipline_alert",
            "table_id": table_id, "table_name": table_name, "table_number": table_number,
            "violation_type": vtype, "wait_time": wait_time,
            "channel_id": self.channel_id,
            "timestamp": current_time.isoformat(),
            "T_seated": cust.get("T_seated"),
            "T_order_start": cust.get("T_order_start"),
            "T_order_end": cust.get("T_order_end"),
            "T_food_served": cust.get("T_food_served"),
        }

        if not self.gif_recorder.is_recording_alert:
            self.gif_recorder.start_alert_recording(alert_info)
            self._pending_alert_info = alert_info

        self.total_alerts += 1
        logger.warning(
            f"[{self.channel_id}] {vtype} ALERT: {table_name} = {wait_time:.1f}s"
        )

        if self.socketio:
            self.socketio.emit("service_discipline_alert", {
                **alert_info, "wait_time": round(wait_time, 1),
            })

    def _finalize_gif_alert(self):
        gif_info = self.gif_recorder.get_last_gif_info()
        if not (gif_info and self.db_manager and self._pending_alert_info):
            return
        try:
            gif_path = gif_info.get('gif_path', '')
            gif_filename = os.path.basename(gif_path) if gif_path else \
                f"service_alert_{self.channel_id}_{datetime.now():%Y%m%d_%H%M%S}.gif"
            payload = {
                'gif_filename': gif_filename, 'gif_path': gif_path,
                'frame_count': gif_info.get('frame_count', 0),
                'duration': gif_info.get('duration', 0.0),
            }
            ai = self._pending_alert_info
            msg = (f"Service discipline violation: {ai['table_name']} "
                   f"(#{ai['table_number']}) {ai['violation_type']} = {ai['wait_time']:.1f}s")
            if self.app:
                with self.app.app_context():
                    self.db_manager.save_alert_gif(
                        self.channel_id, 'service_discipline_alert',
                        payload, alert_message=msg, alert_data=ai,
                    )
            self._pending_alert_info = None
        except Exception as e:
            logger.error(f"[{self.channel_id}] Failed to save alert GIF: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup_stale_tracks(self, now_ts):
        timeout = self.settings["track_timeout"]
        to_remove = [tid for tid, t in self.person_tracks.items()
                     if (now_ts - t["last_seen"]) > timeout]
        for tid in to_remove:
            t = self.person_tracks[tid]
            tbl = t.get("table_id")
            if tbl and tbl in self.table_tracking:
                if t["type"] == "customer":
                    if tid in self.table_tracking[tbl]["customer_track_ids"]:
                        self.table_tracking[tbl]["customer_track_ids"].remove(tid)
                else:
                    if tid in self.table_tracking[tbl]["waiter_track_ids"]:
                        self.table_tracking[tbl]["waiter_track_ids"].remove(tid)
            self._track_uniform_memory.pop(tid, None)
            for k in [k for k in list(self.ongoing_interactions) if tid in k]:
                del self.ongoing_interactions[k]
            del self.person_tracks[tid]

    # ------------------------------------------------------------------
    # Annotations (simplified — kept the new event-based one only)
    # ------------------------------------------------------------------
    def _draw_annotations(self, frame, current_time):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        if not self.table_rois:
            cv2.putText(annotated, "No table ROIs configured", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        now_ts = current_time.timestamp()

        for table_id, roi_info in self.table_rois.items():
            polygon = roi_info.get("polygon", [])
            if len(polygon) < 3:
                continue
            poly_px = [(int(p[0] * w), int(p[1] * h)) for p in polygon]
            cv2.polylines(annotated, [np.array(poly_px, np.int32)], True, (0, 255, 255), 3)
            label_pos = poly_px[0]
            cv2.putText(annotated, f"Table {table_id}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            info = self.table_tracking.get(table_id, {})
            for cid in info.get("customer_track_ids", []):
                cust = self.person_tracks.get(cid)
                if not cust:
                    continue
                bbox = cust.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                if cust.get("T_food_served"):
                    color = (0, 255, 0)
                elif cust.get("T_order_start"):
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                txt = f"ID:{cid}"
                if cust.get("T_seated"):
                    ow = (cust.get("T_order_start") or now_ts) - cust["T_seated"]
                    txt += f" | Order:{ow:.0f}s"
                if cust.get("T_order_start"):
                    sw = (cust.get("T_food_served") or now_ts) - cust["T_order_start"]
                    txt += f" | Service:{sw:.0f}s"
                cv2.putText(annotated, txt, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for wid in info.get("waiter_track_ids", []):
                w_obj = self.person_tracks.get(wid)
                if not w_obj:
                    continue
                bbox = w_obj.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(annotated, f"Waiter ID:{wid}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return annotated

    def _annotate_frame_for_gif(self, frame):
        if not self._pending_alert_info:
            return frame
        tbl = self._pending_alert_info.get('table_id')
        if not tbl or tbl not in self.table_rois:
            return frame
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        polygon = self.table_rois[tbl].get("polygon", [])
        if len(polygon) < 3:
            return annotated
        poly_px = [(int(p[0] * w), int(p[1] * h)) for p in polygon]
        pts = np.array(poly_px, np.int32)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 180))
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
        cv2.polylines(annotated, [pts], True, (0, 0, 255), 3)
        cx = int(np.mean([p[0] for p in poly_px]))
        cy = int(np.mean([p[1] for p in poly_px]))
        ai = self._pending_alert_info
        label = ai.get('table_name', f"Table {tbl}")
        cv2.putText(annotated, label, (cx - 60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        info_label = f"{ai.get('violation_type', '')}: {ai.get('wait_time', 0):.0f}s"
        cv2.putText(annotated, info_label, (cx - 80, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return annotated

    # ------------------------------------------------------------------
    # Status / dashboard helpers
    # ------------------------------------------------------------------
    def _send_realtime_update(self):
        if not self.socketio:
            return
        waiting = 0
        for info in self.table_tracking.values():
            for cid in info.get("customer_track_ids", []):
                c = self.person_tracks.get(cid)
                if c and c.get("T_seated") and c.get("T_order_start") is None:
                    waiting += 1
                    break
        try:
            self.socketio.emit('service_discipline_update', {
                'channel_id': self.channel_id,
                'waiting_tables': waiting,
                'total_alerts': self.total_alerts,
                'tables_configured': len(self.table_rois),
                'timestamp': datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"realtime update error: {e}")

    def get_current_status(self):
        waiting = 0
        for info in self.table_tracking.values():
            for cid in info.get("customer_track_ids", []):
                c = self.person_tracks.get(cid)
                if c and c.get("T_seated") and c.get("T_order_start") is None:
                    waiting += 1
                    break
        return {
            "waiting_tables": waiting,
            "total_alerts": self.total_alerts,
            "tables_configured": len(self.table_rois),
        }