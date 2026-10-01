from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from engine.config_loader import ROIZone, TripwireRule
from engine.geometry import bbox_bottom_center, bbox_iou, point_in_polygon, scale_points
from engine.parking.base import SlotState
from engine.parking.spatial import bbox_ios, bbox_polygon_overlap_ratio
from engine.tracker_interface import TrackResult


class CarSlotTracker:
    DEFAULT_VEHICLE_CLASSES = {"car", "truck", "bus"}

    def __init__(
        self,
        total_slots: Optional[int] = None,
        dwell_threshold_sec: float = 10.0,
        vehicle_classes: Optional[Set[str]] = None,
        acquisition_conf_thresh: float = 0.32,
        retention_conf_thresh: float = 0.20,
        wheel_contact_margin_px: float = 0.0,
        corridor_obstruction_dwell_sec: float = 60.0,
        corridor_shift_threshold_px: float = 15.0,
        debug_diagnostics: bool = False,
        debug_target_zone: Optional[str] = None,
        debug_log_path: str = "logs/s4_diagnostics.log",
        debug_snapshots: bool = False,
        debug_snapshot_max_files: int = 300,
        debug_snapshot_min_interval_s: float = 10.0,
        debug_snapshot_dir: str = "logs/snapshots",
    ) -> None:
        self._total_slots_override = total_slots
        self.dwell_threshold_sec = dwell_threshold_sec
        self.vehicle_classes = set(vehicle_classes) if vehicle_classes is not None else set(self.DEFAULT_VEHICLE_CLASSES)

        self.acquisition_conf_thresh = float(acquisition_conf_thresh)
        self.retention_conf_thresh = float(retention_conf_thresh)
        self.wheel_contact_margin_px = float(wheel_contact_margin_px)

        self.corridor_obstruction_dwell_sec = float(corridor_obstruction_dwell_sec)
        self.corridor_shift_threshold_px = float(corridor_shift_threshold_px)
        self._corridor_vehicles: Dict[int, Dict[str, Any]] = {}

        self.slot_states: Dict[str, SlotState] = {}
        self._last_candidates: List[Tuple[float, str, int]] = []
        self._last_assigned_slots: Dict[str, int] = {}
        self.gate_in_count: int = 0
        self.gate_out_count: int = 0
        self.tripwire_flash: Dict[str, float] = {}

        # Diagnostics & Snapshot Worker
        self.debug_diagnostics = debug_diagnostics
        self.debug_target_zone = debug_target_zone
        self.debug_log_path = debug_log_path
        self.debug_snapshots = debug_snapshots
        self.debug_snapshot_max_files = max(1, int(debug_snapshot_max_files))
        self.debug_snapshot_min_interval_s = max(0.0, float(debug_snapshot_min_interval_s))
        self.debug_snapshot_dir = debug_snapshot_dir

        self._snapshot_queue: Optional[queue.Queue] = None
        self._snapshot_worker: Optional[threading.Thread] = None
        self._snapshot_stop_event = threading.Event()
        self._last_truck_snapshot_time: float = 0.0
        self._existing_snapshot_files: List[Tuple[int, str, str]] = []
        self._snapshot_lock = threading.Lock()

        self._debug_logger: Optional[Any] = None
        self._debug_logger_failed: bool = False
        self._last_diag_log_time: Dict[int, float] = {}
        self._prev_slot_phases: Dict[str, str] = {}
        self._has_logged_diag_error: bool = False

    @property
    def total_slots(self) -> int:
        if self._total_slots_override is not None:
            return self._total_slots_override
        return len(self.slot_states)

    @property
    def occupied_slots(self) -> int:
        return sum(1 for s in self.slot_states.values() if s.occupied)

    @property
    def available_slots(self) -> int:
        return max(0, self.total_slots - self.occupied_slots)

    @property
    def has_obstruction(self) -> bool:
        return any(v.get("is_obstruction", False) for v in self._corridor_vehicles.values())

    @property
    def active_obstructions(self) -> List[Dict[str, Any]]:
        return [
            {
                "track_id": tid,
                "bbox": v["bbox"],
                "dwell_sec": round(v["dwell_duration"], 1),
                "class_label": v.get("class_label", "car"),
            }
            for tid, v in self._corridor_vehicles.items()
            if v.get("is_obstruction", False)
        ]

    def _get_debug_logger(self) -> Optional[Any]:
        if not self.debug_diagnostics or not self.debug_target_zone:
            return None
        if self._debug_logger is not None:
            return self._debug_logger
        if self._debug_logger_failed:
            return None
        try:
            log_file = Path(self.debug_log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            logger_name = f"diagnostics_{self.debug_target_zone}_{abs(hash(str(log_file.resolve())))}"
            diag_logger = logging.getLogger(logger_name)
            diag_logger.setLevel(logging.INFO)
            diag_logger.propagate = False

            if not diag_logger.handlers:
                rfh = RotatingFileHandler(
                    str(log_file),
                    maxBytes=5 * 1024 * 1024,
                    backupCount=10,
                    encoding="utf-8",
                )
                rfh.setFormatter(logging.Formatter("%(message)s"))
                diag_logger.addHandler(rfh)

            self._debug_logger = diag_logger
            return self._debug_logger
        except Exception as exc:
            self._debug_logger_failed = True
            if not self._has_logged_diag_error:
                self._has_logged_diag_error = True
                try:
                    logging.getLogger("engine.parking.car_slot").warning(
                        f"Diagnostic logger initialization failed, disabling retries: {exc}"
                    )
                except Exception:
                    pass
            return None

    def _emit_diagnostic_log(self, payload: Dict[str, Any]) -> None:
        try:
            diag_logger = self._get_debug_logger()
            if diag_logger is None:
                return
            diag_logger.info(json.dumps(payload, separators=(",", ":")))
        except Exception as exc:
            if not self._has_logged_diag_error:
                self._has_logged_diag_error = True
                try:
                    logging.getLogger("engine.parking.car_slot").warning(
                        f"Diagnostic logging exception swallowed: {exc}"
                    )
                except Exception:
                    pass

    def _ensure_snapshot_worker(self) -> bool:
        if not (self.debug_diagnostics and self.debug_snapshots and self.debug_target_zone):
            return False
        with self._snapshot_lock:
            if self._snapshot_queue is not None:
                return True
            try:
                snap_dir = os.path.abspath(self.debug_snapshot_dir)
                os.makedirs(snap_dir, exist_ok=True)

                self._snapshot_queue = queue.Queue(maxsize=20)

                existing: List[Tuple[int, str, str]] = []
                if os.path.exists(snap_dir):
                    for fname in os.listdir(snap_dir):
                        if fname.startswith("debug_s4_") and fname.endswith(".jpg"):
                            parts = fname[:-4].split("_")
                            if len(parts) >= 3:
                                try:
                                    ep = int(parts[2])
                                    ev = parts[3] if len(parts) >= 4 else "unknown"
                                    existing.append((ep, fname, ev))
                                except Exception:
                                    pass
                existing.sort(key=lambda x: x[0])
                self._existing_snapshot_files = existing

                self._snapshot_stop_event.clear()
                self._snapshot_worker = threading.Thread(
                    target=self._snapshot_worker_loop,
                    name="S4SnapshotWorker",
                    daemon=True,
                )
                self._snapshot_worker.start()
                return True
            except Exception as exc:
                if not self._has_logged_diag_error:
                    self._has_logged_diag_error = True
                    try:
                        logging.getLogger("engine.parking.car_slot").warning(
                            f"Snapshot worker creation failed: {exc}"
                        )
                    except Exception:
                        pass
                return False

    def _snapshot_worker_loop(self) -> None:
        while not self._snapshot_stop_event.is_set():
            try:
                item = self._snapshot_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._snapshot_queue.task_done()
                break

            filename, frame, epoch_ms, event_type = item
            try:
                file_path = os.path.join(self.debug_snapshot_dir, filename)
                cv2.imwrite(file_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                with self._snapshot_lock:
                    self._existing_snapshot_files.append((epoch_ms, filename, event_type))
                    max_files = self.debug_snapshot_max_files
                    while len(self._existing_snapshot_files) > max_files:
                        detect_files = [x for x in self._existing_snapshot_files if x[2] == "detect"]
                        phase_files = [x for x in self._existing_snapshot_files if x[2] == "phase"]

                        if len(detect_files) > 200:
                            to_remove = detect_files[0]
                            self._existing_snapshot_files.remove(to_remove)
                        elif len(phase_files) > 100:
                            to_remove = phase_files[0]
                            self._existing_snapshot_files.remove(to_remove)
                        else:
                            to_remove = self._existing_snapshot_files.pop(0)

                        del_path = os.path.join(self.debug_snapshot_dir, to_remove[1])
                        try:
                            if os.path.exists(del_path):
                                os.remove(del_path)
                        except Exception:
                            pass
            except Exception as exc:
                if not self._has_logged_diag_error:
                    self._has_logged_diag_error = True
                    try:
                        logging.getLogger("engine.parking.car_slot").warning(
                            f"Snapshot worker write exception: {exc}"
                        )
                    except Exception:
                        pass
            finally:
                self._snapshot_queue.task_done()

    def _trigger_snapshot(
        self,
        raw_frame: Optional[np.ndarray],
        current_time: float,
        event_type: str,
    ) -> Optional[str]:
        if not (self.debug_diagnostics and self.debug_snapshots and self.debug_target_zone):
            return None
        if raw_frame is None:
            return None
        if not self._ensure_snapshot_worker():
            return "dropped"

        epoch_ms = int(current_time * 1000)
        filename = f"debug_s4_{epoch_ms}_{event_type}.jpg"
        try:
            self._snapshot_queue.put_nowait((filename, raw_frame.copy(), epoch_ms, event_type))
            return filename
        except queue.Full:
            return "dropped"
        except Exception:
            return "dropped"

    def close(self) -> None:
        if hasattr(self, "_snapshot_stop_event") and self._snapshot_stop_event is not None:
            self._snapshot_stop_event.set()
        if hasattr(self, "_snapshot_queue") and self._snapshot_queue is not None:
            try:
                self._snapshot_queue.put_nowait(None)
            except Exception:
                pass
        if hasattr(self, "_snapshot_worker") and self._snapshot_worker is not None and self._snapshot_worker.is_alive():
            self._snapshot_worker.join(timeout=1.0)
        if getattr(self, "_debug_logger", None) is not None:
            try:
                for h in list(self._debug_logger.handlers):
                    h.flush()
                    h.close()
                    self._debug_logger.removeHandler(h)
            except Exception:
                pass

    def apply_tripwire_signal(
        self,
        slot_id: str,
        direction: str,
        track_id: int,
        current_time: float,
    ) -> None:
        if slot_id not in self.slot_states:
            slot_idx_str = slot_id.split("_")[-1]
            try:
                slot_num = int(slot_idx_str)
            except ValueError:
                slot_num = len(self.slot_states) + 1
            self.slot_states[slot_id] = SlotState(
                slot_id=slot_id,
                slot_num=slot_num,
                slot_idx_str=slot_idx_str,
                label=f"Slot {slot_num}",
            )

        state = self.slot_states[slot_id]

        if direction == "A_TO_B":
            if state.phase == "VACANT":
                state.phase = "ENTERING"
                state.tripwire_crossed_in_time = current_time
                state.track_id = track_id
        elif direction == "B_TO_A":
            if state.phase == "ENTERING":
                state.phase = "VACANT"
                state.track_id = None
                state.tripwire_crossed_in_time = None
                state.first_seen_time = None
                state.dwell_duration = 0.0
            elif state.phase == "OCCUPIED":
                state.phase = "LEAVING"
                state.leaving_since = current_time

    def record_gate_crossing(
        self,
        direction_or_note: str,
        tripwire_id: str = "",
        timestamp: float = 0.0,
    ) -> None:
        text = str(direction_or_note).upper()
        if (
            "B_TO_A" in text
            or "B TO A" in text
            or " OUT" in text
            or text.endswith("OUT")
            or text == "OUT"
        ):
            self.gate_out_count += 1
        elif (
            "A_TO_B" in text
            or "A TO B" in text
            or " IN" in text
            or text.endswith("IN")
            or text == "IN"
        ):
            self.gate_in_count += 1
        else:
            self.gate_in_count += 1

        if tripwire_id and timestamp > 0:
            self.tripwire_flash[tripwire_id] = timestamp + 1.0

    def _is_same_vehicle(
        self,
        state: "SlotState",
        new_track: TrackResult,
        pts_scaled: np.ndarray,
        current_time: float,
        grace_period_sec: float = 3.0,
        iou_threshold: float = 0.35,
    ) -> bool:
        effective_grace = 15.0 if (state.latch_occupied or getattr(state, "dwell_duration", 0.0) >= 10.0) else grace_period_sec
        if state.last_seen_time > 0:
            elapsed = current_time - state.last_seen_time
            if elapsed > effective_grace:
                return False

        x1, y1, x2, y2 = new_track.bbox
        max_pt_y = float(np.max(pts_scaled[:, 1])) if len(pts_scaled) > 0 else 0.0
        y2_ai = (y2 * (360.0 / 1080.0)) if (max_pt_y > 360.0 or y2 > 360.0) else y2
        if y2_ai >= 225.0:
            return False
        cx = float((x1 + x2) / 2.0)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        p_center = (cx, float(y2))
        p_inset = (cx, float(y2 - bh * 0.05))
        p_left = (float(x1 + bw * 0.22), float(y2 - bh * 0.04))
        p_right = (float(x2 - bw * 0.22), float(y2 - bh * 0.04))
        p_axle = (cx, float(y2 - bh * 0.12))

        d_ground = max(
            cv2.pointPolygonTest(pts_scaled, p_center, True),
            cv2.pointPolygonTest(pts_scaled, p_inset, True),
            cv2.pointPolygonTest(pts_scaled, p_left, True),
            cv2.pointPolygonTest(pts_scaled, p_right, True),
            cv2.pointPolygonTest(pts_scaled, p_axle, True),
        )

        lower_bbox = (x1, y1 + bh * 0.5, x2, y2)
        lower_overlap = bbox_polygon_overlap_ratio(lower_bbox, pts_scaled)

        if getattr(state, "slot_id", None) == "zone_08":
            max_x = float(np.max(pts_scaled[:, 0])) if len(pts_scaled) > 0 else 0.0
            if (max_x > 1000.0 and cx > 1810.0) or (max_x <= 1000.0 and cx > 603.3):
                return False

        if d_ground >= -12.0 or lower_overlap >= 0.20:
            return True
        if state.last_bbox is not None:
            iou = bbox_iou(new_track.bbox, state.last_bbox)
            if iou >= iou_threshold:
                return True

        return False

    def update(
        self,
        tracks: List[TrackResult],
        polygons: List[ROIZone],
        scale_x: float,
        scale_y: float,
        current_time: float,
        is_warmup: bool = False,
        frame_count: Optional[int] = None,
        raw_ai_frame: Optional[np.ndarray] = None,
        raw_detections: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        active_slots = [p for p in polygons if p.active]
        total_capacity = self._total_slots_override if self._total_slots_override is not None else len(polygons)

        vehicle_tracks = [
            t
            for t in tracks
            if (t.is_confirmed or is_warmup) and (t.class_label in self.vehicle_classes)
        ]

        slot_geometries: Dict[str, Dict[str, Any]] = {}
        for idx, slot in enumerate(active_slots):
            slot_id = slot.zone_id
            slot_idx_str = slot_id.split("_")[-1]
            try:
                slot_num = int(slot_idx_str)
            except ValueError:
                slot_num = idx + 1

            if slot_id not in self.slot_states:
                self.slot_states[slot_id] = SlotState(
                    slot_id=slot_id,
                    slot_num=slot_num,
                    slot_idx_str=slot_idx_str,
                    label=slot.label or f"Slot {slot_num}",
                )

            pts_scaled = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )
            M = cv2.moments(pts_scaled)
            area = M["m00"]
            slot_cx = M["m10"] / (area + 1e-5) if area > 0 else float(np.mean(pts_scaled[:, 0]))
            slot_cy = M["m01"] / (area + 1e-5) if area > 0 else float(np.mean(pts_scaled[:, 1]))
            rx, ry, rw, rh = cv2.boundingRect(pts_scaled)
            slot_bbox = (float(rx), float(ry), float(rx + rw), float(ry + rh))

            slot_geometries[slot_id] = {
                "pts_scaled": pts_scaled,
                "cx": slot_cx,
                "cy": slot_cy,
                "diag": max(10.0, math.hypot(rw, rh)),
                "bbox": slot_bbox,
            }

        frame_snapshot_status: Optional[str] = None
        frame_snapshot_triggered: bool = False

        if (
            self.debug_diagnostics
            and self.debug_snapshots
            and self.debug_target_zone
            and (self.debug_target_zone in slot_geometries)
        ):
            target_geom = slot_geometries[self.debug_target_zone]
            t_pts = target_geom["pts_scaled"]
            t_sbbox = target_geom["bbox"]

            candidate_large_objs: List[Any] = []
            if raw_detections is not None:
                candidate_large_objs = [
                    d for d in raw_detections
                    if getattr(d, "class_label", None) in ("truck", "bus")
                ]
            else:
                candidate_large_objs = [
                    t for t in tracks
                    if getattr(t, "class_label", None) in ("truck", "bus")
                ]

            for l_obj in candidate_large_objs:
                bx1, by1, bx2, by2 = getattr(l_obj, "bbox")
                bcx = float((bx1 + bx2) / 2.0)
                bcy = float((by1 + by2) / 2.0)
                b_wheel = (bcx, float(by2))
                b_lower = (bcx, float(by1 + (by2 - by1) * 0.75))
                b_center = (bcx, bcy)

                d_w = cv2.pointPolygonTest(t_pts, b_wheel, True)
                d_l = cv2.pointPolygonTest(t_pts, b_lower, True)
                d_c = cv2.pointPolygonTest(t_pts, b_center, True)
                det_d_max = max(d_w, d_l, d_c)

                boxes_overlap = not (
                    bx2 < t_sbbox[0]
                    or bx1 > t_sbbox[2]
                    or by2 < t_sbbox[1]
                    or by1 > t_sbbox[3]
                )
                is_target_overlap = (-40.0 <= det_d_max <= 10.0) or boxes_overlap
                if is_target_overlap:
                    if (current_time - self._last_truck_snapshot_time) >= self.debug_snapshot_min_interval_s:
                        self._last_truck_snapshot_time = current_time
                        frame_snapshot_status = self._trigger_snapshot(raw_ai_frame, current_time, "detect")
                        frame_snapshot_triggered = True
                    break

        candidates: List[Tuple[float, str, TrackResult]] = []

        if self._last_diag_log_time and len(self._last_diag_log_time) > 20:
            cutoff = current_time - 60.0
            self._last_diag_log_time = {
                tid: t for tid, t in self._last_diag_log_time.items() if t >= cutoff
            }

        for slot in active_slots:
            s_id = slot.zone_id
            state = self.slot_states[s_id]
            geom = slot_geometries[s_id]
            pts_scaled = geom["pts_scaled"]
            slot_cx = geom["cx"]
            slot_cy = geom["cy"]
            slot_diag = geom["diag"]
            slot_bbox = geom["bbox"]

            slot_target_classes = getattr(slot, "target_classes", None)
            allowed_classes = set(slot_target_classes) if slot_target_classes else self.vehicle_classes

            for vt in vehicle_tracks:
                x1, y1, x2, y2 = vt.bbox
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)

                p_center = (cx, float(y2))
                p_inset = (cx, float(y2 - bh * 0.05))
                p_left = (float(x1 + bw * 0.22), float(y2 - bh * 0.04))
                p_right = (float(x2 - bw * 0.22), float(y2 - bh * 0.04))
                p_axle = (cx, float(y2 - bh * 0.12))

                d_center = cv2.pointPolygonTest(pts_scaled, p_center, True)
                d_inset = cv2.pointPolygonTest(pts_scaled, p_inset, True)
                d_left = cv2.pointPolygonTest(pts_scaled, p_left, True)
                d_right = cv2.pointPolygonTest(pts_scaled, p_right, True)
                d_axle = cv2.pointPolygonTest(pts_scaled, p_axle, True)

                stance_probes = [d_center, d_inset, d_left, d_right, d_axle]
                d_ground = max(stance_probes)
                d_wheel = d_center
                d_lower = d_axle
                d_max = d_ground

                iou_slot = bbox_iou(vt.bbox, slot_bbox)

                iou_anchor = 0.0
                if state.phase == "OCCUPIED" and state.last_bbox is not None:
                    iou_anchor = bbox_iou(vt.bbox, state.last_bbox)
                anchor_threshold = 0.15 if state.latch_occupied else 0.25

                num_in = sum(1 for d in stance_probes if d >= 0.0)

                lower_bbox = (x1, y1 + bh * 0.5, x2, y2)
                lower_overlap = bbox_polygon_overlap_ratio(lower_bbox, pts_scaled)

                is_class_allowed = (vt.class_label in allowed_classes)
                if not is_class_allowed:
                    if s_id != "zone_01" and vt.class_label in ("truck", "bus") and (d_center >= 0.0 or lower_overlap >= 0.35):
                        is_class_allowed = True
                if not is_class_allowed:
                    continue

                eff_margin = max(8.0, self.wheel_contact_margin_px)
                has_stance_contact = (
                    (d_ground >= -eff_margin)
                    or (lower_overlap >= 0.15)
                    or (d_ground >= -12.0 and lower_overlap >= 0.10)
                )

                cx_1080p = (cx / sx) if sx > 0 else cx
                is_outside_s8 = (s_id == "zone_08" and cx_1080p > 1810.0)

                y2_ai = y2 if (sy <= 0.5) else (y2 * (360.0 / 1080.0))
                if y2_ai >= 225.0:
                    continue

                is_vacant_slot = (state.phase == "VACANT")
                if is_vacant_slot:
                    if s_id in ("zone_02", "zone_04") and vt.class_label in ("car", "truck") and (lower_overlap >= 0.15 or d_ground >= -eff_margin):
                        acq_thresh = min(0.20, self.acquisition_conf_thresh)
                    elif vt.class_label in ("car", "truck") and s_id != "zone_01" and lower_overlap >= 0.15:
                        acq_thresh = min(0.20, self.acquisition_conf_thresh)
                    else:
                        acq_thresh = self.acquisition_conf_thresh

                    conf_ok = (vt.confidence >= acq_thresh)
                    passed_gate = conf_ok and has_stance_contact and (d_ground >= -eff_margin)
                else:
                    acq_thresh = self.acquisition_conf_thresh
                    conf_ok = (vt.confidence >= self.retention_conf_thresh)
                    passed_gate = (conf_ok and (has_stance_contact or d_ground >= -12.0)) or (iou_anchor >= anchor_threshold)

                if is_outside_s8:
                    passed_gate = False

                if self.debug_diagnostics and s_id == self.debug_target_zone:
                    try:
                        boxes_overlap = not (
                            vt.bbox[2] < slot_bbox[0]
                            or vt.bbox[0] > slot_bbox[2]
                            or vt.bbox[3] < slot_bbox[1]
                            or vt.bbox[1] > slot_bbox[3]
                        )
                        is_spatial_trigger = (-40.0 <= d_max <= 10.0) or boxes_overlap

                        if is_spatial_trigger:
                            is_stable_occupied_owner = (
                                passed_gate
                                and state.track_id == vt.track_id
                                and state.phase == "OCCUPIED"
                            )
                            throttle_sec = 15.0 if is_stable_occupied_owner else 0.5

                            last_log = self._last_diag_log_time.get(vt.track_id, 0.0)
                            if (current_time - last_log) >= throttle_sec:
                                self._last_diag_log_time[vt.track_id] = current_time
                                iso_now = datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat()
                                cand_score = None
                                if passed_gate:
                                    dist_to_center = math.hypot(cx - slot_cx, cy - slot_cy)
                                    dist_norm = dist_to_center / slot_diag
                                    continuity_bonus = 25.0 if (state.track_id == vt.track_id and state.phase == "OCCUPIED") else 0.0
                                    latch_bonus = 20.0 if state.latch_occupied else 0.0
                                    cand_score = round(
                                        (d_ground * 3.0) - (dist_norm * 30.0) + (iou_slot * 30.0) + (iou_anchor * 30.0) + continuity_bonus + latch_bonus,
                                        2,
                                    )

                                if not passed_gate:
                                    if is_outside_s8:
                                        reason_str = f"outside_s8_boundary (cx_1080p={cx_1080p:.1f} > 1810.0)"
                                    elif is_vacant_slot and not conf_ok:
                                        reason_str = f"low_acquisition_conf ({vt.confidence:.2f} < {acq_thresh:.2f})"
                                    elif not conf_ok:
                                        reason_str = f"low_retention_conf ({vt.confidence:.2f} < {self.retention_conf_thresh:.2f})"
                                    elif not has_stance_contact:
                                        reason_str = f"no_stance_contact (d_ground={d_ground:.2f}, lower_overlap={lower_overlap:.2f})"
                                    else:
                                        reason_str = "anchor_mismatch"
                                else:
                                    reason_str = "PASSED_SPATIAL_GATE"

                                payload = {
                                    "timestamp": round(current_time, 3),
                                    "iso_time": iso_now,
                                    "processed_count": frame_count,
                                    "zone_id": s_id,
                                    "track_id": vt.track_id,
                                    "class_label": vt.class_label,
                                    "confidence": round(vt.confidence, 3),
                                    "bbox": [round(c, 1) for c in vt.bbox],
                                    "wheel_pt": [round(cx, 1), round(y2, 1)],
                                    "lower_pt": [round(cx, 1), round(y1 + bh * 0.75, 1)],
                                    "center_pt": [round(cx, 1), round(cy, 1)],
                                    "d_wheel": round(d_wheel, 2),
                                    "d_lower": round(d_lower, 2),
                                    "d_center": round(d_center, 2),
                                    "d_ground": round(d_ground, 2),
                                    "d_max": round(d_max, 2),
                                    "iou_slot": round(iou_slot, 3),
                                    "iou_anchor": round(iou_anchor, 3),
                                    "score": cand_score,
                                    "candidate_status": "ACCEPTED" if passed_gate else "REJECTED",
                                    "reason": reason_str,
                                    "phase": state.phase,
                                    "dwell": round(state.dwell_duration, 2),
                                    "latch_occupied": state.latch_occupied,
                                    "is_warmup": is_warmup,
                                    "is_confirmed": vt.is_confirmed,
                                    "snapshot": frame_snapshot_status,
                                }
                                self._emit_diagnostic_log(payload)
                    except Exception as diag_err:
                        if not self._has_logged_diag_error:
                            self._has_logged_diag_error = True
                            try:
                                logging.getLogger("engine.parking.car_slot").warning(
                                    f"Diagnostic logging exception swallowed: {diag_err}"
                                )
                            except Exception:
                                pass

                if passed_gate:
                    dist_to_center = math.hypot(cx - slot_cx, cy - slot_cy)
                    dist_norm = dist_to_center / slot_diag
                    ioz_stance = float(num_in) / float(len(stance_probes))

                    continuity_bonus = 25.0 if (state.track_id == vt.track_id and state.phase == "OCCUPIED") else 0.0
                    latch_bonus = 20.0 if state.latch_occupied else 0.0
                    score = (
                        (d_ground * 3.0)
                        - (dist_norm * 30.0)
                        + (iou_slot * 30.0)
                        + (lower_overlap * 20.0)
                        + (ioz_stance * 15.0)
                        + (iou_anchor * 30.0)
                        + continuity_bonus
                        + latch_bonus
                    )
                    candidates.append((score, s_id, vt))

            if self.debug_diagnostics and s_id == self.debug_target_zone:
                try:
                    vehicle_track_ids = {t.track_id for t in vehicle_tracks}
                    for non_vt in tracks:
                        if non_vt.track_id in vehicle_track_ids:
                            continue

                        nx1, ny1, nx2, ny2 = non_vt.bbox
                        ncx = float((nx1 + nx2) / 2.0)
                        ncy = float((ny1 + ny2) / 2.0)
                        n_wheel_pt = (ncx, float(ny2))
                        n_lower_pt = (ncx, float(ny1 + (ny2 - ny1) * 0.75))
                        n_center_pt = (ncx, ncy)

                        n_d_wheel = cv2.pointPolygonTest(pts_scaled, n_wheel_pt, True)
                        n_d_lower = cv2.pointPolygonTest(pts_scaled, n_lower_pt, True)
                        n_d_center = cv2.pointPolygonTest(pts_scaled, n_center_pt, True)
                        n_d_max = max(n_d_wheel, n_d_lower, n_d_center)
                        n_iou_slot = bbox_iou(non_vt.bbox, slot_bbox)

                        boxes_overlap = not (
                            non_vt.bbox[2] < slot_bbox[0]
                            or non_vt.bbox[0] > slot_bbox[2]
                            or non_vt.bbox[3] < slot_bbox[1]
                            or non_vt.bbox[1] > slot_bbox[3]
                        )
                        is_spatial_trigger = (-40.0 <= n_d_max <= 10.0) or boxes_overlap

                        if is_spatial_trigger:
                            last_log = self._last_diag_log_time.get(non_vt.track_id, 0.0)
                            if (current_time - last_log) >= 0.5:
                                self._last_diag_log_time[non_vt.track_id] = current_time
                                iso_now = datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat()

                                if non_vt.class_label not in self.vehicle_classes:
                                    reason_str = f"unsupported_class ({non_vt.class_label})"
                                elif not (non_vt.is_confirmed or is_warmup):
                                    hits_val = getattr(non_vt, "hits", None)
                                    if hits_val is not None:
                                        reason_str = f"unconfirmed_track (hits={hits_val}, is_confirmed={non_vt.is_confirmed})"
                                    else:
                                        reason_str = f"unconfirmed_track (is_confirmed={non_vt.is_confirmed})"
                                else:
                                    reason_str = f"d_max_below_margin ({n_d_max:.2f} < -3.0)"

                                payload = {
                                    "timestamp": round(current_time, 3),
                                    "iso_time": iso_now,
                                    "processed_count": frame_count,
                                    "zone_id": s_id,
                                    "track_id": non_vt.track_id,
                                    "class_label": non_vt.class_label,
                                    "confidence": round(non_vt.confidence, 3),
                                    "bbox": [round(c, 1) for c in non_vt.bbox],
                                    "wheel_pt": [round(ncx, 1), round(ny2, 1)],
                                    "lower_pt": [round(ncx, 1), round(ny1 + (ny2 - ny1) * 0.75, 1)],
                                    "center_pt": [round(ncx, 1), round(ncy, 1)],
                                    "d_wheel": round(n_d_wheel, 2),
                                    "d_lower": round(n_d_lower, 2),
                                    "d_center": round(n_d_center, 2),
                                    "d_max": round(n_d_max, 2),
                                    "iou_slot": round(n_iou_slot, 3),
                                    "iou_anchor": 0.0,
                                    "score": None,
                                    "candidate_status": "REJECTED",
                                    "reason": reason_str,
                                    "phase": state.phase,
                                    "dwell": round(state.dwell_duration, 2),
                                    "latch_occupied": state.latch_occupied,
                                    "is_warmup": is_warmup,
                                    "is_confirmed": non_vt.is_confirmed,
                                    "snapshot": frame_snapshot_status,
                                }
                                self._emit_diagnostic_log(payload)
                except Exception as diag_err:
                    if not self._has_logged_diag_error:
                        self._has_logged_diag_error = True
                        try:
                            logging.getLogger("engine.parking.car_slot").warning(
                                f"Diagnostic non-vehicle tracking exception swallowed: {diag_err}"
                            )
                        except Exception:
                            pass

        candidates.sort(key=lambda c: c[0], reverse=True)
        if self.debug_diagnostics:
            self._last_candidates = [(round(c[0], 2), c[1], c[2].track_id) for c in candidates]

        assigned_slot_to_track: Dict[str, TrackResult] = {}
        assigned_track_ids: Set[int] = set()

        for score, s_id, vt in candidates:
            if s_id not in assigned_slot_to_track and vt.track_id not in assigned_track_ids:
                assigned_slot_to_track[s_id] = vt
                assigned_track_ids.add(vt.track_id)

        if self.debug_diagnostics:
            self._last_assigned_slots = {k: v.track_id for k, v in assigned_slot_to_track.items()}

        corridor_vehicles_in_frame: List[TrackResult] = []
        for cv_t in tracks:
            if getattr(cv_t, "class_label", None) in ("car", "truck", "bus"):
                cv_x1, cv_y1, cv_x2, cv_y2 = cv_t.bbox
                cv_y2_ai = cv_y2 if (sy <= 0.5) else (cv_y2 * (360.0 / 1080.0))
                if cv_y2_ai >= 225.0:
                    corridor_vehicles_in_frame.append(cv_t)

        def is_slot_occluded_by_corridor(slot_geom_item: Dict[str, Any]) -> bool:
            s_bbox = slot_geom_item["bbox"]
            s_xmin, s_ymin, s_xmax, s_ymax = s_bbox
            slot_w = max(1.0, s_xmax - s_xmin)
            for cv in corridor_vehicles_in_frame:
                cx1, cy1, cx2, cy2 = cv.bbox
                c_y1_ai = cy1 if (sy <= 0.5) else (cy1 * (360.0 / 1080.0))
                if c_y1_ai <= (s_ymax + 15.0):
                    ix1 = max(s_xmin, cx1)
                    ix2 = min(s_xmax, cx2)
                    inter_w = max(0.0, ix2 - ix1)
                    if (inter_w / slot_w) >= 0.35:
                        return True
            return False

        for slot in active_slots:
            slot_id = slot.zone_id
            state = self.slot_states[slot_id]
            geom = slot_geometries[slot_id]
            pts_scaled = geom["pts_scaled"]

            s1_has_contact = False
            if slot_id == "zone_01":
                slot_eff_margin = max(8.0, self.wheel_contact_margin_px)
                for cand_t in tracks:
                    cx1, cy1, cx2, cy2 = cand_t.bbox
                    c_y2_ai = cy2 if (sy <= 0.5) else (cy2 * (360.0 / 1080.0))
                    if c_y2_ai >= 225.0:
                        continue
                    ccx = float((cx1 + cx2) / 2.0)
                    cbw = max(1.0, cx2 - cx1)
                    cbh = max(1.0, cy2 - cy1)
                    c_probes = [
                        (ccx, float(cy2)),
                        (ccx, float(cy2 - cbh * 0.05)),
                        (float(cx1 + cbw * 0.22), float(cy2 - cbh * 0.04)),
                        (float(cx2 - cbw * 0.22), float(cy2 - cbh * 0.04)),
                        (ccx, float(cy2 - cbh * 0.12)),
                    ]
                    c_d_probes = [cv2.pointPolygonTest(pts_scaled, pt, True) for pt in c_probes]
                    c_d_ground = max(c_d_probes)
                    c_lower_bbox = (cx1, cy1 + cbh * 0.5, cx2, cy2)
                    c_lower_overlap = bbox_polygon_overlap_ratio(c_lower_bbox, pts_scaled)
                    c_iou_anc = bbox_iou(cand_t.bbox, state.last_bbox) if state.last_bbox is not None else 0.0
                    if (
                        c_d_ground >= -slot_eff_margin
                        or c_lower_overlap >= 0.15
                        or (c_d_ground >= -12.0 and c_lower_overlap >= 0.10)
                        or c_iou_anc >= 0.15
                    ):
                        s1_has_contact = True
                        break

            matched_track = assigned_slot_to_track.get(slot_id)

            if matched_track is not None:
                if state.track_id == matched_track.track_id:
                    if state.first_seen_time is not None:
                        state.dwell_duration = current_time - state.first_seen_time
                    else:
                        state.first_seen_time = current_time
                        state.dwell_duration = 0.0
                    state.vehicle_class = matched_track.class_label
                    state.last_seen_time = current_time
                    state.last_bbox = matched_track.bbox
                    state.polygon_clear_since = None

                elif state.phase in ("OCCUPIED", "ENTERING", "LEAVING") and self._is_same_vehicle(
                    state, matched_track, pts_scaled, current_time
                ):
                    state.track_id = matched_track.track_id
                    state.vehicle_class = matched_track.class_label
                    state.last_seen_time = current_time
                    state.last_bbox = matched_track.bbox
                    state.polygon_clear_since = None
                    if state.first_seen_time is not None:
                        state.dwell_duration = current_time - state.first_seen_time

                else:
                    if slot_id == "zone_01" and state.phase == "OCCUPIED" and (state.dwell_duration >= 10.0 or state.latch_occupied):
                        state.track_id = matched_track.track_id
                        state.vehicle_class = matched_track.class_label
                        state.dwell_duration = max(state.dwell_duration, 10.0)
                        state.last_seen_time = current_time
                        state.last_bbox = matched_track.bbox
                        state.polygon_clear_since = None
                        state.latch_occupied = True
                    else:
                        state.track_id = matched_track.track_id
                        state.vehicle_class = matched_track.class_label
                        state.first_seen_time = current_time
                        state.dwell_duration = 0.0
                        state.last_seen_time = current_time
                        state.last_bbox = matched_track.bbox
                        state.polygon_clear_since = None

                if is_warmup:
                    if matched_track.is_confirmed:
                        state.warmup_hits += 1
                    state.phase = "OCCUPIED"
                    state.dwell_duration = max(state.dwell_duration, self.dwell_threshold_sec)
                    state.leaving_since = None
                    if state.warmup_hits >= 10 and matched_track.is_confirmed:
                        state.latch_occupied = True
                        state.is_warmup_latch = True
                else:
                    if matched_track.is_confirmed:
                        state.is_warmup_latch = False

                    if state.phase == "VACANT":
                        if state.dwell_duration >= self.dwell_threshold_sec:
                            state.phase = "OCCUPIED"
                    elif state.phase == "ENTERING":
                        if state.dwell_duration >= self.dwell_threshold_sec:
                            state.phase = "OCCUPIED"
                    elif state.phase == "LEAVING":
                        state.phase = "OCCUPIED"
                        state.leaving_since = None
                        state.polygon_clear_since = None
                    elif state.phase == "OCCUPIED":
                        state.leaving_since = None

                    if state.phase == "OCCUPIED" and state.dwell_duration >= state.latch_dwell_threshold_sec:
                        state.latch_occupied = True
                        state.is_warmup_latch = False

            else:
                if is_warmup and not state.latch_occupied:
                    state.phase = "VACANT"
                    state.track_id = None
                    state.vehicle_class = None
                    state.first_seen_time = None
                    state.dwell_duration = 0.0
                    state.warmup_hits = 0
                elif state.track_id is not None and state.track_id in assigned_track_ids and not state.latch_occupied:
                    if slot_id == "zone_01" and s1_has_contact:
                        pass
                    else:
                        state.phase = "VACANT"
                        state.latch_occupied = False
                        state.is_warmup_latch = False
                        state.track_id = None
                        state.vehicle_class = None
                        state.first_seen_time = None
                        state.dwell_duration = 0.0
                        state.leaving_since = None
                        state.polygon_clear_since = None
                        state.tripwire_crossed_in_time = None
                        state.warmup_hits = 0
                elif state.phase == "ENTERING":
                    if (
                        state.tripwire_crossed_in_time is not None
                        and (current_time - state.tripwire_crossed_in_time) > state.entering_timeout_sec
                    ):
                        state.phase = "VACANT"
                        state.track_id = None
                        state.first_seen_time = None
                        state.dwell_duration = 0.0
                        state.tripwire_crossed_in_time = None
                elif state.phase == "OCCUPIED":
                    confirm_threshold = 10.0 if state.latch_occupied else max(5.0, state.vacant_confirm_sec)

                    is_occluded = is_slot_occluded_by_corridor(geom)
                    if state.dwell_duration >= 10.0 or state.latch_occupied:
                        if (slot_id == "zone_01" and s1_has_contact) or is_occluded:
                            state.last_seen_time = current_time
                            state.polygon_clear_since = None
                            state.latch_occupied = True

                    if state.last_seen_time > 0 and (current_time - state.last_seen_time) > 0.5:
                        if state.polygon_clear_since is None:
                            state.polygon_clear_since = current_time
                        elif not is_warmup and state.is_warmup_latch and (current_time - state.polygon_clear_since) >= 4.0:
                            state.latch_occupied = False
                            state.is_warmup_latch = False
                            state.phase = "VACANT"
                            state.track_id = None
                            state.vehicle_class = None
                            state.first_seen_time = None
                            state.dwell_duration = 0.0
                            state.leaving_since = None
                            state.polygon_clear_since = None
                            state.tripwire_crossed_in_time = None
                            state.warmup_hits = 0
                        elif (current_time - state.polygon_clear_since) >= confirm_threshold:
                            state.phase = "LEAVING"
                            state.leaving_since = current_time
                            state.polygon_clear_since = None
                    else:
                        state.polygon_clear_since = None
                elif state.phase == "LEAVING":
                    leave_time = state.leaving_since if state.leaving_since is not None else state.last_seen_time
                    if leave_time > 0 and (current_time - leave_time) >= state.exit_grace_sec:
                        state.phase = "VACANT"
                        state.latch_occupied = False
                        state.is_warmup_latch = False
                        state.track_id = None
                        state.vehicle_class = None
                        state.first_seen_time = None
                        state.dwell_duration = 0.0
                        state.leaving_since = None
                        state.tripwire_crossed_in_time = None
                        state.warmup_hits = 0
                elif state.phase == "VACANT":
                    if state.last_seen_time > 0 and (current_time - state.last_seen_time) > 2.0:
                        state.track_id = None
                        state.vehicle_class = None
                        state.first_seen_time = None
                        state.dwell_duration = 0.0

            if self.debug_diagnostics and slot_id == self.debug_target_zone:
                try:
                    if slot_id not in self._prev_slot_phases:
                        self._prev_slot_phases[slot_id] = state.phase
                    else:
                        prev_p = self._prev_slot_phases[slot_id]
                        if state.phase != prev_p:
                            self._prev_slot_phases[slot_id] = state.phase
                            if self.debug_snapshots:
                                if not frame_snapshot_triggered:
                                    frame_snapshot_status = self._trigger_snapshot(raw_ai_frame, current_time, "phase")
                                    frame_snapshot_triggered = True
                            iso_now = datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat()
                            payload = {
                                "timestamp": round(current_time, 3),
                                "iso_time": iso_now,
                                "processed_count": frame_count,
                                "zone_id": slot_id,
                                "event": "PHASE_TRANSITION",
                                "previous_phase": prev_p,
                                "phase": state.phase,
                                "track_id": state.track_id,
                                "vehicle_class": state.vehicle_class,
                                "dwell": round(state.dwell_duration, 2),
                                "latch_occupied": state.latch_occupied,
                                "is_warmup": is_warmup,
                                "snapshot": frame_snapshot_status,
                            }
                            self._emit_diagnostic_log(payload)
                except Exception as diag_err:
                    if not self._has_logged_diag_error:
                        self._has_logged_diag_error = True
                        try:
                            logging.getLogger("engine.parking.car_slot").warning(
                                f"Diagnostic phase transition logging exception swallowed: {diag_err}"
                            )
                        except Exception:
                            pass

        # Maneuvering Corridor Obstruction Engine
        active_tids_in_frame = {vt.track_id for vt in vehicle_tracks}
        for vt in vehicle_tracks:
            if vt.track_id in assigned_track_ids:
                if vt.track_id in self._corridor_vehicles:
                    del self._corridor_vehicles[vt.track_id]
                continue

            x1, y1, x2, y2 = vt.bbox
            vcx = float((x1 + x2) / 2.0)
            vcy = float((y1 + y2) / 2.0)
            v_stance = (vcx, float(y2))

            in_any_slot = False
            for geom in slot_geometries.values():
                if cv2.pointPolygonTest(geom["pts_scaled"], v_stance, False) >= 0:
                    in_any_slot = True
                    break
                if cv2.pointPolygonTest(geom["pts_scaled"], (vcx, vcy), False) >= 0:
                    in_any_slot = True
                    break

            if in_any_slot:
                if vt.track_id in self._corridor_vehicles:
                    del self._corridor_vehicles[vt.track_id]
                continue

            curr_pos = (vcx, vcy)
            if vt.track_id not in self._corridor_vehicles:
                self._corridor_vehicles[vt.track_id] = {
                    "first_seen_time": current_time,
                    "last_seen_time": current_time,
                    "anchor_centroid": curr_pos,
                    "last_centroid": curr_pos,
                    "max_shift": 0.0,
                    "bbox": vt.bbox,
                    "class_label": vt.class_label,
                    "dwell_duration": 0.0,
                    "is_obstruction": False,
                }
            else:
                entry = self._corridor_vehicles[vt.track_id]
                entry["last_seen_time"] = current_time
                entry["bbox"] = vt.bbox
                entry["last_centroid"] = curr_pos
                shift = math.hypot(curr_pos[0] - entry["anchor_centroid"][0], curr_pos[1] - entry["anchor_centroid"][1])
                entry["max_shift"] = max(entry["max_shift"], shift)

                if shift > self.corridor_shift_threshold_px:
                    entry["anchor_centroid"] = curr_pos
                    entry["first_seen_time"] = current_time
                    entry["dwell_duration"] = 0.0
                    entry["is_obstruction"] = False
                else:
                    entry["dwell_duration"] = current_time - entry["first_seen_time"]
                    if entry["dwell_duration"] >= self.corridor_obstruction_dwell_sec:
                        entry["is_obstruction"] = True

        stale_corridor_tids = [
            tid for tid, entry in self._corridor_vehicles.items()
            if (current_time - entry["last_seen_time"]) > 3.0 or tid not in active_tids_in_frame
        ]
        for tid in stale_corridor_tids:
            del self._corridor_vehicles[tid]

        occupied_count = sum(1 for s in self.slot_states.values() if s.phase == "OCCUPIED")
        available_count = max(0, total_capacity - occupied_count)
        return {
            "total_slots": total_capacity,
            "occupied_slots": occupied_count,
            "available_slots": available_count,
            "gate_in": self.gate_in_count,
            "gate_out": self.gate_out_count,
            "slot_states": self.slot_states,
            "obstruction_alert": self.has_obstruction,
            "obstructions": self.active_obstructions,
        }
