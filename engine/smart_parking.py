from __future__ import annotations

import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import cv2
import numpy as np

from engine.config_loader import ROIZone, TripwireRule
from engine.geometry import bbox_bottom_center, bbox_iou, point_in_polygon, scale_points
from engine.tracker_interface import TrackResult


@dataclass
class SlotState:

    slot_id: str
    slot_num: int
    slot_idx_str: str
    label: str
    phase: Literal["VACANT", "ENTERING", "OCCUPIED", "LEAVING"] = "VACANT"
    track_id: Optional[int] = None
    vehicle_class: Optional[str] = None
    first_seen_time: Optional[float] = None
    dwell_duration: float = 0.0
    last_seen_time: float = 0.0
    tripwire_crossed_in_time: Optional[float] = None
    entering_timeout_sec: float = 30.0
    leaving_since: Optional[float] = None
    repark_window_sec: float = 5.0
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    polygon_clear_since: Optional[float] = None
    vacant_confirm_sec: float = 5.0
    exit_grace_sec: float = 2.0
    latch_occupied: bool = False
    latch_dwell_threshold_sec: float = 5.0
    warmup_hits: int = 0
    is_warmup_latch: bool = False

    @property
    def occupied(self) -> bool:
        return self.phase == "OCCUPIED"


def bbox_polygon_overlap_ratio(
    bbox: Tuple[float, float, float, float],
    pts_scaled: np.ndarray,
) -> float:
    bx1, by1, bx2, by2 = bbox
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    bbox_area = bw * bh

    px, py, pw, ph = cv2.boundingRect(pts_scaled)
    ix1 = max(bx1, float(px))
    iy1 = max(by1, float(py))
    ix2 = min(bx2, float(px + pw))
    iy2 = min(by2, float(py + ph))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    rx1, ry1 = int(math.floor(ix1)), int(math.floor(iy1))
    rx2, ry2 = int(math.ceil(ix2)), int(math.ceil(iy2))
    mw = rx2 - rx1
    mh = ry2 - ry1
    if mw <= 0 or mh <= 0:
        return 0.0

    mask = np.zeros((mh, mw), dtype=np.uint8)
    poly_shifted = pts_scaled - np.array([rx1, ry1], dtype=np.int32)
    cv2.fillPoly(mask, [poly_shifted], 255)

    bx_s1 = max(0, int(round(bx1 - rx1)))
    by_s1 = max(0, int(round(by1 - ry1)))
    bx_s2 = min(mw, int(round(bx2 - rx1)))
    by_s2 = min(mh, int(round(by2 - ry1)))

    if bx_s2 <= bx_s1 or by_s2 <= by_s1:
        return 0.0

    box_submask = mask[by_s1:by_s2, bx_s1:bx_s2]
    intersection_area = float(cv2.countNonZero(box_submask))
    return intersection_area / bbox_area


def bbox_ios(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(intersection / min(area_a, area_b))


def deduplicate_motorcycle_tracks(
    tracks: List[TrackResult],
    iou_thresh: float = 0.30,
    ios_thresh: float = 0.50,
    max_centroid_dist_px: float = 28.0,
) -> List[TrackResult]:

    sorted_tracks = sorted(tracks, key=lambda t: t.confidence, reverse=True)
    deduped: List[TrackResult] = []
    for cand in sorted_tracks:
        c_cx = float((cand.bbox[0] + cand.bbox[2]) / 2.0)
        c_cy = float((cand.bbox[1] + cand.bbox[3]) / 2.0)
        is_dup = False
        for acc in deduped:
            iou = bbox_iou(cand.bbox, acc.bbox)
            ios = bbox_ios(cand.bbox, acc.bbox)
            a_cx = float((acc.bbox[0] + acc.bbox[2]) / 2.0)
            a_cy = float((acc.bbox[1] + acc.bbox[3]) / 2.0)
            dist_px = math.hypot(c_cx - a_cx, c_cy - a_cy)
            if iou >= iou_thresh or ios >= ios_thresh or dist_px <= max_centroid_dist_px:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)
    return deduped


def is_valid_motorcycle_anatomy(
    bbox: Tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    max_w_1080p: float = 260.0,
    max_h_1080p: float = 360.0,
    min_w_1080p: float = 20.0,
    min_h_1080p: float = 25.0,
    max_area_1080p: float = 65000.0,
    min_aspect_ratio: float = 0.20,
    max_aspect_ratio: float = 2.50,
) -> bool:
    sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
    sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

    x1, y1, x2, y2 = bbox
    bw_curr = max(1.0, float(x2 - x1))
    bh_curr = max(1.0, float(y2 - y1))

    bw_1080p = bw_curr / sx
    bh_1080p = bh_curr / sy
    area_1080p = bw_1080p * bh_1080p
    aspect_ratio = bw_1080p / bh_1080p

    if (
        bw_1080p > max_w_1080p
        or bh_1080p > max_h_1080p
        or bw_1080p < min_w_1080p
        or bh_1080p < min_h_1080p
        or area_1080p > max_area_1080p
        or aspect_ratio < min_aspect_ratio
        or aspect_ratio > max_aspect_ratio
    ):
        return False
    return True


@dataclass
class StationaryMotorUnit:
    unit_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    centroid: Tuple[float, float]
    first_seen_time: float
    last_seen_time: float
    consecutive_hits: int = 1
    missed_frames: int = 0
    is_latched: bool = False


class SmartParkingTracker:
    DEFAULT_VEHICLE_CLASSES = {"car", "truck", "bus"}

    def __init__(
        self,
        total_slots: Optional[int] = None,
        dwell_threshold_sec: float = 10.0,
        vehicle_classes: Optional[Set[str]] = None,
        parking_mode: str = "slot",
        block_capacity: int = 30,
        stationary_dwell_sec: float = 0.0,
        block_exclusion_x_1080p: Optional[int] = None,
        debug_diagnostics: bool = False,
        debug_target_zone: Optional[str] = None,
        debug_log_path: str = "logs/s4_diagnostics.log",
        debug_snapshots: bool = False,
        debug_snapshot_max_files: int = 300,
        debug_snapshot_min_interval_s: float = 10.0,
        debug_snapshot_dir: str = "logs/snapshots",
        acquisition_conf_thresh: float = 0.32,
        retention_conf_thresh: float = 0.20,
        wheel_contact_margin_px: float = 0.0,
        corridor_obstruction_dwell_sec: float = 60.0,
        corridor_shift_threshold_px: float = 15.0,
    ) -> None:
        self._total_slots_override = total_slots
        self.dwell_threshold_sec = dwell_threshold_sec
        self.vehicle_classes = vehicle_classes or self.DEFAULT_VEHICLE_CLASSES
        self.parking_mode = parking_mode
        self.block_capacity = block_capacity
        self.stationary_dwell_sec = stationary_dwell_sec
        self._block_exclusion_x_1080p: Optional[int] = block_exclusion_x_1080p
        self.acquisition_conf_thresh = float(acquisition_conf_thresh)
        self.retention_conf_thresh = float(retention_conf_thresh)
        self.wheel_contact_margin_px = float(wheel_contact_margin_px)
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
        self._last_gate_in: int = 0
        self._last_gate_out: int = 0
        self._zone_dwell_map: Dict[int, float] = {}
        self._stationary_motor_units: Dict[int, StationaryMotorUnit] = {}
        self._next_motor_unit_id: int = 1
        self._density_history: deque = deque(maxlen=25)
        self._last_density_time: Optional[float] = None
        self._last_stable_occupied: Optional[int] = None
        self._candidate_occupied: Optional[int] = None
        self._candidate_occupied_since: float = 0.0
        self.slot_states: Dict[str, SlotState] = {}
        self._last_candidates: List[Tuple[float, str, int]] = []
        self._last_assigned_slots: Dict[str, int] = {}
        self.gate_in_count: int = 0
        self.gate_out_count: int = 0
        self.tripwire_flash: Dict[str, float] = {}
        self.corridor_obstruction_dwell_sec: float = float(corridor_obstruction_dwell_sec)
        self.corridor_shift_threshold_px: float = float(corridor_shift_threshold_px)
        self._corridor_vehicles: Dict[int, Dict[str, Any]] = {}

    def _get_debug_logger(self) -> Optional[Any]:
        if not self.debug_diagnostics or not self.debug_target_zone:
            return None
        if self._debug_logger is not None:
            return self._debug_logger
        if self._debug_logger_failed:
            return None
        try:
            import logging
            from logging.handlers import RotatingFileHandler
            from pathlib import Path

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
                    import logging
                    logging.getLogger("engine.smart_parking").warning(
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
            import json
            diag_logger.info(json.dumps(payload, separators=(",", ":")))
        except Exception as exc:
            if not self._has_logged_diag_error:
                self._has_logged_diag_error = True
                try:
                    import logging
                    logging.getLogger("engine.smart_parking").warning(
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
                                except ValueError:
                                    pass
                existing.sort(key=lambda x: x[0])
                self._existing_snapshot_files = existing

                self._snapshot_stop_event.clear()
                self._snapshot_worker = threading.Thread(
                    target=self._snapshot_worker_loop,
                    name=f"snap_worker_{self.debug_target_zone}",
                    daemon=True,
                )
                self._snapshot_worker.start()
                return True
            except Exception as exc:
                if not self._has_logged_diag_error:
                    self._has_logged_diag_error = True
                    try:
                        import logging
                        logging.getLogger("engine.smart_parking").warning(
                            f"Snapshot worker initialization failed: {exc}"
                        )
                    except Exception:
                        pass
                return False

    def _snapshot_worker_loop(self) -> None:
        while not self._snapshot_stop_event.is_set():
            try:
                item = self._snapshot_queue.get(timeout=0.5)
            except Exception:
                continue

            if item is None:
                break

            filename, frame_bgr, epoch_ms, event_type = item
            try:
                snap_path = os.path.join(self.debug_snapshot_dir, filename)
                cv2.imwrite(snap_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                with self._snapshot_lock:
                    self._existing_snapshot_files.append((epoch_ms, filename, event_type))
                    self._existing_snapshot_files.sort(key=lambda x: x[0])

                    max_files = max(1, int(self.debug_snapshot_max_files))
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
                        import logging
                        logging.getLogger("engine.smart_parking").warning(
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

    @property
    def total_slots(self) -> int:
        if self.parking_mode == "motorcycle_block":
            return self.block_capacity
        if self._total_slots_override is not None:
            return self._total_slots_override
        return len(self.slot_states)

    @property
    def occupied_slots(self) -> int:
        if self.parking_mode == "motorcycle_block":
            return self._last_stable_occupied if self._last_stable_occupied is not None else 0
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

    def apply_tripwire_signal(
        self,
        slot_id: str,
        direction: str,
        track_id: int,
        current_time: float,
    ) -> None:
        if self.parking_mode == "motorcycle_block":
            return

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
        if state.last_seen_time > 0:
            elapsed = current_time - state.last_seen_time
            if elapsed > grace_period_sec:
                return False

        x1, y1, x2, y2 = new_track.bbox
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

    def _block_stats(self, occupied: int) -> Dict[str, Any]:
        self._last_stable_occupied = occupied
        available = max(0, self.block_capacity - occupied)
        return {
            "total_slots": self.block_capacity,
            "occupied_slots": occupied,
            "available_slots": available,
            "gate_in": self.gate_in_count,
            "gate_out": self.gate_out_count,
            "slot_states": {},
            "parking_mode": "motorcycle_block",
        }

    def _update_block_mode(
        self,
        tracks: List[TrackResult],
        polygons: List[ROIZone],
        scale_x: float,
        scale_y: float,
        current_time: float,
        is_warmup: bool = False,
    ) -> Dict[str, Any]:
        active_zone = next((p for p in polygons if p.active), None)
        if active_zone is None:
            return self._block_stats(0)

        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        pts_scaled = np.array(
            [[int(pt.x * sx), int(pt.y * sy)] for pt in active_zone.points],
            dtype=np.int32,
        )

        vehicle_tracks = [
            t
            for t in tracks
            if (t.is_confirmed or is_warmup or getattr(t, "age", 0) >= 1) and (t.class_label in self.vehicle_classes)
        ]

        drum_excl_x_scaled: Optional[int] = None
        if self._block_exclusion_x_1080p is not None:
            drum_excl_x_scaled = int(self._block_exclusion_x_1080p * sx)

        candidate_tracks: List[TrackResult] = []
        for track in vehicle_tracks:
            rx1, ry1, rx2, ry2 = track.bbox
            cx = float((rx1 + rx2) / 2.0)
            wheel_y = float(ry2)
            wheel_pt = (int(cx), int(wheel_y))

            if drum_excl_x_scaled is not None and int(cx) < drum_excl_x_scaled:
                continue

            if not is_valid_motorcycle_anatomy(track.bbox, sx, sy):
                continue

            wheel_in = cv2.pointPolygonTest(pts_scaled, wheel_pt, False) >= 0
            if not wheel_in:
                continue

            overlap_ratio = bbox_polygon_overlap_ratio(track.bbox, pts_scaled)
            if overlap_ratio < 0.20:
                continue

            candidate_tracks.append(track)

        valid_tracks = deduplicate_motorcycle_tracks(
            candidate_tracks,
            iou_thresh=0.30,
            ios_thresh=0.50,
            max_centroid_dist_px=28.0,
        )

        UNIT_LATCH_SEC = 2.0   
        UNIT_TTL_SEC = 10.0    

        matched_unit_ids: Set[int] = set()

        for cand in sorted(valid_tracks, key=lambda t: t.confidence, reverse=True):
            cand_cx = float((cand.bbox[0] + cand.bbox[2]) / 2.0)
            cand_cy = float((cand.bbox[1] + cand.bbox[3]) / 2.0)

            best_unit_id = None
            best_dist = 30.0  

            for uid, unit in self._stationary_motor_units.items():
                if uid in matched_unit_ids:
                    continue
                dist = math.hypot(cand_cx - unit.centroid[0], cand_cy - unit.centroid[1])
                if dist < best_dist:
                    best_dist = dist
                    best_unit_id = uid

            if best_unit_id is not None:
                unit = self._stationary_motor_units[best_unit_id]
                unit.bbox = cand.bbox
                unit.confidence = cand.confidence
                unit.centroid = (cand_cx, cand_cy)
                unit.last_seen_time = current_time
                unit.consecutive_hits += 1
                unit.missed_frames = 0
                if not unit.is_latched:
                    elapsed_visible = current_time - unit.first_seen_time
                    if elapsed_visible >= UNIT_LATCH_SEC or is_warmup:
                        unit.is_latched = True
                matched_unit_ids.add(best_unit_id)
            else:
                uid = self._next_motor_unit_id
                self._next_motor_unit_id += 1
                is_latched_init = is_warmup
                hits_init = 1
                self._stationary_motor_units[uid] = StationaryMotorUnit(
                    unit_id=uid,
                    bbox=cand.bbox,
                    confidence=cand.confidence,
                    centroid=(cand_cx, cand_cy),
                    first_seen_time=current_time,
                    last_seen_time=current_time,
                    consecutive_hits=hits_init,
                    missed_frames=0,
                    is_latched=is_latched_init,
                )
                matched_unit_ids.add(uid)
        unmatched_unit_ids = set(self._stationary_motor_units.keys()) - matched_unit_ids
        to_delete = []
        for uid in unmatched_unit_ids:
            unit = self._stationary_motor_units[uid]
            unit.missed_frames += 1
            unit.consecutive_hits = 0

            time_since_last_seen = current_time - unit.last_seen_time

            if unit.is_latched:
                if time_since_last_seen > UNIT_TTL_SEC:
                    to_delete.append(uid)
            else:
                if time_since_last_seen > 1.0:
                    to_delete.append(uid)

        for uid in to_delete:
            del self._stationary_motor_units[uid]
        if self.stationary_dwell_sec > 0.0:
            density_count = sum(
                1 for u in self._stationary_motor_units.values()
                if (u.is_latched or (current_time - u.first_seen_time) >= self.stationary_dwell_sec)
                and is_valid_motorcycle_anatomy(u.bbox, sx, sy)
                and (drum_excl_x_scaled is None or u.centroid[0] >= drum_excl_x_scaled)
            )
        else:
            density_count = sum(
                1 for u in self._stationary_motor_units.values()
                if is_valid_motorcycle_anatomy(u.bbox, sx, sy)
                and (drum_excl_x_scaled is None or u.centroid[0] >= drum_excl_x_scaled)
            )

        self._last_density_time = current_time
        self._density_history.append(density_count)
        # Moving Median Window 5 detik (25 frame @ ~5fps) — halus dan adaptif, anti-starvation
        consolidated_density = int(round(float(np.median(self._density_history))))

        tw_count = max(0, self.gate_in_count - self.gate_out_count)
        occupied = int(min(self.block_capacity, max(tw_count, consolidated_density)))
        self._last_stable_occupied = occupied

        return self._block_stats(occupied)

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
        if self.parking_mode == "motorcycle_block":
            return self._update_block_mode(
                tracks=tracks,
                polygons=polygons,
                scale_x=scale_x,
                scale_y=scale_y,
                current_time=current_time,
                is_warmup=is_warmup,
            )

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

        # Pemicu 2: deteksi mentah kelas truck/bus yang beririsan dengan debug_target_zone
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

        # Pembersihan berkala cache throttle diagnostik (>60s)
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

                # 5-Point Weighted Stance Probe Architecture:
                # 1. p_center: bottom-center (bumper / axle center line)
                # 2. p_inset: inset slight vertical margin (front/rear overhang clearance)
                # 3. p_left: left tire stance contact
                # 4. p_right: right tire stance contact
                # 5. p_axle: deep axle center contact (chassis center)
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

                # Evaluasi kontak tapak bodi bawah terhadap poligon slot
                lower_bbox = (x1, y1 + bh * 0.5, x2, y2)
                lower_overlap = bbox_polygon_overlap_ratio(lower_bbox, pts_scaled)

                # Evaluasi kecocokan kelas kendaraan (Defense-in-depth):
                # Slot S1 (zone_01) dan S2 (zone_02) wajib strict hanya 'car' (menolak bayangan kanopi / false positive).
                # Khusus slot S4 (zone_04) di bawah atap gelap menoleransi misklasifikasi 'truck'
                # jika memiliki kontak fisik kuat di dalam slot (lower_overlap >= 0.20 atau d_ground >= 0.0 px).
                is_class_allowed = (vt.class_label in allowed_classes)
                if not is_class_allowed:
                    if s_id == "zone_04" and vt.class_label in ("truck", "bus") and (lower_overlap >= 0.20 or d_ground >= 0.0):
                        is_class_allowed = True
                if not is_class_allowed:
                    continue

                # Toleransi signed distance ground contact hingga -8.0 px untuk mobil di bawah bayangan kanopi gelap
                eff_margin = max(8.0, self.wheel_contact_margin_px)
                has_stance_contact = (
                    (d_ground >= -eff_margin)
                    or (lower_overlap >= 0.15)
                    or (d_ground >= -12.0 and lower_overlap >= 0.10)
                )

                # Strict Bounding Centroid X-Filter untuk S8 (zone_08):
                # Tapak roda tengah mobil abu-abu di luar area parkir berada di koordinat x > 1810 px (pada base frame 1080p).
                # Jika track kendaraan memiliki cx > 1810 px (atau cx_infer > 603.3 pada 640p), reject kendaraan dari kandidat S8.
                cx_1080p = (cx / sx) if sx > 0 else cx
                is_outside_s8 = (s_id == "zone_08" and cx_1080p > 1810.0)

                is_vacant_slot = (state.phase == "VACANT")
                if is_vacant_slot:
                    # Mobil hitam di Slot S4 (zone_04) berada di bawah bayangan atap seng gelap (conf ~0.20 - 0.28).
                    # Ambang akuisisi dilonggarkan ke 0.20 khusus mobil di S4 atau jika lower_overlap >= 0.15,
                    # sementara slot S1 tetap menggunakan acquisition_conf_thresh default (0.32) untuk menolak bayangan kanopi.
                    if s_id == "zone_04" and vt.class_label in ("car", "truck") and (lower_overlap >= 0.15 or d_ground >= -eff_margin):
                        acq_thresh = min(0.20, self.acquisition_conf_thresh)
                    elif vt.class_label in ("car", "truck") and s_id != "zone_01" and lower_overlap >= 0.15:
                        acq_thresh = min(0.20, self.acquisition_conf_thresh)
                    else:
                        acq_thresh = self.acquisition_conf_thresh

                    conf_ok = (vt.confidence >= acq_thresh)
                    passed_gate = conf_ok and has_stance_contact
                else:
                    acq_thresh = self.acquisition_conf_thresh
                    conf_ok = (vt.confidence >= self.retention_conf_thresh)
                    passed_gate = (conf_ok and (has_stance_contact or d_ground >= -12.0)) or (iou_anchor >= anchor_threshold)

                # Isolasi mutlak mobil abu-abu di sebelah kanan S8 (murni algoritma tanpa mengubah poligon):
                if is_outside_s8:
                    passed_gate = False

                # Diagnostic logging pasif untuk target zone pada vehicle_tracks
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
                            # Throttle: 15s untuk ACCEPTED pemilik slot stabil (OCCUPIED), 0.5s untuk lainnya
                            is_stable_occupied_owner = (
                                passed_gate
                                and state.track_id == vt.track_id
                                and state.phase == "OCCUPIED"
                            )
                            throttle_sec = 15.0 if is_stable_occupied_owner else 0.5

                            last_log = self._last_diag_log_time.get(vt.track_id, 0.0)
                            if (current_time - last_log) >= throttle_sec:
                                self._last_diag_log_time[vt.track_id] = current_time
                                from datetime import datetime, timezone
                                iso_now = datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat()
                                cand_score = None
                                if passed_gate:
                                    dist_to_center = math.hypot(cx - slot_cx, cy - slot_cy)
                                    dist_norm = dist_to_center / slot_diag
                                    continuity_bonus = 25.0 if (state.track_id == vt.track_id and state.phase == "OCCUPIED") else 0.0
                                    latch_bonus = 20.0 if state.latch_occupied else 0.0
                                    cand_score = round(
                                        (d_ground * 3.0) - (dist_norm * 30.0) + (iou_slot * 30.0) + (iou_anchor * 30.0) + continuity_bonus + latch_bonus,
                                        2
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
                                import logging
                                logging.getLogger("engine.smart_parking").warning(
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

            # Evaluasi diagnostik pasif terpisah untuk track non-vehicle / unconfirmed (HANYA MEMBACA)
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
                                from datetime import datetime, timezone
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
                            import logging
                            logging.getLogger("engine.smart_parking").warning(
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

        # ── State Transitions Tiap Slot ────────────────────────────────────────
        for slot in active_slots:
            slot_id = slot.zone_id
            state = self.slot_states[slot_id]
            geom = slot_geometries[slot_id]
            pts_scaled = geom["pts_scaled"]

            matched_track = assigned_slot_to_track.get(slot_id)

            if matched_track is not None:
                # Kendaraan terdeteksi di dalam poligon — 3 cabang kepemilikan:
                if state.track_id == matched_track.track_id:
                    # Kasus 1: ID sama — kendaraan yang sama
                    if state.first_seen_time is not None:
                        # 1a: Kendaraan sudah pernah terlihat → akumulasi dwell normal
                        state.dwell_duration = current_time - state.first_seen_time
                    else:
                        # 1b: Kendaraan baru pertama kali masuk poligon (setelah tripwire A_TO_B)
                        state.first_seen_time = current_time
                        state.dwell_duration = 0.0
                    state.vehicle_class = matched_track.class_label
                    state.last_seen_time = current_time
                    state.last_bbox = matched_track.bbox
                    state.polygon_clear_since = None  # reset timer kosong

                elif state.phase in ("OCCUPIED", "ENTERING", "LEAVING") and self._is_same_vehicle(
                    state, matched_track, pts_scaled, current_time
                ):
                    # Kasus 2: ID berbeda TAPI spasial/IoU cocok → ID CHURN
                    # Update ID secara senyap TANPA mereset dwell timer atau fase
                    state.track_id = matched_track.track_id
                    state.vehicle_class = matched_track.class_label
                    state.last_seen_time = current_time
                    state.last_bbox = matched_track.bbox
                    state.polygon_clear_since = None  # reset timer kosong
                    # Pertahankan first_seen_time — hitung ulang dwell dari referensi asal
                    if state.first_seen_time is not None:
                        state.dwell_duration = current_time - state.first_seen_time

                else:
                    # Kasus 3: Kendaraan benar-benar baru (atau slot dari VACANT/kosong)
                    state.track_id = matched_track.track_id
                    state.vehicle_class = matched_track.class_label
                    state.first_seen_time = current_time
                    state.dwell_duration = 0.0
                    state.last_seen_time = current_time
                    state.last_bbox = matched_track.bbox
                    state.polygon_clear_since = None

                if is_warmup:
                    # Evaluasi baseline okupansi slot pada masa cold-start:
                    if matched_track.is_confirmed:
                        state.warmup_hits += 1
                    state.phase = "OCCUPIED"
                    state.dwell_duration = max(state.dwell_duration, self.dwell_threshold_sec)
                    state.leaving_since = None
                    # Latch hanya setelah minimal 10 konfirmasi warmup konsisten dan track confirmed
                    if state.warmup_hits >= 10 and matched_track.is_confirmed:
                        state.latch_occupied = True
                        state.is_warmup_latch = True
                else:
                    if matched_track.is_confirmed:
                        state.is_warmup_latch = False

                    if state.phase == "VACANT":
                        # Fallback dwell (tanpa tripwire crossing terdeteksi)
                        if state.dwell_duration >= self.dwell_threshold_sec:
                            state.phase = "OCCUPIED"
                    elif state.phase == "ENTERING":
                        # Menunggu konfirmasi dwell di dalam slot
                        if state.dwell_duration >= self.dwell_threshold_sec:
                            state.phase = "OCCUPIED"
                    elif state.phase == "LEAVING":
                        # Manuver maju-mundur (re-park debounce dalam window 5 detik)
                        state.phase = "OCCUPIED"
                        state.leaving_since = None
                        state.polygon_clear_since = None
                    elif state.phase == "OCCUPIED":
                        state.leaving_since = None

                    # Latch occupancy jika kendaraan telah terparkir stabil melampaui ambang batas dwell
                    if state.phase == "OCCUPIED" and state.dwell_duration >= state.latch_dwell_threshold_sec:
                        state.latch_occupied = True
                        state.is_warmup_latch = False

            else:
                # Tidak ada kendaraan terdeteksi di dalam poligon
                if is_warmup and not state.latch_occupied:
                    # Selama masa warmup, jika tidak terkonfirmasi lagi dan belum latched, reset ke VACANT
                    state.phase = "VACANT"
                    state.track_id = None
                    state.vehicle_class = None
                    state.first_seen_time = None
                    state.dwell_duration = 0.0
                    state.warmup_hits = 0
                elif state.track_id is not None and state.track_id in assigned_track_ids and not state.latch_occupied:
                    # INSTANT YIELD: Mobil ini terbukti terparkir di slot lain secara eksklusif (Exclusive 1-to-1 Assignment)
                    # Segera bebaskan slot ini kembali ke status VACANT jika belum latched!
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
                    # Timeout jika sinyal masuk sudah lewat tanpa kendaraan masuk poligon
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
                    # Penguatan hysteresis fisik ganda & toleransi oklusi:
                    # Slot hanya boleh LEAVING setelah poligon terbukti kosong >= confirm_threshold.
                    # Gunakan threshold adaptif: 10.0 detik untuk slot latched stabil (tahan oklusi hingga 10s),
                    # atau vacant_confirm_sec (5.0s) normal.
                    confirm_threshold = 10.0 if state.latch_occupied else state.vacant_confirm_sec
                    if state.last_seen_time > 0 and (current_time - state.last_seen_time) > 0.5:
                        # Mulai atau lanjutkan timer poligon kosong
                        if state.polygon_clear_since is None:
                            state.polygon_clear_since = current_time
                        elif not is_warmup and state.is_warmup_latch and (current_time - state.polygon_clear_since) >= 4.0:
                            # Auto-reset KHUSUS warmup phantom latch yang tidak pernah terkonfirmasi pasca-warmup:
                            # paksa reset ke VACANT setelah >= 4.0s untuk memulihkan false latch cold-start.
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
                        # Kendaraan terlihat di frame sebelumnya (baru hilang < 0.5s), reset timer
                        state.polygon_clear_since = None
                elif state.phase == "LEAVING":
                    # Konfirmasi poligon kosong setelah masa leave / grace period
                    leave_time = state.leaving_since if state.leaving_since is not None else state.last_seen_time
                    if leave_time > 0 and (current_time - leave_time) >= state.exit_grace_sec:
                        state.phase = "VACANT"
                        # Reset latch_occupied secara legal HANYA pada transisi final LEAVING -> VACANT
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
                            from datetime import datetime, timezone
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
                            import logging
                            logging.getLogger("engine.smart_parking").warning(
                                f"Diagnostic phase transition logging exception swallowed: {diag_err}"
                            )
                        except Exception:
                            pass

        # ── Maneuvering Corridor Obstruction Engine ───────────────────────────
        # Evaluasi kendaraan terkonfirmasi yang berada di luar seluruh poligon slot S1-S8
        active_tids_in_frame = {vt.track_id for vt in vehicle_tracks}
        for vt in vehicle_tracks:
            # Jika kendaraan ini sudah sah mengokupansi salah satu slot, bukan halangan koridor
            if vt.track_id in assigned_track_ids:
                if vt.track_id in self._corridor_vehicles:
                    del self._corridor_vehicles[vt.track_id]
                continue

            x1, y1, x2, y2 = vt.bbox
            vcx = float((x1 + x2) / 2.0)
            vcy = float((y1 + y2) / 2.0)
            v_stance = (vcx, float(y2))

            # Uji apakah kendaraan ini berada di dalam batas slot poligon manapun
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

            # Kendaraan berada di koridor manuver / aspal jalan di luar petak slot
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
                    # Kendaraan sedang melaju / bermanuver (> 15px shift) -> reset anchor & dwell
                    entry["anchor_centroid"] = curr_pos
                    entry["first_seen_time"] = current_time
                    entry["dwell_duration"] = 0.0
                    entry["is_obstruction"] = False
                else:
                    # Stasioner di jalur manuver (shift <= 15px)
                    entry["dwell_duration"] = current_time - entry["first_seen_time"]
                    if entry["dwell_duration"] >= self.corridor_obstruction_dwell_sec:
                        entry["is_obstruction"] = True

        # Pembersihan entri koridor yang sudah hilang dari frame (> 3 detik)
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

    def render_overlay(
        self,
        canvas: np.ndarray,
        polygons: List[ROIZone],
        tripwires: List[TripwireRule],
        tracks: List[TrackResult],
        scale_x: float,
        scale_y: float,
        parking_stats: Dict[str, Any],
        current_time: float = 0.0,
        fps: float = 20.0,
        camera_id: str = "cam_01",
        camera_name: str = "Koridor Utama",
    ) -> None:
        """
        Render visual overlay CCTV Minimalist (Capacity & Slot Status Only):
        1. Bounding Box & Vehicle Labels DIHAPUS penuh (declutter visual lantai).
        2. Titik roda, panah arah, titik P1/P2, dan teks label tripwire DIHAPUS.
        3. Tripwire: Garis hairline 1px tipis oranye/amber tanpa embel-embel teks.
        4. Poligon Petak Parkir:
           - VACANT (Kosong): Outline warna Cyan/Emerald transparan halus tanpa teks di aspal.
           - OCCUPIED (Terisi): Outline Merah tipis + fill transparan lembut (alpha=0.10)
             + mini label di centroid: [S{num}] (fontScale=0.35).
        5. Capacity HUD: 1 baris ringkas di pojok kanan atas:
           PARKING: {occupied}/{total} OCCUPIED | TERISI: [{occupied_slots_str}]
        """
        if parking_stats.get("parking_mode") == "motorcycle_block" or self.parking_mode == "motorcycle_block":
            self._render_block_overlay(
                canvas=canvas,
                polygons=polygons,
                tripwires=tripwires,
                tracks=tracks,
                scale_x=scale_x,
                scale_y=scale_y,
                parking_stats=parking_stats,
                current_time=current_time,
            )
            return

        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        total_slots = parking_stats.get("total_slots", len(polygons))
        slot_states = parking_stats.get("slot_states", self.slot_states)
        occupied_slots = parking_stats.get(
            "occupied_slots",
            sum(1 for s in slot_states.values() if s.phase == "OCCUPIED")
        )
        available_slots = parking_stats.get("available_slots", max(0, total_slots - occupied_slots))

        active_slots = [p for p in polygons if p.active]

        # ── 1. Minimalist Slot Polygons (1px Outline & Centroid Label) ──
        occupied_polys = []
        occupied_badges = []

        for slot in active_slots:
            pts_scaled = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )

            state = slot_states.get(slot.zone_id)
            phase = state.phase if state else "VACANT"

            if phase == "OCCUPIED":
                # Outline 1px Merah tipis + Fill transparan 0.10 alpha
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                occupied_polys.append(pts_scaled)
                # Mini label kecil di centroid: [S{num}]
                cx = int(np.mean([pt[0] for pt in pts_scaled]))
                cy = int(np.mean([pt[1] for pt in pts_scaled]))
                occupied_badges.append((cx, cy, f"[S{state.slot_num}]"))
            elif phase == "ENTERING":
                # Outline 1px Amber halus
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 215, 255), thickness=1, lineType=cv2.LINE_AA)
            elif phase == "LEAVING":
                # Outline 1px Kuning tipis
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                # Slot VACANT: outline 1px Cyan/Emerald transparan halus tanpa teks apapun di aspal
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(220, 220, 0), thickness=1, lineType=cv2.LINE_AA)

        # Arsiran fill transparan lembut pada slot OCCUPIED (alpha = 0.10)
        if occupied_polys:
            overlay = canvas.copy()
            for opp in occupied_polys:
                cv2.fillPoly(overlay, [opp], color=(0, 0, 220))
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # Mini label kecil di centroid slot OCCUPIED: [S{num}]
        for cx, cy, text in occupied_badges:
            self._draw_slot_badge(canvas, text, cx, cy)

        # ── 2. Minimalist Tripwire (Hairline 1px Tipis Oranye/Amber) ──
        active_tripwires = [tw for tw in tripwires if tw.active]
        for tw in active_tripwires:
            p1_scaled = (int(tw.p1.x * sx), int(tw.p1.y * sy))
            p2_scaled = (int(tw.p2.x * sx), int(tw.p2.y * sy))

            flash_until = self.tripwire_flash.get(tw.tripwire_id, 0.0)
            is_flashing = (current_time > 0 and current_time < flash_until)
            tw_color = (255, 255, 255) if is_flashing else (0, 165, 255)

            # Cukup garis hairline 1px tipis oranye/amber tanpa embel-embel teks/titik
            cv2.line(canvas, p1_scaled, p2_scaled, tw_color, 1, cv2.LINE_AA)

        # ── 3. Minimalist Capacity HUD (1 Baris Ringkas Pojok Kanan Atas) ──
        occ_slots = [
            f"S{s.slot_num}"
            for s in sorted(slot_states.values(), key=lambda x: x.slot_num)
            if s.phase == "OCCUPIED"
        ]
        occ_text = ", ".join(occ_slots) if occ_slots else "-"

        hud_line = f"PARKING: {occupied_slots}/{total_slots} OCCUPIED | TERISI: [{occ_text}]"
        self._draw_hud_pill_top_right(canvas, hud_line, available_slots)

    def _draw_hud_pill_top_right(
        self,
        canvas: np.ndarray,
        line: str,
        available: int,
    ) -> None:
        """
        Render status HUD di POJOK KANAN ATAS frame dengan Dark Semi-Transparent Pill
        (1 baris ringkas).
        Pojok kiri atas sengaja dikosongkan agar OSD timestamp asli kamera terlihat jelas.
        """
        font_sc = 0.35
        font_thick = 1
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), _ = cv2.getTextSize(line, font_face, font_sc, font_thick)

        pad_h, pad_v = 10, 6
        pill_w = tw + pad_h * 2
        pill_h = th + pad_v * 2

        margin_right = 8
        margin_top = 8

        x2 = canvas.shape[1] - margin_right
        x1 = max(0, x2 - pill_w)
        y1 = margin_top
        y2 = min(canvas.shape[0] - 1, y1 + pill_h)

        # Background semi-transparan gelap (alpha 0.75)
        sub = canvas[y1:y2, x1:x2]
        bg_color = np.full_like(sub, (16, 20, 28), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.25, bg_color, 0.75, 0)

        # Border tipis 1px
        border_col = (0, 220, 100) if available > 0 else (0, 60, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1, cv2.LINE_AA)

        # Teks 1 baris
        tx = x1 + pad_h
        ty = y1 + pad_v + th
        text_col = (0, 235, 120) if available > 0 else (0, 80, 255)
        cv2.putText(canvas, line, (tx, ty), font_face, font_sc, text_col, font_thick, cv2.LINE_AA)

    def _render_block_overlay(
        self,
        canvas: np.ndarray,
        polygons: List[ROIZone],
        tripwires: List[TripwireRule],
        tracks: List[TrackResult],
        scale_x: float,
        scale_y: float,
        parking_stats: Dict[str, Any],
        current_time: float = 0.0,
    ) -> None:
        """
        Render visual overlay CCTV Minimalist untuk Block Motor Parking (cam_03):
        1. Poligon area: outline tebal 2px hijau jika ada slot, merah jika penuh + fill 0.10 alpha.
        2. Centroid label: "{occupied}/{total} MOTOR".
        3. Tripwire: garis hairline 1px tipis oranye/amber (flash 1.0s putih).
        4. HUD 2-baris di pojok kanan atas:
           Baris 1: PARKING: {occupied}/{total} TERISI | KOSONG: {available} UNIT
           Baris 2: IN: {gate_in} | OUT: {gate_out}
        """
        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        occupied = parking_stats.get("occupied_slots", 0)
        total = parking_stats.get("total_slots", self.block_capacity)
        available = parking_stats.get("available_slots", max(0, total - occupied))
        gate_in = parking_stats.get("gate_in", self.gate_in_count)
        gate_out = parking_stats.get("gate_out", self.gate_out_count)

        zone_color = (0, 220, 100) if available > 0 else (0, 60, 255)

        active_slots = [p for p in polygons if p.active]
        for slot in active_slots:
            pts = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )
            # Outline tebal 2px
            cv2.polylines(canvas, [pts], isClosed=True, color=zone_color, thickness=2, lineType=cv2.LINE_AA)

            # Fill transparan 0.10 alpha
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color=zone_color)
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # ── Visualisasi Bounding Box & Badge Motor Terdeteksi di Dalam Poligon ──
        if active_slots:
            pts_zone = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in active_slots[0].points],
                dtype=np.int32,
            )
            drum_excl_x_render = int(self._block_exclusion_x_1080p * sx) if self._block_exclusion_x_1080p is not None else None

            # Visualisasi kotak motor: Utamakan unit ter-latch dari memori temporal agar stabil tanpa flicker
            # Filter render: tampilkan unit stabil dari Spatial Slot Anchor Memory (TTL=10 detik, konfirmasi stabil >= 2s)
            if self._stationary_motor_units:
                render_units = sorted(
                    self._stationary_motor_units.values(),
                    key=lambda u: u.centroid[0],
                )
                render_items = [
                    (u.bbox, u.confidence)
                    for u in render_units
                    if (u.is_latched or (self.stationary_dwell_sec == 0.0) or (current_time > 0 and (current_time - u.first_seen_time) >= 2.0))
                    and (current_time <= 0 or (current_time - u.last_seen_time) <= 10.0)
                    and is_valid_motorcycle_anatomy(u.bbox, sx, sy)
                    and (drum_excl_x_render is None or u.centroid[0] >= drum_excl_x_render)
                ]
            else:
                candidate_render: List[TrackResult] = []
                for track in tracks:
                    if track.class_label not in self.vehicle_classes:
                        continue

                    rx1, ry1, rx2, ry2 = track.bbox
                    cx = float((rx1 + rx2) / 2.0)
                    wheel_y = float(ry2)
                    wheel_pt = (int(cx), int(wheel_y))

                    if drum_excl_x_render is not None and cx < drum_excl_x_render:
                        continue

                    if not is_valid_motorcycle_anatomy(track.bbox, sx, sy):
                        continue

                    wheel_in = cv2.pointPolygonTest(pts_zone, wheel_pt, False) >= 0
                    if not wheel_in:
                        continue

                    overlap_ratio = bbox_polygon_overlap_ratio(track.bbox, pts_zone)
                    if overlap_ratio < 0.20:
                        continue

                    candidate_render.append(track)

                valid_render = deduplicate_motorcycle_tracks(
                    candidate_render,
                    iou_thresh=0.30,
                    ios_thresh=0.50,
                    max_centroid_dist_px=28.0,
                )
                render_items = [(t.bbox, t.confidence) for t in valid_render]

            for motor_idx, (r_bbox, r_conf) in enumerate(render_items, 1):
                x1, y1, x2, y2 = [int(round(v)) for v in r_bbox]
                cx = float((r_bbox[0] + r_bbox[2]) / 2.0)
                wheel_y = float(r_bbox[3])

                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

                cv2.circle(canvas, (int(cx), int(wheel_y)), 2, (0, 255, 0), -1, cv2.LINE_AA)

                badge_text = f"#{motor_idx} ({r_conf:.2f})"

                font_scale = 0.32
                font_thick = 1
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(badge_text, font_face, font_scale, font_thick)

                bx1 = max(0, x1)
                by1 = max(0, y1 - th - 3)
                bx2 = min(canvas.shape[1] - 1, bx1 + tw + 4)
                by2 = min(canvas.shape[0] - 1, by1 + th + 3)

                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 24, 32), -1)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    badge_text,
                    (bx1 + 2, by1 + th),
                    font_face,
                    font_scale,
                    (0, 255, 255),
                    font_thick,
                    cv2.LINE_AA,
                )

        # ── 100% SINKRONISASI VISUAL-TELEMETRI MUTLAK ──────────────────────────
        # `occupied` dibaca dari parking_stats yang sudah dihitung di _update_block_mode
        # (termasuk tripwire dominance). Render TIDAK menimpa nilai ini.
        # Kotak kuning = render_items (latched units); badge/HUD = occupied dari stats.
        # Single source of truth → visual boxes ≡ badge ≡ HUD ≡ API payload.

        # Label kapasitas di centroid area poligon (digambar setelah box agar selalu di atas)
        for slot in active_slots:
            pts = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )
            cx = int(np.mean([pt[0] for pt in pts]))
            cy = int(np.mean([pt[1] for pt in pts]))
            label = f"{occupied}/{total} MOTOR"
            self._draw_slot_badge(canvas, label, cx, cy)

        # Tripwire render (dengan flash)
        active_tripwires = [tw for tw in tripwires if tw.active]
        for tw in active_tripwires:
            p1_scaled = (int(tw.p1.x * sx), int(tw.p1.y * sy))
            p2_scaled = (int(tw.p2.x * sx), int(tw.p2.y * sy))
            flash_until = self.tripwire_flash.get(tw.tripwire_id, 0.0)
            is_flashing = (current_time > 0 and current_time < flash_until)
            tw_color = (255, 255, 255) if is_flashing else (0, 165, 255)
            cv2.line(canvas, p1_scaled, p2_scaled, tw_color, 1, cv2.LINE_AA)

        # HUD 1-baris di pojok kanan atas
        hud_line = f"PARKING: {occupied}/{total} TERISI | KOSONG: {available} UNIT"
        self._draw_hud_pill_top_right(canvas, hud_line, available)

    def _draw_hud_pill_top_right_dual(
        self,
        canvas: np.ndarray,
        line1: str,
        line2: str,
        available: int,
    ) -> None:
        """Render HUD 2-baris di pojok kanan atas dengan Dark Semi-Transparent Pill."""
        font_sc = 0.35
        font_thick = 1
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        (tw1, th1), _ = cv2.getTextSize(line1, font_face, font_sc, font_thick)
        (tw2, th2), _ = cv2.getTextSize(line2, font_face, font_sc, font_thick)

        pad_h, pad_v = 10, 6
        line_gap = 4
        pill_w = max(tw1, tw2) + pad_h * 2
        pill_h = th1 + th2 + pad_v * 2 + line_gap

        margin_right = 8
        margin_top = 8

        x2 = canvas.shape[1] - margin_right
        x1 = max(0, x2 - pill_w)
        y1 = margin_top
        y2 = min(canvas.shape[0] - 1, y1 + pill_h)

        sub = canvas[y1:y2, x1:x2]
        bg_color = np.full_like(sub, (16, 20, 28), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.25, bg_color, 0.75, 0)

        border_col = (0, 220, 100) if available > 0 else (0, 60, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1, cv2.LINE_AA)

        tx = x1 + pad_h
        ty1 = y1 + pad_v + th1
        ty2 = ty1 + th2 + line_gap
        text_col1 = (0, 235, 120) if available > 0 else (0, 80, 255)
        text_col2 = (180, 180, 180)
        cv2.putText(canvas, line1, (tx, ty1), font_face, font_sc, text_col1, font_thick, cv2.LINE_AA)
        cv2.putText(canvas, line2, (tx, ty2), font_face, font_sc, text_col2, font_thick, cv2.LINE_AA)


    def _draw_slot_badge(
        self,
        canvas: np.ndarray,
        text: str,
        cx: int,
        cy: int,
    ) -> None:
        """Render mini label kecil di centroid petak parkir OCCUPIED: [S{num}] (fontScale=0.35)."""
        font_scale = 0.35
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pad_h, pad_v = 4, 3
        bx1 = max(2, cx - tw // 2 - pad_h)
        by1 = max(2, cy - th // 2 - pad_v)
        bx2 = min(canvas.shape[1] - 3, bx1 + tw + pad_h * 2)
        by2 = min(canvas.shape[0] - 3, by1 + th + pad_v * 2)

        # Semi-transparan dark background dengan border merah 1px
        sub = canvas[by1:by2, bx1:bx2]
        dark_bg = np.full_like(sub, (15, 15, 30), dtype=np.uint8)
        canvas[by1:by2, bx1:bx2] = cv2.addWeighted(sub, 0.25, dark_bg, 0.75, 0)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            text,
            (bx1 + pad_h, by1 + pad_v + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    def _draw_micro_chevron(
        self,
        canvas: np.ndarray,
        p1_x: int,
        p1_y: int,
        p2_x: int,
        p2_y: int,
        direction: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Gambarkan chevron mikro ramping (ukuran ~5px) di tengah garis tripwire."""
        dx = float(p2_x - p1_x)
        dy = float(p2_y - p1_y)
        length = max(1.0, np.hypot(dx, dy))
        nx = -dy / length
        ny = dx / length
        ux = dx / length
        uy = dy / length

        mid_x = (p1_x + p2_x) / 2.0
        mid_y = (p1_y + p2_y) / 2.0
        chev_depth = 5.0
        chev_wing = 4.0

        if direction in ("A_TO_B", "BOTH"):
            tip_x = int(mid_x + nx * chev_depth)
            tip_y = int(mid_y + ny * chev_depth)
            w1_x = int(mid_x - ux * chev_wing)
            w1_y = int(mid_y - uy * chev_wing)
            w2_x = int(mid_x + ux * chev_wing)
            w2_y = int(mid_y + uy * chev_wing)
            cv2.line(canvas, (w1_x, w1_y), (tip_x, tip_y), color, 1, cv2.LINE_AA)
            cv2.line(canvas, (w2_x, w2_y), (tip_x, tip_y), color, 1, cv2.LINE_AA)

        if direction in ("B_TO_A", "BOTH"):
            tip_x = int(mid_x - nx * chev_depth)
            tip_y = int(mid_y - ny * chev_depth)
            w1_x = int(mid_x - ux * chev_wing)
            w1_y = int(mid_y - uy * chev_wing)
            w2_x = int(mid_x + ux * chev_wing)
            w2_y = int(mid_y + uy * chev_wing)
            cv2.line(canvas, (w1_x, w1_y), (tip_x, tip_y), color, 1, cv2.LINE_AA)
            cv2.line(canvas, (w2_x, w2_y), (tip_x, tip_y), color, 1, cv2.LINE_AA)

    def _draw_mini_badge(
        self,
        canvas: np.ndarray,
        text: str,
        cx: int,
        cy: int,
        outline_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int] = (15, 15, 45),
    ) -> None:
        """Helper backward-compatible render mini-badge."""
        self._draw_slot_badge(canvas, text, cx, cy)

    def _draw_telemetry_banner(
        self, canvas: np.ndarray, text: str, available: int, total: int
    ) -> None:
        """Helper backward-compatible telemetry banner."""
        pass
