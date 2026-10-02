from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from engine.config_loader import ROIZone, TripwireRule
from engine.parking.base import SlotState, StationaryMotorUnit
from engine.parking.car_slot import CarSlotTracker
from engine.parking.motorcycle_block import (
    MotorcycleBlockTracker,
    deduplicate_motorcycle_tracks,
    is_valid_motorcycle_anatomy,
)
from engine.parking.spatial import bbox_ios, bbox_polygon_overlap_ratio
from engine.parking.visualizer import ParkingVisualizer
from engine.tracker_interface import TrackResult


class SmartParkingTracker:
    DEFAULT_VEHICLE_CLASSES = {"car", "truck", "bus"}

    def __init__(
        self,
        total_slots: Optional[int] = None,
        dwell_threshold_sec: float = 3.0,
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
        acquisition_conf_thresh: float = 0.25,
        retention_conf_thresh: float = 0.20,
        wheel_contact_margin_px: float = 0.0,
        corridor_obstruction_dwell_sec: float = 60.0,
        corridor_shift_threshold_px: float = 15.0,
    ) -> None:
        self.parking_mode = parking_mode
        self.block_capacity = block_capacity
        self.stationary_dwell_sec = stationary_dwell_sec
        self._block_exclusion_x_1080p = block_exclusion_x_1080p
        self.dwell_threshold_sec = dwell_threshold_sec

        if vehicle_classes is not None:
            self.vehicle_classes = set(vehicle_classes)
        elif parking_mode == "motorcycle_block":
            self.vehicle_classes = {"motorcycle", "bicycle"}
        else:
            self.vehicle_classes = set(self.DEFAULT_VEHICLE_CLASSES)

        if self.parking_mode == "motorcycle_block":
            self._motorcycle_tracker = MotorcycleBlockTracker(
                block_capacity=block_capacity,
                stationary_dwell_sec=stationary_dwell_sec,
                block_exclusion_x_1080p=block_exclusion_x_1080p,
                vehicle_classes=self.vehicle_classes,
            )
            self._car_tracker = None
        else:
            self._motorcycle_tracker = None
            self._car_tracker = CarSlotTracker(
                total_slots=total_slots,
                dwell_threshold_sec=dwell_threshold_sec,
                vehicle_classes=self.vehicle_classes,
                acquisition_conf_thresh=acquisition_conf_thresh,
                retention_conf_thresh=retention_conf_thresh,
                wheel_contact_margin_px=wheel_contact_margin_px,
                corridor_obstruction_dwell_sec=corridor_obstruction_dwell_sec,
                corridor_shift_threshold_px=corridor_shift_threshold_px,
                debug_diagnostics=debug_diagnostics,
                debug_target_zone=debug_target_zone,
                debug_log_path=debug_log_path,
                debug_snapshots=debug_snapshots,
                debug_snapshot_max_files=debug_snapshot_max_files,
                debug_snapshot_min_interval_s=debug_snapshot_min_interval_s,
                debug_snapshot_dir=debug_snapshot_dir,
            )

        self._visualizer = ParkingVisualizer(
            parking_mode=self.parking_mode,
            block_capacity=self.block_capacity,
            block_exclusion_x_1080p=self._block_exclusion_x_1080p,
            stationary_dwell_sec=self.stationary_dwell_sec,
            vehicle_classes=self.vehicle_classes,
        )

    # ── Property Forwarding ──────────────────────────────────────────────
    @property
    def total_slots(self) -> int:
        if self._motorcycle_tracker is not None:
            return self._motorcycle_tracker.total_slots
        return self._car_tracker.total_slots

    @property
    def occupied_slots(self) -> int:
        if self._motorcycle_tracker is not None:
            return self._motorcycle_tracker.occupied_slots
        return self._car_tracker.occupied_slots

    @property
    def available_slots(self) -> int:
        if self._motorcycle_tracker is not None:
            return self._motorcycle_tracker.available_slots
        return self._car_tracker.available_slots

    @property
    def slot_states(self) -> Dict[str, SlotState]:
        if self._car_tracker is not None:
            return self._car_tracker.slot_states
        return {}

    @slot_states.setter
    def slot_states(self, value: Dict[str, SlotState]) -> None:
        if self._car_tracker is not None:
            self._car_tracker.slot_states = value

    @property
    def gate_in_count(self) -> int:
        return self._car_tracker.gate_in_count if self._car_tracker else 0

    @gate_in_count.setter
    def gate_in_count(self, val: int) -> None:
        if self._car_tracker:
            self._car_tracker.gate_in_count = val

    @property
    def gate_out_count(self) -> int:
        return self._car_tracker.gate_out_count if self._car_tracker else 0

    @gate_out_count.setter
    def gate_out_count(self, val: int) -> None:
        if self._car_tracker:
            self._car_tracker.gate_out_count = val

    @property
    def tripwire_flash(self) -> Dict[str, float]:
        return self._car_tracker.tripwire_flash if self._car_tracker else {}

    @tripwire_flash.setter
    def tripwire_flash(self, val: Dict[str, float]) -> None:
        if self._car_tracker:
            self._car_tracker.tripwire_flash = val

    @property
    def has_obstruction(self) -> bool:
        return self._car_tracker.has_obstruction if self._car_tracker else False

    @property
    def active_obstructions(self) -> List[Dict[str, Any]]:
        return self._car_tracker.active_obstructions if self._car_tracker else []

    @property
    def _stationary_motor_units(self) -> Dict[int, StationaryMotorUnit]:
        if self._motorcycle_tracker is not None:
            return self._motorcycle_tracker._stationary_motor_units
        return {}

    @_stationary_motor_units.setter
    def _stationary_motor_units(self, val: Dict[int, StationaryMotorUnit]) -> None:
        if self._motorcycle_tracker is not None:
            self._motorcycle_tracker._stationary_motor_units = val

    @property
    def _density_history(self):
        if self._motorcycle_tracker is not None:
            return self._motorcycle_tracker._density_history
        return []

    @property
    def _last_stable_occupied(self) -> Optional[int]:
        if self._motorcycle_tracker is not None:
            return self._motorcycle_tracker._last_stable_occupied
        return None

    @property
    def _last_candidates(self) -> List[Tuple[float, str, int]]:
        return self._car_tracker._last_candidates if self._car_tracker else []

    @property
    def _last_assigned_slots(self) -> Dict[str, int]:
        return self._car_tracker._last_assigned_slots if self._car_tracker else {}

    # ── Method Forwarding ────────────────────────────────────────────────
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
            if self._motorcycle_tracker is None:
                self._motorcycle_tracker = MotorcycleBlockTracker(
                    block_capacity=self.block_capacity,
                    stationary_dwell_sec=self.stationary_dwell_sec,
                    block_exclusion_x_1080p=self._block_exclusion_x_1080p,
                    vehicle_classes=self.vehicle_classes,
                )
            return self._motorcycle_tracker.update(
                tracks=tracks,
                polygons=polygons,
                scale_x=scale_x,
                scale_y=scale_y,
                current_time=current_time,
                is_warmup=is_warmup,
            )
        else:
            return self._car_tracker.update(
                tracks=tracks,
                polygons=polygons,
                scale_x=scale_x,
                scale_y=scale_y,
                current_time=current_time,
                is_warmup=is_warmup,
                frame_count=frame_count,
                raw_ai_frame=raw_ai_frame,
                raw_detections=raw_detections,
            )

    def apply_tripwire_signal(
        self,
        slot_id: str,
        direction: str,
        track_id: int,
        current_time: float,
    ) -> None:
        if self._car_tracker is not None:
            self._car_tracker.apply_tripwire_signal(
                slot_id=slot_id,
                direction=direction,
                track_id=track_id,
                current_time=current_time,
            )

    def record_gate_crossing(
        self,
        direction_or_note: str,
        tripwire_id: str = "",
        timestamp: float = 0.0,
    ) -> None:
        if self._car_tracker is not None:
            self._car_tracker.record_gate_crossing(
                direction_or_note=direction_or_note,
                tripwire_id=tripwire_id,
                timestamp=timestamp,
            )

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
        self._visualizer.render_overlay(
            canvas=canvas,
            polygons=polygons,
            tripwires=tripwires,
            tracks=tracks,
            scale_x=scale_x,
            scale_y=scale_y,
            parking_stats=parking_stats,
            current_time=current_time,
            fps=fps,
            camera_id=camera_id,
            camera_name=camera_name,
            tripwire_flash=self.tripwire_flash,
            stationary_motor_units=self._stationary_motor_units,
            slot_states=self.slot_states,
            gate_in_count=self.gate_in_count,
            gate_out_count=self.gate_out_count,
        )

    def close(self) -> None:
        if self._car_tracker is not None:
            self._car_tracker.close()

    # ── Backward-compatible Drawing Helpers ──────────────────────────────
    def _draw_hud_pill_top_right(self, canvas: np.ndarray, line: str, available: int) -> None:
        self._visualizer._draw_hud_pill_top_right(canvas, line, available)

    def _draw_hud_pill_top_right_dual(
        self, canvas: np.ndarray, line1: str, line2: str, available: int
    ) -> None:
        self._visualizer._draw_hud_pill_top_right_dual(canvas, line1, line2, available)

    def _draw_slot_badge(self, canvas: np.ndarray, text: str, cx: int, cy: int) -> None:
        self._visualizer._draw_slot_badge(canvas, text, cx, cy)

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
        self._visualizer._draw_micro_chevron(canvas, p1_x, p1_y, p2_x, p2_y, direction, color)

    def _draw_mini_badge(
        self,
        canvas: np.ndarray,
        text: str,
        cx: int,
        cy: int,
        outline_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int] = (15, 15, 45),
    ) -> None:
        self._visualizer._draw_mini_badge(canvas, text, cx, cy, outline_color, bg_color)

    def _draw_telemetry_banner(
        self, canvas: np.ndarray, text: str, available: int, total: int
    ) -> None:
        self._visualizer._draw_telemetry_banner(canvas, text, available, total)

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
        self._visualizer._render_block_overlay(
            canvas=canvas,
            polygons=polygons,
            tripwires=tripwires,
            tracks=tracks,
            scale_x=scale_x,
            scale_y=scale_y,
            parking_stats=parking_stats,
            current_time=current_time,
            stationary_motor_units=self._stationary_motor_units,
            gate_in_count=self.gate_in_count,
            gate_out_count=self.gate_out_count,
        )
