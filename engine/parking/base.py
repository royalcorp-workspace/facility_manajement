"""
engine.parking.base
===================
Shared dataclasses and state models for smart parking.
Includes:
- SlotState: State representation for single car parking slots (cam_01).
- StationaryMotorUnit: State representation for clustered stationary motorcycles (cam_03).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


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
