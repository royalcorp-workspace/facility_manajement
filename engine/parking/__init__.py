"""
engine.parking
==============
Modular smart parking package with separation of concerns:
- base: Shared dataclasses (SlotState, StationaryMotorUnit)
- spatial: Geometric calculations (IoU, IoS, overlap ratio)
- car_slot: Dedicated car parking slot tracker for cam_01 (CarSlotTracker)
- motorcycle_block: Dedicated dense motorcycle block tracker for cam_03 (MotorcycleBlockTracker)
- visualizer: OpenCV rendering utilities and HUD overlay components (ParkingVisualizer)
- facade: Unified SmartParkingTracker facade preserving public API contracts
"""

from __future__ import annotations

from engine.geometry import bbox_iou
from engine.parking.base import SlotState, StationaryMotorUnit
from engine.parking.car_slot import CarSlotTracker
from engine.parking.facade import SmartParkingTracker
from engine.parking.motorcycle_block import (
    MotorcycleBlockTracker,
    deduplicate_motorcycle_tracks,
    is_valid_motorcycle_anatomy,
)
from engine.parking.spatial import bbox_ios, bbox_polygon_overlap_ratio
from engine.parking.visualizer import ParkingVisualizer

__all__ = [
    "SmartParkingTracker",
    "SlotState",
    "StationaryMotorUnit",
    "CarSlotTracker",
    "MotorcycleBlockTracker",
    "ParkingVisualizer",
    "deduplicate_motorcycle_tracks",
    "is_valid_motorcycle_anatomy",
    "bbox_polygon_overlap_ratio",
    "bbox_ios",
    "bbox_iou",
]
