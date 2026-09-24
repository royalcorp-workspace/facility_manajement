"""
Unit & Integration Test untuk Initial Cold-Start Scan, Motion Gating Bypass,
dan Baseline Occupancy Evaluation pada Smart Parking.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from typing import List

from engine.config_loader import ROIPoint, ROIZone
from engine.tracker_interface import TrackResult
from engine.smart_parking import SmartParkingTracker


def create_7_parking_slots() -> List[ROIZone]:
    """Buat 7 poligon slot parkir (zone_01 s.d zone_07)."""
    slots = []
    # Koordinat 1080p horizontal berdampingan
    # Slot 1 s.d 7 berjejer di y: 400..700
    for i in range(1, 8):
        x_left = 100 + (i - 1) * 200
        x_right = x_left + 180
        pts = [
            ROIPoint(x=float(x_left), y=400.0),
            ROIPoint(x=float(x_right), y=400.0),
            ROIPoint(x=float(x_right), y=700.0),
            ROIPoint(x=float(x_left), y=700.0),
        ]
        slots.append(
            ROIZone(
                zone_id=f"zone_{i:02d}",
                type="polygon",
                color_hex="#00FF00",
                points=pts,
                trigger_on=["enter"],
                active=True,
                label=f"Slot {i:02d}",
            )
        )
    return slots


def test_cold_start_baseline_occupancy():
    print("\n[TEST 1] Cold-Start Baseline Occupancy Evaluation...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slots = create_7_parking_slots()

    # 6 mobil terparkir di Slot 1, 2, 4, 5, 6, 7 (Slot 3 kosong)
    # Roda (bottom_center) berada di tengah slot y=680
    scale_x = 3.0  # 1920 / 640
    scale_y = 3.0  # 1080 / 360

    # Di canvas AI (640x360), koordinat dibagi 3
    # Misal slot 1: x_left=100 -> AI x=33..93, y=133..233
    occupied_slot_indices = [1, 2, 4, 5, 6, 7]
    tracks = []
    for idx in occupied_slot_indices:
        ai_x1 = (100 + (idx - 1) * 200 + 20) / 3.0
        ai_x2 = (100 + (idx - 1) * 200 + 160) / 3.0
        ai_y1 = 450.0 / 3.0
        ai_y2 = 680.0 / 3.0
        tracks.append(
            TrackResult(
                track_id=idx * 10,
                bbox=(ai_x1, ai_y1, ai_x2, ai_y2),
                confidence=0.85,
                class_label="car",
                class_id=2,
                age=1,
                is_confirmed=False,  # Unconfirmed pada frame awal!
            )
        )

    # Frame 1: Cold start warmup (is_warmup=True)
    stats = tracker.update(
        tracks=tracks,
        polygons=slots,
        scale_x=scale_x,
        scale_y=scale_y,
        current_time=100.0,
        is_warmup=True,
    )

    print(f"  Total Slots    : {stats['total_slots']}")
    print(f"  Occupied Count : {stats['occupied_slots']}")
    print(f"  Available Count: {stats['available_slots']}")
    print(f"  Occupied Slots : {[s.slot_id for s in tracker.slot_states.values() if s.occupied]}")

    assert stats["total_slots"] == 7, f"Expected 7 total slots, got {stats['total_slots']}"
    assert stats["occupied_slots"] == 6, f"Expected 6 occupied slots, got {stats['occupied_slots']}"
    assert stats["available_slots"] == 1, f"Expected 1 available slot, got {stats['available_slots']}"
    assert not tracker.slot_states["zone_03"].occupied, "Slot 03 harus kosong (VACANT)"
    assert tracker.slot_states["zone_01"].phase == "OCCUPIED"
    assert tracker.slot_states["zone_02"].phase == "OCCUPIED"
    assert tracker.slot_states["zone_04"].phase == "OCCUPIED"
    assert tracker.slot_states["zone_05"].phase == "OCCUPIED"
    assert tracker.slot_states["zone_06"].phase == "OCCUPIED"
    assert tracker.slot_states["zone_07"].phase == "OCCUPIED"

    print("  ✓ PASS: Seluruh 6 mobil terparkir langsung berstatus OCCUPIED (1/7 AVAILABLE).")


def test_post_warmup_persistence_with_coasting():
    print("\n[TEST 2] Post-Warmup Persistence & Coasting Mode...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slots = create_7_parking_slots()
    scale_x = 3.0
    scale_y = 3.0

    occupied_slot_indices = [1, 2, 4, 5, 6, 7]
    tracks = []
    for idx in occupied_slot_indices:
        ai_x1 = (100 + (idx - 1) * 200 + 20) / 3.0
        ai_x2 = (100 + (idx - 1) * 200 + 160) / 3.0
        ai_y1 = 450.0 / 3.0
        ai_y2 = 680.0 / 3.0
        tracks.append(
            TrackResult(
                track_id=idx * 10,
                bbox=(ai_x1, ai_y1, ai_x2, ai_y2),
                confidence=0.85,
                class_label="car",
                class_id=2,
                age=30,
                is_confirmed=True,
            )
        )

    # 1. Warmup initial
    tracker.update(tracks, slots, scale_x, scale_y, current_time=100.0, is_warmup=True)

    # 2. Simulasi 10 detik kemudian (is_warmup=False, tracks tetap ada dari coasting)
    stats = tracker.update(tracks, slots, scale_x, scale_y, current_time=110.0, is_warmup=False)

    assert stats["occupied_slots"] == 6
    assert stats["available_slots"] == 1
    assert tracker.slot_states["zone_01"].phase == "OCCUPIED"
    print("  ✓ PASS: Status OCCUPIED bertahan sempurna setelah masa warmup selesai.")


def main():
    print("=" * 60)
    print("TEST SUITE: COLD-START SCAN & BASELINE OCCUPANCY")
    print("=" * 60)
    test_cold_start_baseline_occupancy()
    test_post_warmup_persistence_with_coasting()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
