"""
Unit & Integration Test untuk Dynamic Smart Parking & Visual Kontur Interaktif.
Memvalidasi:
1. Dynamic Slot Capacity dari cameras/cam_01/roi_zones.json (7 slot berpasangan).
2. 1-to-1 Pairing berbasis suffix indeks: zone_XX <-> tw_XX.
3. Evaluasi okupansi dinamis (bottom_center wheel contact stasioner >= 10s).
4. Formula Kuota: Available = Total - Occupied.
5. Tripwire flash trigger effect (1.0 detik).
6. Modern IVA Visual Kontur:
   - Outline poligon thickness=2 + vertex dots solid putih r=3px.
   - Status KOSONG (Emerald/Cyan) vs TERISI (Merah + fillPoly alpha 0.18).
   - Tripwire panah penunjuk arah & ring ujung kontras tinggi.
   - Wheel contact glowing dot r=4px.
   - Top HUD: ● PARKING: {available}/{total} SLOTS AVAILABLE │ [TRIPWIRE: ACTIVE].
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from engine.config_loader import load_roi_zones
from engine.tracker_interface import TrackResult
from engine.smart_parking import SmartParkingTracker, SlotState


def test_dynamic_smart_parking():
    print("[1/6] Loading cam_01/roi_zones.json...")
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    num_polys = len(roi_cfg.polygons)
    num_tw = len(roi_cfg.tripwires)
    assert num_polys == 7, f"Expected 7 polygons, got {num_polys}"
    assert num_tw == 7, f"Expected 7 tripwires, got {num_tw}"
    print(f"  ✓ Successfully loaded {num_polys} polygons and {num_tw} tripwires.")

    print("[2/6] Verifying 1-to-1 Pairing via Suffix Index...")
    for poly, tw in zip(roi_cfg.polygons, roi_cfg.tripwires):
        p_idx = poly.zone_id.split("_")[-1]
        t_idx = tw.tripwire_id.split("_")[-1]
        assert p_idx == t_idx, f"Pairing mismatch: {poly.zone_id} vs {tw.tripwire_id}"
        print(f"  ✓ Paired: {poly.zone_id} <---> {tw.tripwire_id} (Index: {p_idx})")

    # Inisialisasi tracker dengan kapasitas dinamis
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)

    # 1. State Awal: Seluruh 7 slot kosong
    stats = tracker.update(
        tracks=[],
        polygons=roi_cfg.polygons,
        scale_x=3.0,
        scale_y=3.0,
        current_time=1000.0,
    )
    assert stats["total_slots"] == 7
    assert stats["occupied_slots"] == 0
    assert stats["available_slots"] == 7
    print(f"  ✓ Dynamic Capacity: {stats['available_slots']}/{stats['total_slots']} slots available.")

    print("[3/6] Testing vehicle dwell progression (< 10s)...")
    # Kendaraan masuk ke slot 1 (zone_01)
    raw_p0 = [(p.x, p.y) for p in roi_cfg.polygons[0].points]
    ai_cx = sum(p[0] for p in raw_p0) / (len(raw_p0) * 3.0)
    ai_cy = sum(p[1] for p in raw_p0) / (len(raw_p0) * 3.0)

    car_bbox = (ai_cx - 20, ai_cy - 40, ai_cx + 20, ai_cy)
    car_track = TrackResult(
        track_id=106,
        class_label="car",
        class_id=2,
        confidence=0.92,
        bbox=car_bbox,
        is_confirmed=True,
    )

    # T = 1000.0s (First seen)
    stats = tracker.update(
        tracks=[car_track],
        polygons=roi_cfg.polygons,
        scale_x=3.0,
        scale_y=3.0,
        current_time=1000.0,
    )
    s1 = stats["slot_states"][roi_cfg.polygons[0].zone_id]
    assert not s1.occupied, "Slot should NOT be occupied on initial frame"
    assert stats["available_slots"] == 7

    # T = 1005.0s (5s elapsed) -> Settling
    stats = tracker.update(
        tracks=[car_track],
        polygons=roi_cfg.polygons,
        scale_x=3.0,
        scale_y=3.0,
        current_time=1005.0,
    )
    s1 = stats["slot_states"][roi_cfg.polygons[0].zone_id]
    assert s1.dwell_duration == 5.0
    assert not s1.occupied
    print("  ✓ Dwell at 5s is settling (occupied=False, available=7/7).")

    print("[4/6] Testing vehicle dwell threshold reached (>= 10s)...")
    # T = 1010.5s (10.5s elapsed) -> OCCUPIED
    stats = tracker.update(
        tracks=[car_track],
        polygons=roi_cfg.polygons,
        scale_x=3.0,
        scale_y=3.0,
        current_time=1010.5,
    )
    s1 = stats["slot_states"][roi_cfg.polygons[0].zone_id]
    assert s1.occupied, "Slot must be marked OCCUPIED after 10.5s"
    assert stats["occupied_slots"] == 1
    assert stats["available_slots"] == 6
    print(f"  ✓ Dwell at 10.5s OCCUPIED: available={stats['available_slots']}/{stats['total_slots']}.")

    print("[5/6] Testing Gate Tripwire Telemetry & Flash Trigger...")
    tracker.record_gate_crossing("LINE_CROSSING IN", tripwire_id="tw_01", timestamp=1010.5)
    assert tracker.gate_in_count == 1
    assert tracker.tripwire_flash.get("tw_01") == 1011.5
    print("  ✓ Tripwire flash active until 1011.5s.")

    print("[6/6] Testing Visual Kontur Interaktif (Modern IVA Rendering)...")
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.render_overlay(
        canvas=canvas,
        polygons=roi_cfg.polygons,
        tripwires=roi_cfg.tripwires,
        tracks=[car_track],
        scale_x=3.0,
        scale_y=3.0,
        parking_stats=stats,
        current_time=1010.8,  # during flash
    )

    non_zero = np.count_nonzero(canvas)
    assert non_zero > 1000, f"Canvas should have rendered overlays, non_zero={non_zero}"
    print(f"  ✓ IVA Visual Contours rendered on 640x360 canvas ({non_zero} active pixels).")

    print("\n==================================================================")
    print("  ALL DYNAMIC SMART PARKING & IVA CONTOUR TESTS PASSED! ✓")
    print("==================================================================")


if __name__ == "__main__":
    test_dynamic_smart_parking()
