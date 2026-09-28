"""
tools/test_rca_ground_contact_s1_s4.py
======================================
Unit & Integration Test spesifik untuk memvalidasi:
1. Rejection False Positive S1 (Bayangan/Glare dengan confidence rendah < 0.32).
2. Rejection Crosstalk S4 vs S5 (Bodi atas mobil S5 merambah ke poligon S4, roda di S5).
3. Warmup Transient Glitch Recovery (Deteksi 1-frame saat warmup tidak mengunci latch).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from engine.config_loader import load_roi_zones, ROIPoint, ROIZone
from engine.tracker_interface import TrackResult
from engine.smart_parking import SmartParkingTracker, SlotState


def test_s1_low_confidence_shadow_rejection():
    print("\n[TEST 1] Rejection False Positive S1 (Bayangan Rendah < 0.32)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0, acquisition_conf_thresh=0.32)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    s1_poly = roi_cfg.polygons[0]  # zone_01

    raw_p0 = [(p.x, p.y) for p in s1_poly.points]
    ai_cx = sum(p[0] for p in raw_p0) / (len(raw_p0) * 3.0)
    ai_cy = sum(p[1] for p in raw_p0) / (len(raw_p0) * 3.0)

    # Bayangan kanopi terdeteksi sebagai 'car' dengan conf=0.25 (dibawah 0.32)
    shadow_track = TrackResult(
        track_id=999,
        class_label="car",
        class_id=2,
        confidence=0.25,
        bbox=(ai_cx - 20, ai_cy - 20, ai_cx + 20, ai_cy + 20),
        is_confirmed=True,
    )

    stats = tracker.update(
        tracks=[shadow_track],
        polygons=roi_cfg.polygons,
        scale_x=3.0,
        scale_y=3.0,
        current_time=100.0,
        is_warmup=False,
    )

    assert not tracker.slot_states["zone_01"].occupied, "Slot S1 harus tetap VACANT terhadap bayangan conf 0.25!"
    assert stats["available_slots"] == 8, f"Expected 8 available slots, got {stats['available_slots']}"
    print("  ✓ PASS: False detection bayangan S1 (conf=0.25) berhasil di-reject dari VACANT slot.")


def test_s4_s5_perspective_crosstalk_rejection():
    print("\n[TEST 2] Rejection Crosstalk Perspektif (Mobil di S5 tidak boleh mengokupansi S4)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))

    s4 = next(p for p in roi_cfg.polygons if p.zone_id == "zone_04")
    s5 = next(p for p in roi_cfg.polygons if p.zone_id == "zone_05")

    # Ambil tapak aspal S5
    s5_pts = np.array([[p.x / 3.0, p.y / 3.0] for p in s5.points])
    s5_cx = float(np.mean(s5_pts[:, 0]))
    s5_max_y = float(np.max(s5_pts[:, 1]))

    # Mobil tinggi (misal SUV/minibus) terparkir di S5:
    # Rodanya (bottom_center) berada di S5 (s5_cx, s5_max_y - 10)
    # Bbox melebar ke kiri (x1=280 masuk rentang S4), tapi roda di x=350 (murni S5)
    tall_car_s5 = TrackResult(
        track_id=501,
        class_label="car",
        class_id=2,
        confidence=0.88,
        bbox=(280.0, 130.0, 410.0, s5_max_y - 5.0),
        is_confirmed=True,
    )

    # Update frame awal T=100s
    tracker.update([tall_car_s5], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=100.0)
    # Update frame T=111s (dwell 11s)
    stats = tracker.update([tall_car_s5], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=111.0)

    assert tracker.slot_states["zone_05"].occupied, "Slot S5 harus OCCUPIED oleh mobil di S5!"
    assert not tracker.slot_states["zone_04"].occupied, "Slot S4 harus tetap VACANT (tidak boleh tertrigger oleh bodi S5)!"
    assert stats["occupied_slots"] == 1
    assert stats["available_slots"] == 7
    print("  ✓ PASS: Crosstalk S5 -> S4 berhasil dicegah! S5 terisi, S4 tetap kosong.")


def test_warmup_transient_glitch_recovery():
    print("\n[TEST 3] Warmup Transient Glitch Recovery (1-frame false hit tidak mengunci status)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))

    s1_poly = roi_cfg.polygons[0]
    raw_p0 = [(p.x, p.y) for p in s1_poly.points]
    ai_cx = sum(p[0] for p in raw_p0) / (len(raw_p0) * 3.0)
    ai_cy = sum(p[1] for p in raw_p0) / (len(raw_p0) * 3.0)

    glitch_track = TrackResult(
        track_id=1,
        class_label="car",
        class_id=2,
        confidence=0.80,
        bbox=(ai_cx - 20, ai_cy - 20, ai_cx + 20, ai_cy + 20),
        is_confirmed=False,
    )

    # Frame 1: Warmup ada glitch di S1
    tracker.update([glitch_track], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=10.0, is_warmup=True)
    assert not tracker.slot_states["zone_01"].latch_occupied, "Slot S1 TIDAK boleh langsung latched pada frame 1 warmup!"

    # Frame 2: Warmup berikutnya glitch hilang (aslinya kosong)
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=10.1, is_warmup=True)
    assert tracker.slot_states["zone_01"].phase == "VACANT", "Slot S1 harus langsung kembali ke VACANT saat frame 2 kosong!"
    print("  [PASS] Transient glitch saat warmup langsung pulih ke VACANT tanpa terkunci latch.")


def test_s1_truck_shadow_rejection():
    print("\n[TEST 4] Rejection Bayangan Kanopi Truk/Person pada S1 (Strict 'car' Filtering)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    s1_poly = roi_cfg.polygons[0]

    raw_p0 = [(p.x, p.y) for p in s1_poly.points]
    ai_cx = sum(p[0] for p in raw_p0) / (len(raw_p0) * 3.0)
    ai_cy = sum(p[1] for p in raw_p0) / (len(raw_p0) * 3.0)

    # Deteksi bayangan atap kanopi yang terbaca sebagai 'truck' dengan confidence tinggi
    truck_shadow = TrackResult(
        track_id=909,
        class_label="truck",
        class_id=7,
        confidence=0.78,
        bbox=(ai_cx - 25, ai_cy - 25, ai_cx + 25, ai_cy + 25),
        is_confirmed=True,
    )

    tracker.update([truck_shadow], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=100.0)
    stats = tracker.update([truck_shadow], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=112.0)

    assert not tracker.slot_states["zone_01"].occupied, "Slot S1 WAJIB menolak objek non-'car' (truck)!"
    assert stats["available_slots"] == 8
    print("  [PASS] Deteksi bayangan kanopi 'truck' berhasil diabaikan secara ketat pada S1.")


def test_s8_adjacent_grey_car_anti_crosstalk():
    print("\n[TEST 5] Rejection Crosstalk Mobil Abu-Abu Samping S8...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0, wheel_contact_margin_px=0.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))

    # Kasus 5a: Mobil abu-abu terparkir di sebelah kanan S8 (1080p x: 1770..1890, y: 500..645)
    # Pada AI canvas 640x360 (/ 3.0): x1=590.0, x2=630.0, y1=166.0, y2=215.0
    # Titik tengah roda: cx=610.0 (1080p x=1830), di luar poligon S8 (x=1750)
    grey_car = TrackResult(
        track_id=202,
        class_label="car",
        class_id=2,
        confidence=0.88,
        bbox=(590.0, 166.0, 630.0, 215.0),
        is_confirmed=True,
    )

    tracker.update([grey_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=100.0)
    stats = tracker.update([grey_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=112.0)

    assert not tracker.slot_states["zone_08"].occupied, "Slot S8 WAJIB tetap VACANT (tidak terimbas mobil abu-abu di samping)!"
    assert stats["available_slots"] == 8
    print("  [PASS] Mobil abu-abu di sebelah kanan berhasil dicegah mengokupansi S8.")

    # Kasus 5b: Mobil sah yang parkir tepat di dalam S8 (center cx=542.5, y2=195.0 di dalam poligon S8)
    legit_s8_car = TrackResult(
        track_id=808,
        class_label="car",
        class_id=2,
        confidence=0.90,
        bbox=(522.5, 145.0, 562.5, 195.0),
        is_confirmed=True,
    )
    tracker.update([legit_s8_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=120.0)
    stats2 = tracker.update([legit_s8_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=132.0)
    assert tracker.slot_states["zone_08"].occupied, "Mobil sah yang parkir di dalam S8 harus terbaca OCCUPIED!"
    assert stats2["occupied_slots"] == 1
    print("  [PASS] Mobil sah yang parkir di dalam batas baru S8 terdeteksi OCCUPIED dengan sempurna.")


def test_warmup_latch_auto_reset():
    print("\n[TEST 6] Auto-Reset Warmup Latch yang Hilang (Poligon Bersih >= 4.0s Pasca-Warmup)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))

    s1_poly = roi_cfg.polygons[0]
    raw_p0 = [(p.x, p.y) for p in s1_poly.points]
    ai_cx = sum(p[0] for p in raw_p0) / (len(raw_p0) * 3.0)
    ai_cy = sum(p[1] for p in raw_p0) / (len(raw_p0) * 3.0)

    car_track = TrackResult(
        track_id=77,
        class_label="car",
        class_id=2,
        confidence=0.85,
        bbox=(ai_cx - 20, ai_cy - 20, ai_cx + 20, ai_cy + 20),
        is_confirmed=True,
    )

    # 1. Warmup selama 12 frame mengonfirmasi mobil di S1 -> latch_occupied = True
    for f in range(12):
        tracker.update([car_track], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=1.0 + f * 0.1, is_warmup=True)
    assert tracker.slot_states["zone_01"].latch_occupied, "Slot S1 harus ter-latch setelah >= 10 warmup hits!"

    # 2. Pasca-warmup, mobil/glitch langsung hilang (poligon bersih)
    # T = 3.0s (bersih selama 0.7s)
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=3.0, is_warmup=False)
    assert tracker.slot_states["zone_01"].latch_occupied, "Belum 4 detik bersih, latch masih aktif."

    # 3. T = 7.5s (bersih selama 5.2s > 4.0s) -> Auto-reset harus memicu pembebasan slot
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=7.5, is_warmup=False)
    assert not tracker.slot_states["zone_01"].latch_occupied, "Latch S1 harus di-reset setelah >= 4.0s bersih!"
    assert tracker.slot_states["zone_01"].phase == "VACANT", "Slot S1 harus kembali ke VACANT setelah auto-reset!"
    assert tracker.slot_states["zone_01"].dwell_duration == 0.0
    print("  [PASS] Slot ter-latch otomatis pulih ke VACANT setelah poligon bersih >= 4.0 detik pasca-warmup.")


if __name__ == "__main__":
    test_s1_low_confidence_shadow_rejection()
    test_s4_s5_perspective_crosstalk_rejection()
    test_warmup_transient_glitch_recovery()
    test_s1_truck_shadow_rejection()
    test_s8_adjacent_grey_car_anti_crosstalk()
    test_warmup_latch_auto_reset()
    print("\n==================================================================")
    print("  ALL 6 ROOT CAUSE VALIDATION TESTS PASSED! [OK]")
    print("==================================================================")
