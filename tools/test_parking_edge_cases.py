"""
tools/test_parking_edge_cases.py
=================================
Edge Cases Verification Suite untuk cam_01 (Area Parkir Mobil S1-S8):
1. Double Encroachment Test: Mobil serong menyentuh 2 slot -> hanya 1 slot terisi, slot tetangga tetap VACANT.
2. Transient Occlusion Test: Dropout deteksi 3.5 detik -> slot tidak berkedip menjadi VACANT/LEAVING (Sticky Latch Guard).
3. Maneuvering Corridor Obstruction: Mobil stasioner di luar petak slot (shift <= 15px, dwell >= 60s) -> OBSTRUCTION_ALERT terpicu.
4. Bumper Protrusion & Multi-Point Stance: Bumper keluar poligon namun ban/as roda di dalam -> slot terdeteksi akurat.
5. Non-Regression Integration Runner: Memastikan seluruh baseline test tetap 100% lulus.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np

from engine.config_loader import load_roi_zones, ROIPoint, ROIZone
from engine.tracker_interface import TrackResult
from engine.smart_parking import SmartParkingTracker


def test_case_1_double_encroachment():
    """
    Test Case 1: Double / Triple Slot Encroachment
    Mobil parkir menyerong menyentuh batas zona S5 dan S6.
    Sistem harus menugaskan mobil ke slot dengan afinitas/tapak terbesar (S6),
    sementara slot tetangga (S5) WAJIB tetap VACANT. Kuota terisi tepat 1.
    """
    print("\n[TEST 1] Double Encroachment (Mobil Serong di Perbatasan S5 & S6)...")
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)

    s5 = next(p for p in roi_cfg.polygons if p.zone_id == "zone_05")
    s6 = next(p for p in roi_cfg.polygons if p.zone_id == "zone_06")

    pts5 = np.array([[p.x / 3.0, p.y / 3.0] for p in s5.points])
    pts6 = np.array([[p.x / 3.0, p.y / 3.0] for p in s6.points])

    s6_cx = float(np.mean(pts6[:, 0]))
    s6_max_y = float(np.max(pts6[:, 1]))

    # Mobil serong dengan bodi dominan di S6, moncong/sudut kiri menyentuh batas kanan S5:
    # Bbox: x1 di batas S5/S6 (misal 395), x2 di 465 (dalam S6), y2 menyentuh tapak roda bawah
    encroaching_car = TrackResult(
        track_id=506,
        class_label="car",
        class_id=2,
        confidence=0.89,
        bbox=(395.0, 140.0, 465.0, s6_max_y - 4.0),
        is_confirmed=True,
    )

    # Frame 1: Masuk
    tracker.update([encroaching_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=100.0)
    # Frame 2: Dwell melewati 10s
    stats = tracker.update([encroaching_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=111.0)

    s5_state = tracker.slot_states["zone_05"]
    s6_state = tracker.slot_states["zone_06"]

    assert s6_state.occupied, f"Slot S6 harus OCCUPIED oleh mobil dominan, dapat: {s6_state.phase}"
    assert s6_state.track_id == 506, f"Slot S6 harus terikat track_id 506, dapat: {s6_state.track_id}"
    assert not s5_state.occupied, f"Slot S5 WAJIB tetap VACANT (tidak boleh double claim!), dapat: {s5_state.phase}"
    assert s5_state.track_id is None, f"Slot S5 track_id harus None, dapat: {s5_state.track_id}"

    assert stats["occupied_slots"] == 1, f"Total occupied harus tepat 1, dapat: {stats['occupied_slots']}"
    assert stats["available_slots"] == 7, f"Total available harus 7, dapat: {stats['available_slots']}"
    print("  ✓ PASS: Double Encroachment tertangani sempurna! S6 terisi, S5 tetap VACANT, total occupied = 1.")


def test_case_2_transient_occlusion():
    """
    Test Case 2: Transient Occlusion Dropout (3.5 Detik)
    Mobil di S3 telah stabil OCCUPIED (ter-latch).
    Simulasikan deteksi YOLO hilang selama 3.5 detik (orang lewat / truk melintas).
    Slot S3 TIDAK BOLEH berkedip ke LEAVING atau VACANT.
    """
    print("\n[TEST 2] Transient Occlusion (Dropout Deteksi 3.5 Detik pada S3)...")
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)

    s3 = next(p for p in roi_cfg.polygons if p.zone_id == "zone_03")
    pts3 = np.array([[p.x / 3.0, p.y / 3.0] for p in s3.points])
    s3_cx = float(np.mean(pts3[:, 0]))
    s3_cy = float(np.mean(pts3[:, 1]))
    s3_max_y = float(np.max(pts3[:, 1]))

    car3 = TrackResult(
        track_id=303,
        class_label="car",
        class_id=2,
        confidence=0.92,
        bbox=(s3_cx - 25.0, s3_cy - 20.0, s3_cx + 25.0, s3_max_y - 5.0),
        is_confirmed=True,
    )

    # 1. Mobil parkir stabil selama 12 detik
    tracker.update([car3], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=200.0)
    tracker.update([car3], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=212.0)

    s3_state = tracker.slot_states["zone_03"]
    assert s3_state.occupied, "S3 harus OCCUPIED sebelum oklusi"
    assert s3_state.latch_occupied, "S3 harus latch_occupied setelah dwell >= 5.0s"

    # 2. Simulasi oklusi: dropout 3.5 detik (frame tanpa deteksi setiap 0.5s)
    t = 212.0
    for step in range(7):
        t += 0.5
        stats_occ = tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=t)
        assert s3_state.occupied, f"Pada t={t:.1f}s (oklusi {t - 212.0:.1f}s), S3 flapping ke {s3_state.phase}!"
        assert stats_occ["occupied_slots"] == 1, "Kapasitas terisi tidak boleh turun selama oklusi 3.5s!"

    # 3. Kendaraan terlihat kembali setelah oklusi
    stats_recovered = tracker.update([car3], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=216.0)
    assert s3_state.occupied
    assert s3_state.polygon_clear_since is None, "polygon_clear_since harus reset setelah terlihat kembali"
    print("  ✓ PASS: Oklusi 3.5s berhasil ditoleransi tanpa flapping! S3 stabil OCCUPIED.")


def test_case_3_corridor_obstruction():
    """
    Test Case 3: Maneuvering Corridor Obstruction
    Mobil berhenti di koridor aspal manuver (di luar seluruh poligon S1-S8).
    - Shift <= 15px secara kontinu selama >= 60.0 detik.
    - Sistem wajib memicu status OBSTRUCTION_ALERT = True dan mendaftarkan objek halangan.
    - Slot S1-S8 tidak boleh terimbas.
    - Jika mobil bergerak (shift > 15px), alarm tidak terpicu atau di-reset.
    """
    print("\n[TEST 3] Maneuvering Corridor Obstruction Alert...")
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    tracker = SmartParkingTracker(
        dwell_threshold_sec=10.0,
        corridor_obstruction_dwell_sec=60.0,
        corridor_shift_threshold_px=15.0,
    )

    # Titik aspal koridor di depan petak parkir (misal y=260-310 pada 640x360, jauh di bawah S1-S8)
    # Bbox: x1=200, y1=260, x2=280, y2=320 (aspal manuver)
    obstruction_car = TrackResult(
        track_id=888,
        class_label="car",
        class_id=2,
        confidence=0.91,
        bbox=(200.0, 260.0, 280.0, 320.0),
        is_confirmed=True,
    )

    # 1. Mobil berhenti di koridor pada T = 500s
    tracker.update([obstruction_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=500.0)
    assert not tracker.has_obstruction, "Belum 60s, belum boleh ada obstruction alert"

    # 2. Mobil tetap stasioner selama 30 detik (shift 2px) -> belum alert
    stationary_mid = TrackResult(
        track_id=888,
        class_label="car",
        class_id=2,
        confidence=0.91,
        bbox=(201.0, 261.0, 281.0, 321.0),
        is_confirmed=True,
    )
    stats_30s = tracker.update([stationary_mid], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=530.0)
    assert not stats_30s["obstruction_alert"], "Pada 30s dwell belum boleh memicu obstruction alert"

    # 3. Mobil tetap stasioner mencapai 61 detik (T = 561s, shift 3px) -> OBSTRUCTION_ALERT HARUS AKTIF
    stationary_61s = TrackResult(
        track_id=888,
        class_label="car",
        class_id=2,
        confidence=0.91,
        bbox=(202.0, 260.0, 282.0, 320.0),
        is_confirmed=True,
    )
    stats_61s = tracker.update([stationary_61s], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=561.0)
    assert stats_61s["obstruction_alert"] is True, "OBSTRUCTION_ALERT harus aktif setelah stasioner >= 60s di koridor!"
    assert len(stats_61s["obstructions"]) == 1, "Harus ada 1 entri halangan aktif"
    obs_info = stats_61s["obstructions"][0]
    assert obs_info["track_id"] == 888
    assert obs_info["dwell_sec"] >= 60.0

    # Pastikan seluruh slot S1-S8 tetap VACANT
    for s_id, state in tracker.slot_states.items():
        assert state.phase == "VACANT", f"{s_id} tidak boleh mengklaim mobil koridor"
    assert stats_61s["occupied_slots"] == 0
    assert stats_61s["available_slots"] == 8

    # 4. Uji Reset jika Kendaraan Bergerak Melaju (> 15px shift)
    moving_car = TrackResult(
        track_id=888,
        class_label="car",
        class_id=2,
        confidence=0.91,
        bbox=(230.0, 260.0, 310.0, 320.0),  # shift x 28px > 15px
        is_confirmed=True,
    )
    stats_moving = tracker.update([moving_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=562.0)
    assert not stats_moving["obstruction_alert"], "Halangan harus reset saat kendaraan bergerak melaju!"
    print("  ✓ PASS: Maneuvering Corridor Obstruction Engine berhasil mendeteksi halangan & auto-reset saat bergerak.")


def test_case_4_bumper_protrusion_stance():
    """
    Test Case 4: Mobil Terparkir Terlalu Maju (Bumper Protrusion)
    Bumper depan sedikit keluar garis poligon slot (d_center < 0), namun
    titik kontak ban/as roda (left/right tire/inset) masih di dalam slot.
    Dengan 5-point stance probe, mobil tetap berhasil diakuisisi ke dalam slot.
    """
    print("\n[TEST 4] Bumper Protrusion & Multi-Point Stance Contact...")
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)

    s2 = next(p for p in roi_cfg.polygons if p.zone_id == "zone_02")
    pts2 = np.array([[p.x / 3.0, p.y / 3.0] for p in s2.points])
    s2_cx = float(np.mean(pts2[:, 0]))
    s2_max_y = float(np.max(pts2[:, 1]))

    # Mobil parkir agak maju:
    # y2 melampaui batas bawah poligon sebesar 2px (sehingga bottom-center d_wheel < 0),
    # namun inset (y2 - 0.05*h) dan as roda masih mantap di dalam poligon
    protruding_car = TrackResult(
        track_id=202,
        class_label="car",
        class_id=2,
        confidence=0.88,
        bbox=(s2_cx - 20.0, s2_max_y - 45.0, s2_cx + 20.0, s2_max_y + 1.0),
        is_confirmed=True,
    )

    tracker.update([protruding_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=300.0)
    stats = tracker.update([protruding_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=311.0)

    s2_state = tracker.slot_states["zone_02"]
    assert s2_state.occupied, f"Slot S2 harus OCCUPIED meskipun bumper menjulur 1px, dapat: {s2_state.phase}"
    assert stats["occupied_slots"] == 1
    print("  ✓ PASS: Multi-point stance contact berhasil mengakomodasi bumper protrusion tanpa false rejection.")


def test_case_5_non_regression_checks():
    """
    Test Case 5: Non-Regression Check
    Memastikan seluruh test suite yang ada tetap 100% lulus.
    """
    print("\n[TEST 5] Running Baseline Non-Regression Suite...")
    from tools.test_rca_ground_contact_s1_s4 import (
        test_s1_low_confidence_shadow_rejection,
        test_s4_s5_perspective_crosstalk_rejection,
        test_warmup_transient_glitch_recovery,
        test_s1_truck_shadow_rejection,
        test_s8_adjacent_grey_car_anti_crosstalk,
        test_warmup_latch_auto_reset,
    )

    test_s1_low_confidence_shadow_rejection()
    test_s4_s5_perspective_crosstalk_rejection()
    test_warmup_transient_glitch_recovery()
    test_s1_truck_shadow_rejection()
    test_s8_adjacent_grey_car_anti_crosstalk()
    test_warmup_latch_auto_reset()
    print("  ✓ PASS: Seluruh 6 baseline RCA validation tests 100% PASS.")


if __name__ == "__main__":
    print("==================================================================")
    print("  RESILIENT MULTI-ZONE PARKING EDGE CASES TEST SUITE (cam_01)")
    print("==================================================================")
    test_case_1_double_encroachment()
    test_case_2_transient_occlusion()
    test_case_3_corridor_obstruction()
    test_case_4_bumper_protrusion_stance()
    test_case_5_non_regression_checks()
    print("\n==================================================================")
    print("  ALL 5 RESILIENT PARKING EDGE CASE TESTS PASSED! (100% SUCCESS) ✓")
    print("==================================================================")
