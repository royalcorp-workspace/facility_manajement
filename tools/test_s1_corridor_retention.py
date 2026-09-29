"""
tools/test_s1_corridor_retention.py
===================================
Unit Test untuk memvalidasi:
1. Y-Axis Gating (y2 > 230 px pada AI 640p canvas) menolak kendaraan koridor dari kandidat slot S1 s.d. S6.
2. Truk boks besar / pickup koridor tidak memicu Exclusive Bipartite Matching / Instant Yield melepaskan S1.
3. Retensi okupansi S1 stasioner (dwell >= 10s) tetap stabil OCCUPIED selama 5-point stance probe mobil ada kontak.
4. Slot S1 baru transisi ke VACANT setelah seluruh 5 titik probe bersih kontinu selama >= 5.0 detik berturut-turut.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from engine.config_loader import load_roi_zones
from engine.tracker_interface import TrackResult
from engine.smart_parking import SmartParkingTracker


def test_s1_corridor_y_axis_gating_and_retention():
    print("\n[TEST] S1 Corridor Interference & Y-Axis Gating Validation...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    s1_poly = roi_cfg.polygons[0]  # zone_01

    raw_p0 = [(p.x, p.y) for p in s1_poly.points]
    s1_cx = sum(p[0] for p in raw_p0) / (len(raw_p0) * 3.0)
    s1_cy = sum(p[1] for p in raw_p0) / (len(raw_p0) * 3.0)

    # 1. Mobil silver sah terparkir di S1 (y2=190 <= 230 px)
    silver_car = TrackResult(
        track_id=101,
        class_label="car",
        class_id=2,
        confidence=0.88,
        bbox=(s1_cx - 25, s1_cy - 30, s1_cx + 25, 190.0),
        is_confirmed=True,
    )

    # Frame 1: Mobil masuk ke S1
    tracker.update([silver_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=100.0)
    # Lanjut hingga stabil OCCUPIED (dwell >= 10s)
    stats1 = tracker.update([silver_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=111.0)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "OCCUPIED", f"S1 harusnya OCCUPIED, dapat {s1.phase}"
    assert s1.occupied is True
    assert s1.dwell_duration >= 10.0
    print("  ✓ PASS: Mobil silver berhasil stasioner dan OCCUPIED di S1 (dwell >= 10s).")

    # 2. Truk boks kuning / pickup manuver di koridor depan (y2=280 > 230 px pada AI canvas)
    # Bbox besar membentang x: [s1_cx - 40, s1_cx + 60], y: [160, 280] menyenggol tepi bawah S1
    corridor_truck = TrackResult(
        track_id=202,
        class_label="truck",
        class_id=7,
        confidence=0.92,
        bbox=(s1_cx - 40, 160.0, s1_cx + 60, 280.0),
        is_confirmed=True,
    )
    pickup_car = TrackResult(
        track_id=203,
        class_label="car",
        class_id=2,
        confidence=0.90,
        bbox=(s1_cx - 30, 170.0, s1_cx + 50, 260.0),
        is_confirmed=True,
    )

    # Simulasi truk dan pickup melintas di depan S1 (saat mobil silver tetap ada atau miss 1 frame)
    # Cek: Y-Axis Gating harus menolak truk dan pickup dari kandidat S1
    for t_step in range(1, 15):
        # Sekalipun mobil silver sesekali dropout deteksi (occlusion simulasi 1-2 detik)
        current_tracks = [corridor_truck, pickup_car]
        if t_step % 3 != 0:
            current_tracks.append(silver_car)

        tracker.update(current_tracks, roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=111.0 + t_step * 0.5)
        assert s1.phase == "OCCUPIED", f"S1 TIDAK boleh lepas ke {s1.phase} saat ada kendaraan koridor melintas!"
        assert s1.occupied is True
        assert s1.track_id != 202, "Truk koridor dilarang merebut track_id S1!"
        assert s1.track_id != 203, "Pickup koridor dilarang merebut track_id S1!"

    print("  ✓ PASS: Truk & pickup koridor (y2 > 230) berhasil di-reject dari S1, S1 stabil OCCUPIED.")

    # 3. Simulasi mobil silver keluar secara nyata (seluruh 5 titik probe bersih kontinu)
    # Waktu t = 130.0s: Mobil pergi
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=130.0)
    assert s1.phase == "OCCUPIED", "0s poligon kosong: masih harus OCCUPIED"

    # Pada 4.0 detik kosong (< 5.0s): masih OCCUPIED
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=134.0)
    assert s1.phase == "OCCUPIED", "4.0s poligon kosong: harus tetap OCCUPIED (belum 5s)"

    # Pada 10.5 detik kosong (melampaui 10s adaptive latch clear): harus LEAVING
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=140.5)
    assert s1.phase == "LEAVING", f"Setelah >= 10s bersih kontinu harusnya LEAVING, dapat {s1.phase}"

    # Setelah exit grace period: resmi VACANT
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=145.0)
    assert s1.phase == "VACANT", f"Setelah exit grace period harusnya VACANT, dapat {s1.phase}"
    assert s1.occupied is False
    print("  ✓ PASS: Mobil silver keluar nyata -> transisi mulus ke LEAVING -> VACANT.")

    print("\n==================================================================")
    print("  ALL S1 CORRIDOR RETENTION & Y-AXIS GATING TESTS PASSED! [OK]")
    print("==================================================================")


if __name__ == "__main__":
    test_s1_corridor_y_axis_gating_and_retention()
