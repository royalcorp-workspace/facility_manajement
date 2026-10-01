"""
tools/test_s1_corridor_retention.py
===================================
Test Suite untuk Validasi Universal Corridor Gating & Sightline Occlusion Freeze (cam_01):
1. Kasus A: Truk Berhenti di Depan S1 (Severe Occlusion pada Mobil Silver)
   - Mobil silver stabil OCCUPIED (dwell >= 10s).
   - Truk boks besar (y2=280 >= 225 px) berhenti di koridor depan S1 selama 15 detik.
   - Deteksi mobil silver drop 100% (tertutup total oleh bodi truk).
   - S1 WAJIB TETAP OCCUPIED (Occlusion Freeze aktif, tidak drop/flapping ke VACANT).
2. Kasus B: Moncong Truk Menyenggol Slot S2 (Anti-Claim Encroachment)
   - Slot S2 kosong (VACANT).
   - Truk koridor (y2=260 >= 225 px) melintas/berhenti dengan moncong atas menembus poligon S2.
   - S2 WAJIB TETAP VACANT (Universal corridor gating menolak klaim bodi melayang).
3. Kasus C: Truk Pergi & Pelepasan Sah Pasca-Truk Pergi (Departure Verification)
   - Truk meninggalkan koridor depan -> koridor bersih tuntas.
   - Mobil silver keluar nyata dari S1.
   - Setelah aspal bersih kontinu >= 10.0s pasca-latch, S1 transisi normal LEAVING -> VACANT.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from engine.config_loader import load_roi_zones
from engine.tracker_interface import TrackResult
from engine.smart_parking import SmartParkingTracker


def test_case_a_s1_severe_occlusion_by_corridor_truck():
    """
    Kasus A: Truk boks besar berhenti di koridor depan S1 menutupi mobil silver yang parkir sah.
    Meskipun deteksi mobil silver drop 100% selama 15 detik, status S1 WAJIB DIBEKUKAN (OCCUPIED).
    """
    print("\n[KASUS A] Truk Berhenti di Depan S1 (Severe Foreground Occlusion)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    s1_poly = roi_cfg.polygons[0]  # zone_01

    raw_p0 = [(p.x, p.y) for p in s1_poly.points]
    s1_cx = sum(p[0] for p in raw_p0) / (len(raw_p0) * 3.0)
    s1_cy = sum(p[1] for p in raw_p0) / (len(raw_p0) * 3.0)

    # 1. Mobil silver parkir sah di S1 (y2=190 <= 225 px)
    silver_car = TrackResult(
        track_id=101,
        class_label="car",
        class_id=2,
        confidence=0.88,
        bbox=(s1_cx - 25, s1_cy - 30, s1_cx + 25, 190.0),
        is_confirmed=True,
    )

    # Frame 1: Masuk
    tracker.update([silver_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=100.0)
    # Dwell hingga stabil OCCUPIED (dwell >= 10s)
    stats1 = tracker.update([silver_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=111.0)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "OCCUPIED", f"S1 harusnya OCCUPIED, dapat {s1.phase}"
    assert s1.occupied is True
    assert s1.dwell_duration >= 10.0
    print("  ✓ Mobil silver stabil OCCUPIED di S1 (dwell >= 10s).")

    # 2. Truk boks besar berhenti di koridor depan S1:
    # Tapak roda di koridor sirkulasi (y2 = 280 px >= 225 px pada 640p canvas)
    # Bodi atas menutupi S1 (y1 = 140 <= s1_ymax + 15), x span: [s1_cx - 40, s1_cx + 40]
    corridor_truck = TrackResult(
        track_id=202,
        class_label="truck",
        class_id=7,
        confidence=0.94,
        bbox=(s1_cx - 40.0, 140.0, s1_cx + 40.0, 280.0),
        is_confirmed=True,
    )

    # Simulasi: Truk berhenti di depan S1 selama 15 detik (t = 112s s.d. 127s).
    # Mobil silver drop 100% (tidak terdeteksi detektor karena tertutup fisik oleh bodi truk).
    t = 111.0
    for step in range(30):  # 30 step * 0.5s = 15 detik oklusi total
        t += 0.5
        stats_occ = tracker.update([corridor_truck], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=t)
        assert s1.phase == "OCCUPIED", f"Pada t={t:.1f}s, S1 lepas ke {s1.phase} padahal tertutup truk koridor!"
        assert s1.occupied is True, "S1 harus tetap terhitung occupied!"
        assert stats_occ["occupied_slots"] == 1, f"Kuota okupansi harus tetap 1, didapat {stats_occ['occupied_slots']}"

    print("  ✓ PASS: Oklusi 15 detik tertahan sempurna! S1 dibekukan (OCCUPIED) via Occlusion Freeze State.")


def test_case_b_s2_anti_claim_corridor_encroachment():
    """
    Kasus B: Slot S2 kosong (VACANT). Truk koridor melintas dengan moncong atas menyenggol S2.
    Universal corridor gating (y2 >= 225 px) dan validasi kontak tapak tanah WAJIB menolak klaim S2.
    Slot S2 harus TETAP VACANT (0/8 kuota terisi).
    """
    print("\n[KASUS B] Moncong Truk Menyenggol Slot S2 (Anti-Claim Encroachment)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))

    s2_poly = next(p for p in roi_cfg.polygons if p.zone_id == "zone_02")
    s2_pts = np.array([[p.x / 3.0, p.y / 3.0] for p in s2_poly.points])
    s2_cx = float(np.mean(s2_pts[:, 0]))
    s2_cy = float(np.mean(s2_pts[:, 1]))
    s2_max_y = float(np.max(s2_pts[:, 1]))

    # Inisialisasi awal
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=199.0)
    s2_state = tracker.slot_states["zone_02"]
    assert s2_state.phase == "VACANT", "S2 harus berawal dari VACANT"

    # Truk koridor berada di depan S2:
    # Tapak roda y2 = 265 px (di koridor sirkulasi >= 225 px)
    # Moncong atas y1 = 150 px menembus poligon S2 (overlap bodi atas di aspal S2)
    corridor_truck_s2 = TrackResult(
        track_id=303,
        class_label="truck",
        class_id=7,
        confidence=0.88,
        bbox=(s2_cx - 25.0, 150.0, s2_cx + 35.0, 265.0),
        is_confirmed=True,
    )

    # Truk diam/bermanuver di koridor depan S2 selama 12 detik
    t = 200.0
    for step in range(24):
        t += 0.5
        stats = tracker.update([corridor_truck_s2], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=t)
        assert s2_state.phase == "VACANT", f"Pada t={t:.1f}s, S2 salah terisi menjadi {s2_state.phase} oleh truk koridor!"
        assert not s2_state.occupied, "S2 dilarang keras occupied oleh objek koridor!"
        assert stats["occupied_slots"] == 0, f"Kuota okupansi harus 0, didapat: {stats['occupied_slots']}"

    print("  ✓ PASS: Moncong truk koridor (y2 >= 225) berhasil di-reject mutlak dari S2! S2 tetap VACANT.")


def test_case_c_departure_verification_post_corridor_clear():
    """
    Kasus C: Truk pergi secara tuntas dari koridor depan S1.
    Setelah koridor bersih dan mobil silver benar-benar keluar, S1 transisi sah ke LEAVING -> VACANT.
    """
    print("\n[KASUS C] Pelepasan Sah Pasca-Truk Pergi (Departure Verification)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    s1_poly = roi_cfg.polygons[0]

    raw_p0 = [(p.x, p.y) for p in s1_poly.points]
    s1_cx = sum(p[0] for p in raw_p0) / (len(raw_p0) * 3.0)
    s1_cy = sum(p[1] for p in raw_p0) / (len(raw_p0) * 3.0)

    silver_car = TrackResult(
        track_id=101,
        class_label="car",
        class_id=2,
        confidence=0.88,
        bbox=(s1_cx - 25, s1_cy - 30, s1_cx + 25, 190.0),
        is_confirmed=True,
    )
    corridor_truck = TrackResult(
        track_id=202,
        class_label="truck",
        class_id=7,
        confidence=0.94,
        bbox=(s1_cx - 40.0, 140.0, s1_cx + 40.0, 280.0),
        is_confirmed=True,
    )

    # 1. Mobil stabil OCCUPIED di S1
    tracker.update([silver_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=300.0)
    tracker.update([silver_car], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=312.0)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "OCCUPIED"

    # 2. Truk berhenti di depan S1 (mobil silver tertutup selama 6 detik)
    tracker.update([corridor_truck], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=315.0)
    tracker.update([corridor_truck], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=318.0)
    assert s1.phase == "OCCUPIED", "S1 harus tetap OCCUPIED selama truk ada di koridor"

    # 3. Truk PERGI secara tuntas (koridor bersih) dan mobil silver juga keluar nyata (frame kosong)
    t_exit = 320.0
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=t_exit)
    assert s1.phase == "OCCUPIED", "Detik ke-0 pasca koridor bersih: S1 masih harus OCCUPIED"

    # Pada 4.0 detik bersih (< 5.0s): masih OCCUPIED
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=t_exit + 4.0)
    assert s1.phase == "OCCUPIED", "4.0 detik pasca bersih: harus tetap OCCUPIED (debounce safe release)"

    # Pada 10.5 detik bersih (melampaui 10s confirm_threshold): transisi ke LEAVING
    tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=t_exit + 10.5)
    assert s1.phase == "LEAVING", f"Setelah >= 10s bersih kontinu harusnya LEAVING, dapat {s1.phase}"

    # Pasca exit grace period: transisi resmi ke VACANT
    stats_vacant = tracker.update([], roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=t_exit + 15.0)
    assert s1.phase == "VACANT", f"Setelah exit grace harusnya VACANT, dapat {s1.phase}"
    assert not s1.occupied
    assert stats_vacant["occupied_slots"] == 0
    print("  ✓ PASS: Truk pergi -> mobil keluar nyata -> verifikasi aspal bersih -> transisi sukses ke VACANT.")


def main():
    print("==================================================================")
    print("  CAM_01 CORRIDOR OCCLUSION & UNIVERSAL GATING TEST SUITE")
    print("==================================================================")

    test_case_a_s1_severe_occlusion_by_corridor_truck()
    test_case_b_s2_anti_claim_corridor_encroachment()
    test_case_c_departure_verification_post_corridor_clear()

    print("\n==================================================================")
    print("  ALL 3 CORRIDOR OCCLUSION & GATING TESTS PASSED! (100% SUCCESS) ✓")
    print("==================================================================")


if __name__ == "__main__":
    main()
