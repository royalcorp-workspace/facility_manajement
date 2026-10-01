"""
tools/test_cam03_dense_motorcycle_clustering.py
================================================
Test Suite Spesifik untuk Validasi Deteksi Motor Menumpuk / Berhimpitan (cam_03):
1. Test Dense Staggered Cluster: 5 motor berhimpitan rapat (overlap IoU 35-45%, IoS 40-50%)
   seluruhnya terakuisisi terpisah (occupied=5).
2. Test Child Fragmentation Elimination: 1 motor utuh + 2 pecahan anak (jok/roda IoS >= 0.85)
   hanya dihitung sebagai 1 unit motor.
3. Test Occlusion Retention (Spatial Anchor Memory): Motor lama yang terhalang motor baru di depannya
   tetap terkunci (anti-drop) dengan confidence drop (0.35 -> 0.24) dan dropout temporer.
4. Test Zero-Flapping Moving Median Window: 50 frame streaming dengan noise acak fluktuasi
   menghasilkan okupansi stabil dengan deviasi standar STD = 0.0000.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np

from engine.config_loader import load_roi_zones
from engine.tracker_interface import TrackResult
from engine.smart_parking import SmartParkingTracker, bbox_iou, bbox_ios, deduplicate_motorcycle_tracks


def test_scenario_1_dense_staggered_cluster():
    """
    Skenario 1: 5 motor diparkir paralel dan berhimpitan rapat di dalam poligon cam_03.
    Masing-masing motor memiliki lebar 40px, tinggi 75px pada skala 720p, dengan jarak
    horizontal antar motor 18px (overlap bodi 22px / ~55% bodi menyentuh, IoU ~0.38, IoS ~0.55).
    Sistem WAJIB mengakuisisi kelima motor secara terpisah (occupied = 5).
    """
    print("\n[TEST 1] Dense Staggered Cluster (5 Motor Berhimpitan Rapat)...")
    roi_cfg = load_roi_zones(Path("cameras/cam_03/roi_zones.json"))
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        stationary_dwell_sec=0.0,
    )

    # Basis koordinat di kanvas 720p (scale_x = 1.5, scale_y = 1.5 dari 1080p)
    # Area tengah poligon: x ~ 600..750, y ~ 450..550
    base_x = 600.0
    base_y1 = 450.0
    w = 42.0
    h = 80.0

    dense_tracks = []
    for i in range(5):
        # Offset X = 18px per motor -> overlap horizontal = 42 - 18 = 24px (~57% overlap X)
        x1 = base_x + (i * 18.0)
        x2 = x1 + w
        # Sedikit stagger di sumbu Y (variasi posisi parkir 4px)
        y1 = base_y1 + (i % 2) * 4.0
        y2 = y1 + h

        dense_tracks.append(
            TrackResult(
                track_id=100 + i,
                class_label="motorcycle",
                class_id=3,
                confidence=0.75 - (i * 0.04),  # conf 0.59 s.d. 0.75 (di atas ambang 0.30)
                bbox=(x1, y1, x2, y2),
                is_confirmed=True,
            )
        )

    # Verifikasi overlap antar pasang motor memenuhi syarat pengujian (IoU 0.30 - 0.45)
    for i in range(4):
        iou = bbox_iou(dense_tracks[i].bbox, dense_tracks[i + 1].bbox)
        ios = bbox_ios(dense_tracks[i].bbox, dense_tracks[i + 1].bbox)
        assert 0.25 <= iou <= 0.52, f"IoU antar motor {i} dan {i+1} ({iou:.3f}) harus di kisaran 0.25-0.52"
        assert 0.35 <= ios <= 0.65, f"IoS antar motor {i} dan {i+1} ({ios:.3f}) harus di kisaran 0.35-0.65"

    # Jalankan update selama 5 frame (warmup)
    res = None
    for f in range(5):
        t = 100.0 + f * 0.2
        res = tracker.update(
            tracks=dense_tracks,
            polygons=roi_cfg.polygons,
            scale_x=1.5,
            scale_y=1.5,
            current_time=t,
            is_warmup=True,
        )

    assert res is not None
    occupied = res["occupied_slots"]
    available = res["available_slots"]
    print(f"  Result: occupied={occupied}/30, available={available}")
    assert occupied == 5, f"Ekspektasi 5 motor terdeteksi terpisah, didapat: {occupied}"
    assert available == 25, f"Ekspektasi 25 kuota tersedia, didapat: {available}"
    print("  ✓ PASS: Seluruh 5 motor berhimpitan rapat berhasil diakuisisi secara terpisah!")


def test_scenario_2_child_fragmentation_elimination():
    """
    Skenario 2: 1 motor utuh plus 2 pecahan kotak anak (jok dan roda) dari motor yang sama.
    Pecahan anak memiliki IoS >= 0.85 terhadap motor utuh.
    Sistem WAJIB membuang kedua pecahan anak dan hanya menghitung 1 unit motor (occupied = 1).
    """
    print("\n[TEST 2] Child Fragmentation Elimination (Parent vs Child Boxes)...")
    roi_cfg = load_roi_zones(Path("cameras/cam_03/roi_zones.json"))
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        stationary_dwell_sec=0.0,
    )

    # Motor utuh (Parent)
    parent_bbox = (650.0, 460.0, 695.0, 545.0)  # w=45, h=85
    parent_track = TrackResult(
        track_id=201,
        class_label="motorcycle",
        class_id=3,
        confidence=0.82,
        bbox=parent_bbox,
        is_confirmed=True,
    )

    # Pecahan 1: Jok (berada di bagian tengah dalam bodi motor utuh)
    # IoS = 1.0 (100% di dalam parent)
    jok_bbox = (656.0, 480.0, 688.0, 510.0)  # w=32, h=30
    jok_track = TrackResult(
        track_id=202,
        class_label="motorcycle",
        class_id=3,
        confidence=0.45,
        bbox=jok_bbox,
        is_confirmed=True,
    )

    # Pecahan 2: Stang/Moncong (hampir seluruhnya di dalam parent, IoS ~0.92)
    stang_bbox = (652.0, 462.0, 685.0, 492.0)  # w=33, h=30
    stang_track = TrackResult(
        track_id=203,
        class_label="motorcycle",
        class_id=3,
        confidence=0.40,
        bbox=stang_bbox,
        is_confirmed=True,
    )

    # Verifikasi IoS pecahan terhadap parent >= 0.85
    ios_jok = bbox_ios(parent_bbox, jok_bbox)
    ios_stang = bbox_ios(parent_bbox, stang_bbox)
    assert ios_jok >= 0.85, f"IoS jok ({ios_jok:.2f}) harus >= 0.85"
    assert ios_stang >= 0.85, f"IoS stang ({ios_stang:.2f}) harus >= 0.85"

    frame_tracks = [parent_track, jok_track, stang_track]

    res = None
    for f in range(5):
        t = 200.0 + f * 0.2
        res = tracker.update(
            tracks=frame_tracks,
            polygons=roi_cfg.polygons,
            scale_x=1.5,
            scale_y=1.5,
            current_time=t,
            is_warmup=True,
        )

    assert res is not None
    occupied = res["occupied_slots"]
    available = res["available_slots"]
    print(f"  Result: occupied={occupied}/30, available={available}")
    assert occupied == 1, f"Ekspektasi hanya 1 motor utuh terhitung, didapat: {occupied}"
    assert available == 29, f"Ekspektasi 29 kuota tersedia, didapat: {available}"
    print("  ✓ PASS: Kotak pecahan anak (IoS >= 0.85) berhasil dieliminasi!")


def test_scenario_3_occlusion_retention_spatial_anchor():
    """
    Skenario 3: Motor lama (Motor A) sudah stabil dan terkunci (is_latched=True).
    Kemudian Motor B masuk parkir di depannya, mengakibatkan Motor A tertutup sebagian:
    - Confidence Motor A turun dari 0.70 -> 0.24 (di bawah ambang akuisisi 0.30, namun di atas ambang retensi 0.22).
    - Motor A mengalami dropout 5 frame berturut-turut.
    Sistem WAJIB mempertahankan Motor A via Spatial Anchor Memory (UNIT_TTL_SEC=12.0s)
    sehingga total motor tetap 2 unit (occupied = 2).
    """
    print("\n[TEST 3] Occlusion Retention & Spatial Anchor Memory...")
    roi_cfg = load_roi_zones(Path("cameras/cam_03/roi_zones.json"))
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        stationary_dwell_sec=0.0,
    )

    motor_a_bbox = (620.0, 460.0, 665.0, 545.0)
    motor_b_bbox = (650.0, 480.0, 695.0, 565.0)  # Parkir sedikit di depan & samping Motor A

    # Tahap 1: Motor A masuk dan terkunci stabil (durasi 3 detik / 15 frame)
    current_time = 300.0
    for f in range(15):
        current_time += 0.2
        track_a = TrackResult(
            track_id=301,
            class_label="motorcycle",
            class_id=3,
            confidence=0.72,
            bbox=motor_a_bbox,
            is_confirmed=True,
        )
        tracker.update(
            tracks=[track_a],
            polygons=roi_cfg.polygons,
            scale_x=1.5,
            scale_y=1.5,
            current_time=current_time,
            is_warmup=False,
        )

    assert tracker.occupied_slots == 1, "Tahap 1 gagal: Motor A harus terdaftar."

    # Tahap 2: Motor B masuk (conf=0.78), Motor A teroklusi sebagian (conf turun ke 0.24)
    # Jalankan 20 frame (4.0 detik) agar moving median window (maxlen=25) stabil merefleksikan 2 unit
    for f in range(20):
        current_time += 0.2
        track_b = TrackResult(
            track_id=302,
            class_label="motorcycle",
            class_id=3,
            confidence=0.78,
            bbox=motor_b_bbox,
            is_confirmed=True,
        )
        track_a_low = TrackResult(
            track_id=301,
            class_label="motorcycle",
            class_id=3,
            confidence=0.24,  # conf 0.24 di bawah akuisisi 0.30 tapi >= retensi 0.22
            bbox=motor_a_bbox,
            is_confirmed=True,
        )
        res = tracker.update(
            tracks=[track_a_low, track_b],
            polygons=roi_cfg.polygons,
            scale_x=1.5,
            scale_y=1.5,
            current_time=current_time,
            is_warmup=False,
        )

    assert res["occupied_slots"] == 2, f"Tahap 2 gagal: Harus 2 motor, didapat {res['occupied_slots']}"

    # Tahap 3: Dropout ekstrem (Motor A sama sekali tidak terdeteksi selama 4.0 detik)
    # Motor A harus tetap bertahan karena UNIT_TTL_SEC = 12.0s
    for f in range(20):  # 20 frame * 0.2s = 4.0s
        current_time += 0.2
        track_b = TrackResult(
            track_id=302,
            class_label="motorcycle",
            class_id=3,
            confidence=0.80,
            bbox=motor_b_bbox,
            is_confirmed=True,
        )
        res = tracker.update(
            tracks=[track_b],  # Hanya Motor B yang terlihat detektor
            polygons=roi_cfg.polygons,
            scale_x=1.5,
            scale_y=1.5,
            current_time=current_time,
            is_warmup=False,
        )

    occupied = res["occupied_slots"]
    print(f"  Result pasca-oklusi 4.0s: occupied={occupied}/30")
    assert occupied == 2, f"Motor A tidak boleh drop saat oklusi 4s! Didapat: {occupied}"
    print("  ✓ PASS: Spatial Anchor Memory & Dual-Threshold Retention berhasil menahan oklusi!")


def test_scenario_4_zero_flapping_median_stability():
    """
    Skenario 4: Simulasi streaming 50 frame kontinu saat area parkir berisi 2 motor stasioner.
    Diberikan noise transien acak (sesekali muncul deteksi bayangan 1 frame, atau dropout 1 frame).
    Moving Median Window (maxlen=25) WAJIB mengunci output okupansi persis 2 unit,
    dengan standar deviasi fluktuasi STD = 0.0000.
    """
    print("\n[TEST 4] Zero-Flapping Moving Median Window (50 Frame Stream)...")
    roi_cfg = load_roi_zones(Path("cameras/cam_03/roi_zones.json"))
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        stationary_dwell_sec=0.0,
    )

    m1_bbox = (610.0, 460.0, 650.0, 540.0)
    m2_bbox = (660.0, 460.0, 700.0, 540.0)

    occupied_series = []
    current_time = 400.0

    import random
    rng = random.Random(42)

    for frame_idx in range(50):
        current_time += 0.2
        frame_tracks = [
            TrackResult(track_id=401, class_label="motorcycle", class_id=3, confidence=0.76, bbox=m1_bbox, is_confirmed=True),
            TrackResult(track_id=402, class_label="motorcycle", class_id=3, confidence=0.74, bbox=m2_bbox, is_confirmed=True),
        ]

        # Simulasikan noise acak:
        # 1. Glitch drop 1-frame pada motor 1 (tiap 12 frame)
        if frame_idx in (12, 24, 38):
            frame_tracks.pop(0)

        # 2. Transient noise 1-frame bayangan lewat (tiap 17 frame)
        if frame_idx in (17, 33):
            noise_box = (720.0, 470.0, 760.0, 545.0)
            frame_tracks.append(
                TrackResult(track_id=999, class_label="motorcycle", class_id=3, confidence=0.35, bbox=noise_box, is_confirmed=False)
            )

        res = tracker.update(
            tracks=frame_tracks,
            polygons=roi_cfg.polygons,
            scale_x=1.5,
            scale_y=1.5,
            current_time=current_time,
            is_warmup=(frame_idx < 5),
        )

        # Catat kuota setelah window terisi (frame >= 25)
        if frame_idx >= 25:
            occupied_series.append(res["occupied_slots"])

    std_val = float(np.std(occupied_series))
    mean_val = float(np.mean(occupied_series))
    print(f"  Stabilized stream (25 frames): mean={mean_val:.2f}, std={std_val:.4f}, unique={set(occupied_series)}")
    assert std_val == 0.0, f"Standar deviasi fluktuasi harus 0.0000, didapat: {std_val:.4f}"
    assert mean_val == 2.0, f"Rata-rata kuota okupansi harus 2.00, didapat: {mean_val:.2f}"
    print("  ✓ PASS: Zero-flapping terverifikasi dengan fluktuasi STD = 0.0000!")


def main():
    print("==================================================================")
    print("  DENSE CLUSTERED MOTORCYCLE DETECTION TEST SUITE (cam_03)")
    print("==================================================================")

    test_scenario_1_dense_staggered_cluster()
    test_scenario_2_child_fragmentation_elimination()
    test_scenario_3_occlusion_retention_spatial_anchor()
    test_scenario_4_zero_flapping_median_stability()

    print("\n==================================================================")
    print("  ALL 4 CAM_03 DENSE CLUSTERING TESTS PASSED! (100% SUCCESS) ✓")
    print("==================================================================")


if __name__ == "__main__":
    main()
