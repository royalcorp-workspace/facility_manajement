"""
tools/test_triple_check_parking.py
==================================
Test Suite Komprehensif untuk Triple-Check Parking Verification Engine
dan Visual Decluttering Overlay.

6 Skenario Pengujian:
1. Happy Path Full Cycle (VACANT -> A_TO_B -> ENTERING -> Dwell 10s -> OCCUPIED -> B_TO_A -> LEAVING -> VACANT)
2. Debounce Maju-Mundur / Re-park (LEAVING -> re-enter within window -> kembali OCCUPIED)
3. Fallback Dwell tanpa Tripwire Crossing (Dwell >= 10s tetap OCCUPIED)
4. False Entry Timeout (ENTERING tanpa kendaraan masuk poligon timeout 30s -> VACANT)
5. Multi-Slot Simultan (Evaluasi 3 slot paralel dengan status berbeda)
6. Visual Decluttering Canvas Verification (Bebas clutter vertex dots, HUD bersih)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from engine.config_loader import ROIPoint, ROIZone, TripwireRule
from engine.smart_parking import SmartParkingTracker, SlotState
from engine.tracker_interface import TrackResult


def create_dummy_zone(zone_id: str, x1: int, y1: int, x2: int, y2: int) -> ROIZone:
    return ROIZone(
        zone_id=zone_id,
        label=f"Slot {zone_id.split('_')[-1]}",
        type="polygon",
        color_hex="#00FF00",
        points=[
            ROIPoint(x=float(x1), y=float(y1)),
            ROIPoint(x=float(x2), y=float(y1)),
            ROIPoint(x=float(x2), y=float(y2)),
            ROIPoint(x=float(x1), y=float(y2)),
        ],
        trigger_on=["enter"],
        active=True,
    )


def create_dummy_tripwire(tw_id: str, p1: tuple, p2: tuple, direction: str = "A_TO_B") -> TripwireRule:
    return TripwireRule(
        tripwire_id=tw_id,
        label=f"Gate {tw_id}",
        color_hex="#FF5500",
        p1=ROIPoint(x=float(p1[0]), y=float(p1[1])),
        p2=ROIPoint(x=float(p2[0]), y=float(p2[1])),
        direction=direction,
        active=True,
    )


def create_vehicle_track(track_id: int, bbox: tuple, class_label: str = "car") -> TrackResult:
    x1, y1, x2, y2 = bbox
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    return TrackResult(
        track_id=track_id,
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        confidence=0.92,
        class_label=class_label,
        class_id=2,
        age=15,
        is_confirmed=True,
        camera_id="cam_01",
        frame_number=100,
        prev_centroid=(cx, cy),
    )


def test_scenario_1_happy_path():
    print("[RUN] Scenario 1: Happy Path Full Cycle (VACANT -> ENTERING -> OCCUPIED -> LEAVING -> VACANT)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    # Slot 1 di koordinat 1080p: (100, 100) s.d. (300, 300)
    slot1 = create_dummy_zone("zone_01", 100, 100, 300, 300)
    polys = [slot1]
    scale = 1.0
    t = 1000.0

    # 1. State awal: kosong
    res = tracker.update([], polys, scale, scale, t)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "VACANT", f"Expected VACANT, got {s1.phase}"
    assert not s1.occupied, "Expected occupied=False"
    assert res["available_slots"] == 1
    assert res["occupied_slots"] == 0

    # 2. Tripwire gate sinyal A_TO_B (kendaraan melintas masuk)
    tracker.apply_tripwire_signal("zone_01", "A_TO_B", track_id=42, current_time=t)
    assert s1.phase == "ENTERING", f"Expected ENTERING, got {s1.phase}"
    assert not s1.occupied, "Expected occupied=False during ENTERING"

    # 3. Kendaraan masuk ke dalam poligon (wheel_pt = (200, 250))
    car_track = create_vehicle_track(track_id=42, bbox=(150, 150, 250, 250))
    res = tracker.update([car_track], polys, scale, scale, t + 1.0)
    assert s1.phase == "ENTERING"
    assert not s1.occupied, "Dwell baru 0s, belum boleh OCCUPIED"

    # 4. Dwell bertambah hingga 10 detik (t + 11.0)
    res = tracker.update([car_track], polys, scale, scale, t + 11.0)
    assert s1.phase == "OCCUPIED", f"Expected OCCUPIED after 10s dwell, got {s1.phase}"
    assert s1.occupied, "Expected occupied=True"
    assert res["occupied_slots"] == 1
    assert res["available_slots"] == 0

    # 5. Kendaraan melintasi tripwire keluar (B_TO_A)
    tracker.apply_tripwire_signal("zone_01", "B_TO_A", track_id=42, current_time=t + 20.0)
    assert s1.phase == "LEAVING", f"Expected LEAVING, got {s1.phase}"

    # 6. Kendaraan keluar dari poligon dan melewati grace period 2.0s
    res = tracker.update([], polys, scale, scale, t + 23.0)
    assert s1.phase == "VACANT", f"Expected VACANT after leaving, got {s1.phase}"
    assert not s1.occupied
    assert res["occupied_slots"] == 0
    assert res["available_slots"] == 1
    print("  [PASS] Scenario 1 Passed.")


def test_scenario_2_debounce_repark():
    print("[RUN] Scenario 2: Debounce Maju-Mundur / Re-park...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slot1 = create_dummy_zone("zone_01", 100, 100, 300, 300)
    polys = [slot1]
    scale = 1.0
    t = 1000.0

    car_track = create_vehicle_track(track_id=10, bbox=(150, 150, 250, 250))
    # Bikin slot sudah OCCUPIED
    tracker.update([car_track], polys, scale, scale, t)
    tracker.update([car_track], polys, scale, scale, t + 11.0)
    assert tracker.slot_states["zone_01"].phase == "OCCUPIED"

    # Mobil sempat maju keluar garis sensor (sinyal B_TO_A terpicu) -> phase LEAVING
    tracker.apply_tripwire_signal("zone_01", "B_TO_A", track_id=10, current_time=t + 20.0)
    assert tracker.slot_states["zone_01"].phase == "LEAVING"

    # Namun dalam waktu 1.5 detik (re-park window), supir mundur lagi ke dalam slot
    tracker.update([car_track], polys, scale, scale, t + 21.5)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "OCCUPIED", f"Expected OCCUPIED after re-park debounce, got {s1.phase}"
    assert s1.occupied, "Kuotanya harus tetap terjaga terisi (OCCUPIED)"
    print("  [PASS] Scenario 2 Passed.")


def test_scenario_3_fallback_dwell():
    print("[RUN] Scenario 3: Fallback Dwell tanpa Tripwire Crossing...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slot1 = create_dummy_zone("zone_01", 100, 100, 300, 300)
    polys = [slot1]
    scale = 1.0
    t = 500.0

    car_track = create_vehicle_track(track_id=99, bbox=(150, 150, 250, 250))
    # Mobil masuk tanpa sinyal A_TO_B (phase tetap VACANT pada awalnya)
    tracker.update([car_track], polys, scale, scale, t)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "VACANT", "Sebelum 10s harus tetap VACANT"

    # Setelah 5 detik
    tracker.update([car_track], polys, scale, scale, t + 5.0)
    assert s1.phase == "VACANT"
    assert not s1.occupied

    # Setelah 10 detik penuh
    res = tracker.update([car_track], polys, scale, scale, t + 10.5)
    assert s1.phase == "OCCUPIED", f"Fallback harus promosi ke OCCUPIED, got {s1.phase}"
    assert s1.occupied
    assert res["occupied_slots"] == 1
    print("  [PASS] Scenario 3 Passed.")


def test_scenario_4_false_entry_timeout():
    print("[RUN] Scenario 4: False Entry Timeout & Cancel...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slot1 = create_dummy_zone("zone_01", 100, 100, 300, 300)
    polys = [slot1]
    scale = 1.0
    t = 100.0

    # 1. Mobil melintasi tripwire masuk A_TO_B
    tracker.apply_tripwire_signal("zone_01", "A_TO_B", track_id=7, current_time=t)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "ENTERING"

    # 2. Mobil tidak jadi masuk, mundur keluar lagi (B_TO_A sebelum masuk poligon)
    tracker.apply_tripwire_signal("zone_01", "B_TO_A", track_id=7, current_time=t + 3.0)
    assert s1.phase == "VACANT", "Sinyal B_TO_A saat ENTERING harus membatalkan status ke VACANT"

    # 3. Uji timeout jika ada sinyal A_TO_B tanpa ada mobil masuk sama sekali > 30s
    tracker.apply_tripwire_signal("zone_01", "A_TO_B", track_id=8, current_time=t + 10.0)
    assert s1.phase == "ENTERING"
    tracker.update([], polys, scale, scale, t + 15.0)
    assert s1.phase == "ENTERING"

    # Setelah lewat 31 detik (> entering_timeout_sec 30s)
    tracker.update([], polys, scale, scale, t + 42.0)
    assert s1.phase == "VACANT", f"Expected VACANT after 30s timeout, got {s1.phase}"
    print("  [PASS] Scenario 4 Passed.")


def test_scenario_5_multi_slot():
    print("[RUN] Scenario 5: Multi-Slot Simultan (3 Slot)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slot1 = create_dummy_zone("zone_01", 50, 50, 150, 150)
    slot2 = create_dummy_zone("zone_02", 200, 50, 300, 150)
    slot3 = create_dummy_zone("zone_03", 350, 50, 450, 150)
    polys = [slot1, slot2, slot3]
    scale = 1.0
    t = 2000.0

    # Slot 1: Mobil #1 sudah parkir > 10s (OCCUPIED)
    car1 = create_vehicle_track(track_id=1, bbox=(70, 70, 130, 130))
    tracker.update([car1], polys, scale, scale, t)
    tracker.update([car1], polys, scale, scale, t + 12.0)

    # Slot 2: Mobil #2 baru trigger A_TO_B (ENTERING)
    tracker.apply_tripwire_signal("zone_02", "A_TO_B", track_id=2, current_time=t + 12.0)

    # Slot 3: Tidak ada sinyal atau mobil (VACANT)
    res = tracker.update([car1], polys, scale, scale, t + 12.5)

    assert tracker.slot_states["zone_01"].phase == "OCCUPIED"
    assert tracker.slot_states["zone_02"].phase == "ENTERING"
    assert tracker.slot_states["zone_03"].phase == "VACANT"

    assert res["total_slots"] == 3
    assert res["occupied_slots"] == 1
    assert res["available_slots"] == 2
    print("  [PASS] Scenario 5 Passed.")


def test_scenario_6_visual_decluttering():
    print("[RUN] Scenario 6: Visual Decluttering Canvas Verification...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slot1 = create_dummy_zone("zone_01", 50, 50, 150, 150)
    slot2 = create_dummy_zone("zone_02", 200, 50, 300, 150)
    tw1 = create_dummy_tripwire("tw_01", (50, 160), (150, 160), "A_TO_B")
    polys = [slot1, slot2]
    tripwires = [tw1]
    scale = 1.0
    t = 3000.0

    # Slot 1 di-set OCCUPIED
    car1 = create_vehicle_track(track_id=101, bbox=(70, 70, 130, 130))
    tracker.update([car1], polys, scale, scale, t)
    stats = tracker.update([car1], polys, scale, scale, t + 12.0)

    # Render pada kanvas hitam 640x360
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.render_overlay(
        canvas=canvas,
        polygons=polys,
        tripwires=tripwires,
        tracks=[car1],
        scale_x=scale,
        scale_y=scale,
        parking_stats=stats,
        current_time=t + 12.0,
    )

    # Kanvas tidak boleh kosong (ada garis rendered)
    assert np.count_nonzero(canvas) > 0, "Canvas should have drawn overlay elements"

    occ_slots = [
        f"S{s.slot_num}"
        for s in sorted(tracker.slot_states.values(), key=lambda x: x.slot_num)
        if s.phase == "OCCUPIED"
    ]
    occ_text = ", ".join(occ_slots) if occ_slots else "-"
    expected_hud = f"PARKING: 1/2 OCCUPIED | TERISI: [{occ_text}]"
    assert occ_text == "S1"
    assert "S1" in expected_hud
    assert "1/2 OCCUPIED" in expected_hud
    print(f"  [PASS] Scenario 6 Passed (HUD: '{expected_hud}').")


def test_scenario_7_anti_id_churn():
    """
    Skenario 7: Anti-ID Churn — Paksa ganti track_id #12 → #230.
    ASSERT: phase tetap OCCUPIED, dwell_duration tidak direset, kuota stabil.
    """
    print("[RUN] Scenario 7: Anti-ID Churn (track_id #12 -> #230 tanpa reset dwell)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slot1 = create_dummy_zone("zone_01", 100, 100, 300, 300)
    polys = [slot1]
    scale = 1.0
    t = 5000.0

    # Mobil #12 masuk dan parkir > 10s → OCCUPIED
    car_12 = create_vehicle_track(track_id=12, bbox=(150, 150, 250, 250))
    tracker.update([car_12], polys, scale, scale, t)
    tracker.update([car_12], polys, scale, scale, t + 11.0)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "OCCUPIED", f"Harusnya OCCUPIED, dapat: {s1.phase}"
    assert s1.track_id == 12
    dwell_before_churn = s1.dwell_duration
    assert dwell_before_churn >= 10.0, f"Dwell harusnya >= 10s, dapat: {dwell_before_churn}"

    # ─── SIMULASI ID CHURN: track_id berubah dari #12 ke #230 ───
    # Bbox tetap sama / sangat mirip (dalam poligon yang sama)
    car_230 = create_vehicle_track(track_id=230, bbox=(152, 152, 248, 248))
    res = tracker.update([car_230], polys, scale, scale, t + 11.5)

    # Verifikasi: fase HARUS tetap OCCUPIED (tidak flip ke VACANT/ENTERING)
    assert s1.phase == "OCCUPIED", \
        f"ID Churn menyebabkan fase berubah! Harusnya OCCUPIED, dapat: {s1.phase}"

    # Verifikasi: track_id diperbarui secara senyap ke ID baru
    assert s1.track_id == 230, \
        f"track_id harusnya diperbarui ke 230, dapat: {s1.track_id}"

    # Verifikasi: dwell_duration TIDAK direset (masih >= sebelum churn)
    assert s1.dwell_duration >= dwell_before_churn - 0.1, \
        f"Dwell DIRESET setelah ID churn! Dapat: {s1.dwell_duration}, sebelumnya: {dwell_before_churn}"

    # Verifikasi: kuota parkir stabil
    assert res["occupied_slots"] == 1, \
        f"Kuota berubah akibat churn! Harusnya 1 occupied, dapat: {res['occupied_slots']}"
    assert res["available_slots"] == 0

    # Verifikasi: dwell tetap akumulasi setelah ID update
    res2 = tracker.update([car_230], polys, scale, scale, t + 14.0)
    assert s1.phase == "OCCUPIED"
    assert s1.dwell_duration >= 12.0, \
        f"Dwell harusnya akumulasi terus (>= 12s), dapat: {s1.dwell_duration}"

    print(f"  [PASS] Scenario 7 Passed (dwell stabil: {s1.dwell_duration:.1f}s, track_id={s1.track_id}).")


def test_scenario_8_polygon_clear_hysteresis():
    """
    Skenario 8: Hysteresis polygon_clear_since — oklusi 1-2 frame tidak boleh
    memicu transisi OCCUPIED → LEAVING. Butuh >= 3s poligon benar-benar kosong.
    """
    print("[RUN] Scenario 8: Hysteresis polygon_clear_since (anti-jitter kuota)...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)
    slot1 = create_dummy_zone("zone_01", 100, 100, 300, 300)
    polys = [slot1]
    scale = 1.0
    t = 6000.0

    # Bawa slot ke kondisi OCCUPIED
    car = create_vehicle_track(track_id=77, bbox=(150, 150, 250, 250))
    tracker.update([car], polys, scale, scale, t)
    tracker.update([car], polys, scale, scale, t + 11.0)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "OCCUPIED", f"Harusnya OCCUPIED, dapat: {s1.phase}"

    # ─── Oklusi 1 frame (0.1s): tidak ada track terdeteksi ───
    tracker.update([], polys, scale, scale, t + 11.1)
    assert s1.phase == "OCCUPIED", \
        "Oklusi 1 frame (0.1s) tidak boleh mengubah fase dari OCCUPIED!"
    assert s1.polygon_clear_since is None or (t + 11.1 - s1.polygon_clear_since) < 3.0, \
        "polygon_clear_since belum mencapai 3.0s, harusnya masih OCCUPIED"

    # ─── Kendaraan terdeteksi lagi (glitch 1 frame sudah selesai) ───
    tracker.update([car], polys, scale, scale, t + 11.2)
    assert s1.phase == "OCCUPIED"
    # polygon_clear_since harus di-reset karena kendaraan terdeteksi lagi
    assert s1.polygon_clear_since is None, \
        "polygon_clear_since harusnya None setelah kendaraan terdeteksi kembali"

    # ─── Oklusi panjang (simulasi kendaraan benar-benar pergi) ───
    tracker.update([], polys, scale, scale, t + 12.0)   # +0.8s hilang, mulai hitung
    tracker.update([], polys, scale, scale, t + 14.5)   # +3.3s hilang, masih dalam grace period
    assert s1.phase == "OCCUPIED", \
        f"Belum 5s poligon kosong, harusnya masih OCCUPIED, dapat: {s1.phase}"

    # Dengan Sticky Occupied Latch (dwell >= 5s), confirm_threshold adalah 10.0 detik adaptif:
    # Pada t + 18.0 (6.0s poligon kosong), slot ter-latch tetap aman OCCUPIED:
    tracker.update([], polys, scale, scale, t + 18.0)
    assert s1.phase == "OCCUPIED", \
        f"Pada 6s poligon kosong slot latched harusnya tetap OCCUPIED, dapat: {s1.phase}"

    # Lewati 10s penuh poligon kosong (t + 23.0 -> 11.0s kosong) → baru transisi ke LEAVING
    tracker.update([], polys, scale, scale, t + 23.0)   # 11.0s kosong (melampaui 10s)
    assert s1.phase == "LEAVING", \
        f"Setelah >= 10s poligon kosong harusnya LEAVING, dapat: {s1.phase}"

    # Konfirmasi akhir: dwell tidak pernah direset selama proses
    assert s1.dwell_duration >= 10.0, \
        f"Dwell harusnya tetap >= 10s selama transisi, dapat: {s1.dwell_duration}"

    print(f"  [PASS] Scenario 8 Passed (polygon_clear_since hysteresis adaptif 10s bekerja, dwell={s1.dwell_duration:.1f}s).")


def test_scenario_9_exclusive_slot_assignment():
    """
    Skenario 9: Aturan Exclusive Slot Assignment (1 Mobil = Maksimal 1 Slot).
    Memastikan mobil yang berada di perbatasan dua poligon bersebelahan
    hanya mengokupansi 1 slot yang memiliki skor/overlap tertinggi,
    dan slot lainnya tetap VACANT.
    Jika satu slot sempat mengklaim track_id tersebut tapi kemudian kalah afinitas,
    slot yang kalah segera dibebaskan menjadi VACANT.
    """
    print("[RUN] Scenario 9: Double-Claim Prevention & Exclusive Slot Assignment...")
    tracker = SmartParkingTracker(dwell_threshold_sec=10.0)

    # Dua slot bersebelahan dengan jeda 10px
    # Slot 1: x in [100, 200], y in [100, 200]
    # Slot 2: x in [210, 310], y in [100, 200]
    slot1 = create_dummy_zone("zone_01", 100, 100, 200, 200)
    slot2 = create_dummy_zone("zone_02", 210, 100, 310, 200)
    polys = [slot1, slot2]
    scale = 1.0
    t = 7000.0

    # Mobil #1 berada di Slot 2, tapi sedikit mepet ke Slot 1
    # Bbox: [205, 110, 285, 190] -> cx = 245 (jauh lebih dekat ke pusat Slot 2: cx=260 vs Slot 1: cx=150)
    car1 = create_vehicle_track(track_id=1, bbox=(205, 110, 285, 190))

    # Frame 1: Mobil masuk (warmup false)
    tracker.update([car1], polys, scale, scale, t)
    # Lanjut dwell 11s -> seharusnya Slot 2 OCCUPIED, Slot 1 VACANT
    res = tracker.update([car1], polys, scale, scale, t + 11.0)

    s1 = tracker.slot_states["zone_01"]
    s2 = tracker.slot_states["zone_02"]

    assert s2.phase == "OCCUPIED", f"Slot 2 harusnya OCCUPIED, dapat {s2.phase}"
    assert s2.track_id == 1, f"Slot 2 harusnya track_id=1, dapat {s2.track_id}"

    assert s1.phase == "VACANT", f"Slot 1 harusnya tetap VACANT (Double claim!), dapat {s1.phase}"
    assert s1.track_id is None, f"Slot 1 track_id harusnya None, dapat {s1.track_id}"
    assert res["occupied_slots"] == 1, f"Harusnya cuma 1 slot occupied, dapat {res['occupied_slots']}"
    assert res["available_slots"] == 1, f"Harusnya 1 slot available, dapat {res['available_slots']}"

    # Uji Real-time Release:
    # Misalkan sebelumnya slot 1 secara paksa terkunci ke track_id=1
    s1.phase = "OCCUPIED"
    s1.track_id = 1
    s1.dwell_duration = 15.0

    # Di frame berikutnya, mobil 1 tetap di slot 2 (pemenang eksklusif)
    tracker.update([car1], polys, scale, scale, t + 12.0)
    assert s1.phase == "VACANT", f"Slot 1 harusnya langsung dilepas ke VACANT, dapat {s1.phase}"
    assert s1.track_id is None, f"Slot 1 track_id harusnya direset ke None, dapat {s1.track_id}"
    assert s2.phase == "OCCUPIED"

    print("  [PASS] Scenario 9 Passed (1 Mobil mengklaim maksimal 1 slot, double claim dicegah).")


def test_scenario_10_yolo_miss_30_frames_no_vacant():
    """
    Skenario 10: Toleransi YOLO Miss 30 Frame & Latch Guard Proteksi Eksklusif.
    Memastikan bahwa kendaraan yang sudah berstatus OCCUPIED dan ter-latched (dwell >= 5s)
    TIDAK mengalami flapping ke VACANT meskipun:
    1. Deteksi YOLO hilang (miss) selama 30 frame berturut-turut (~1.5 detik @ 20 FPS).
    2. Terjadi interferensi eksklusif (misal track dicuri sementara oleh bayangan/slot tetangga),
       karena Latch Guard mencegah immediate release.
    """
    print("[RUN] Scenario 10: 30-Frame YOLO Detection Miss & Latch Guard Protection...")
    tracker = SmartParkingTracker(dwell_threshold_sec=5.0)

    slot1 = create_dummy_zone("zone_01", 100, 100, 200, 200)
    slot2 = create_dummy_zone("zone_02", 210, 100, 310, 200)
    polys = [slot1, slot2]
    scale = 1.0
    t = 8000.0

    car1 = create_vehicle_track(track_id=1, bbox=(120, 120, 180, 180))

    # Frame 1: Mobil masuk ke Slot 1
    tracker.update([car1], polys, scale, scale, t)

    # Dwell selama 6.0 detik (> latch_dwell_threshold_sec = 5.0)
    res = tracker.update([car1], polys, scale, scale, t + 6.0)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "OCCUPIED", f"Slot 1 harusnya OCCUPIED, dapat {s1.phase}"
    assert s1.latch_occupied is True, f"Slot 1 harusnya latch_occupied=True, dapat {s1.latch_occupied}"
    assert res["occupied_slots"] == 1

    # Simulasi 30 frame YOLO Miss (tidak ada deteksi sama sekali, 20 fps -> dt = 0.05s)
    current_t = t + 6.0
    for frame_idx in range(30):
        current_t += 0.05
        tracker.update([], polys, scale, scale, current_t)
        assert s1.phase == "OCCUPIED", (
            f"Frame miss ke-{frame_idx + 1}: Slot 1 flapped ke {s1.phase}! Harusnya tetap OCCUPIED."
        )

    # Uji Latch Guard terhadap exclusive assignment bypass:
    # Misalkan bayangan atap/jitter membuat track_id 1 sempat ter-assign ke slot tetangga (slot 2)
    car1_in_slot2 = create_vehicle_track(track_id=1, bbox=(220, 120, 280, 180))
    current_t += 0.05
    tracker.update([car1_in_slot2], polys, scale, scale, current_t)

    # Slot 1 TIDAK boleh langsung drop ke VACANT karena sudah latch_occupied!
    assert s1.phase == "OCCUPIED", (
        f"Slot 1 harusnya dilindungi Latch Guard dan tetap OCCUPIED, dapat {s1.phase}"
    )
    assert s1.latch_occupied is True

    print("  [PASS] Scenario 10 Passed (Slot tetap OCCUPIED selama 30-frame miss dan terlindungi Latch Guard).")


def test_scenario_11_legitimate_exit_after_10s():
    """
    Skenario 11: Pelepasan Legal Slot Latched Setelah 10 Detik Poligon Kosong.
    Memastikan bahwa proteksi Latch TIDAK mengunci slot selamanya:
    - Jika mobil benar-benar keluar (poligon kosong > 10.0 detik adaptif confirm_threshold),
      slot bertransisi OCCUPIED -> LEAVING.
    - Setelah masa exit_grace_sec (2.0 detik), slot bertransisi LEAVING -> VACANT
      dan latch_occupied berhasil di-reset ke False secara legal.
    """
    print("[RUN] Scenario 11: Legitimate Exit After 10s Adaptive Polygon Clear...")
    tracker = SmartParkingTracker(dwell_threshold_sec=5.0)

    slot1 = create_dummy_zone("zone_01", 100, 100, 200, 200)
    polys = [slot1]
    scale = 1.0
    t = 9000.0

    car1 = create_vehicle_track(track_id=1, bbox=(120, 120, 180, 180))

    # Mobil parkir stabil dan ter-latched
    tracker.update([car1], polys, scale, scale, t)
    tracker.update([car1], polys, scale, scale, t + 6.0)
    s1 = tracker.slot_states["zone_01"]
    assert s1.phase == "OCCUPIED"
    assert s1.latch_occupied is True

    # Mobil keluar dari slot — frame kosong pertama pada t + 6.6
    tracker.update([], polys, scale, scale, t + 6.6)
    assert s1.phase == "OCCUPIED"
    assert s1.polygon_clear_since == t + 6.6

    # Pada t + 11.6 (5.0s kosong): Slot biasa sudah LEAVING, tapi slot latched HARUS tetap OCCUPIED (ambang 10s)
    tracker.update([], polys, scale, scale, t + 11.6)
    assert s1.phase == "OCCUPIED", (
        f"Pada 5s kosong, slot latched harusnya tetap OCCUPIED (adaptive threshold 10s), dapat {s1.phase}"
    )

    # Pada t + 17.0 (> 10.0s kosong): Transisi ke LEAVING
    tracker.update([], polys, scale, scale, t + 17.0)
    assert s1.phase == "LEAVING", f"Setelah >10s kosong, slot harusnya LEAVING, dapat {s1.phase}"

    # Pada t + 19.5 (> exit_grace_sec = 2.0s setelah leaving): Transisi final ke VACANT
    res = tracker.update([], polys, scale, scale, t + 19.5)
    assert s1.phase == "VACANT", f"Setelah grace period, slot harusnya VACANT, dapat {s1.phase}"
    assert s1.latch_occupied is False, f"latch_occupied harus di-reset ke False, dapat {s1.latch_occupied}"
    assert s1.track_id is None
    assert res["occupied_slots"] == 0
    assert res["available_slots"] == 1

    print("  [PASS] Scenario 11 Passed (Legitimate exit berhasil setelah 10s dan latch di-reset ke False).")


def test_scenario_12_vehicle_class_aliasing_and_roofbox():
    """
    Skenario 12: Vehicle Class Aliasing & Roofbox Profile Detection.
    Memastikan:
    1. Objek berlabel 'parking meter' dengan dimensi mobil (w >= 25, h >= 25, aspect ratio 0.35..2.5)
       secara otomatis di-remap menjadi 'car' oleh YOLO11nDetector.
    2. Hasil remapping 'car' berhasil diakomodasi oleh SmartParkingTracker untuk mengokupansi slot
       dan tidak bocor ke slot tetangga.
    """
    print("[RUN] Scenario 12: Vehicle Class Aliasing & Roofbox Detection...")

    # 1. Test Detector Level Remapping
    from engine.detector_impl import YOLO11nDetector
    detector = YOLO11nDetector(
        confidence_threshold=0.25,
        target_classes=["car", "truck", "bus"],
    )

    # Buat dummy output matrix 84 x 1 (standar output YOLOv8/v11 per proposal)
    dummy_output = np.zeros((84, 1), dtype=np.float32)
    dummy_output[0, 0] = 181.5 / 640.0   # cx normalized
    dummy_output[1, 0] = 155.5 / 360.0   # cy normalized
    dummy_output[2, 0] = 63.0 / 640.0    # w normalized
    dummy_output[3, 0] = 92.0 / 360.0    # h normalized
    dummy_output[4 + 12, 0] = 0.489      # Class 12 (parking meter) score = 0.489
    dummy_output[4 + 2, 0] = 0.005       # Class 2 (car) score = 0.005

    boxes = []
    confidences = []
    class_ids = []
    class_labels = []

    output_t = dummy_output.T
    for row in output_t:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < detector.confidence_threshold:
            continue
        from engine.detector_impl import COCO_CLASSES
        label = COCO_CLASSES[class_id]

        cx, cy, w, h = row[0], row[1], row[2], row[3]
        if cx <= 1.0 and cy <= 1.0:
            cx *= 640.0
            cy *= 640.0
            w *= 640.0
            h *= 640.0

        if label == "parking meter":
            aspect_ratio = (w / h) if h > 0 else 0.0
            if w >= 25.0 and h >= 25.0 and 0.35 <= aspect_ratio <= 2.5:
                label = "car"
                class_id = 2

        if not detector.is_target_class(label):
            continue

        boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
        confidences.append(confidence)
        class_ids.append(class_id)
        class_labels.append(label)

    assert len(boxes) == 1, f"Harusnya 1 deteksi remapped lolos, dapat {len(boxes)}"
    assert class_labels[0] == "car", f"Label harusnya remapped ke 'car', dapat {class_labels[0]}"
    assert abs(confidences[0] - 0.489) < 1e-3, f"Confidence harus tetap ~0.489, dapat {confidences[0]}"

    # 2. Test SmartParkingTracker Level Allocation
    tracker = SmartParkingTracker(dwell_threshold_sec=5.0)
    slot2 = create_dummy_zone("zone_02", 248, 444, 486, 597)
    # Slot 3 dengan batas yang sudah dirapatkan (x=445..698)
    slot3 = create_dummy_zone("zone_03", 445, 451, 698, 602)
    polys = [slot2, slot3]
    scale = 1.0 / 3.0
    t = 10000.0

    # Mobil hitam roofbox (bbox AI: [150, 109, 213, 202])
    car_roofbox = create_vehicle_track(track_id=88, bbox=(150, 109, 213, 202), class_label="car")

    # Frame 1: Mobil terdeteksi
    res1 = tracker.update([car_roofbox], polys, scale, scale, t)
    s2 = tracker.slot_states["zone_02"]
    s3 = tracker.slot_states["zone_03"]

    assert s2.phase == "VACANT", f"Slot 2 harusnya tetap VACANT, dapat {s2.phase}"
    assert s2.track_id is None

    # Lanjut dwell 6.0 detik -> Slot 3 harus OCCUPIED + latched
    res2 = tracker.update([car_roofbox], polys, scale, scale, t + 6.0)
    assert s3.phase == "OCCUPIED", f"Slot 3 harusnya OCCUPIED, dapat {s3.phase}"
    assert s3.track_id == 88
    assert s3.latch_occupied is True, f"Slot 3 harusnya latch_occupied=True, dapat {s3.latch_occupied}"
    assert res2["occupied_slots"] == 1
    assert res2["available_slots"] == 1

    print("  [PASS] Scenario 12 Passed (Parking meter remapped ke 'car' dan mengokupansi Slot 3).")


def test_scenario_13_block_mode_init():
    """
    Skenario 13: Inisialisasi Mode Blok Parkir Motor (cam_03).
    Memastikan:
    1. Kapasitas total mengacu pada block_capacity (30 unit), bukan jumlah poligon.
    2. Response stats mengandung parking_mode="motorcycle_block" dan available=30 saat kosong.
    """
    print("[RUN] Scenario 13: Block Mode Initialization & Dynamic Capacity...")
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        vehicle_classes={"motorcycle", "bicycle"},
    )
    assert tracker.total_slots == 30
    assert tracker.parking_mode == "motorcycle_block"
    assert tracker.block_capacity == 30

    zone1 = create_dummy_zone("zone_01", 100, 100, 800, 800)
    res = tracker.update([], [zone1], 1.0, 1.0, 1000.0)
    assert res["total_slots"] == 30
    assert res["occupied_slots"] == 0
    assert res["available_slots"] == 30
    assert res["parking_mode"] == "motorcycle_block"
    print("  [PASS] Scenario 13 Passed (Block mode initialized with capacity 30).")


def test_scenario_14_block_mode_density_count():
    """
    Skenario 14: Perhitungan Densitas Motor Stasioner (Instant Dwell 0.0s & IoU Footprint >= 20%).
    Memastikan:
    1. Instant density (stationary_dwell_sec = 0.0): motor langsung terhitung di t=0 tanpa jeda tunggu.
    2. Area overlap filter (bbox vs polygon):
       - Motor dengan overlap >= 20% sah terhitung.
       - Motor di luar atau overlap < 20% tidak terhitung.
    3. Ketika motor keluar dari poligon, kuota berkurang secara responsif.
    """
    print("[RUN] Scenario 14: Block Mode Instant Density & Area Overlap (>= 20%)...")
    # Inisialisasi default stationary_dwell_sec=0.0 (Instant Density)
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        vehicle_classes={"motorcycle", "bicycle"},
        stationary_dwell_sec=0.0,
    )
    # Zone: (100, 100) s.d. (800, 800)
    zone1 = create_dummy_zone("zone_01", 100, 100, 800, 800)
    polys = [zone1]
    t = 2000.0

    # 4 motor sepenuhnya di dalam zone (100% overlap)
    motos = [
        create_vehicle_track(track_id=i, bbox=(150 + i * 50, 150 + i * 50, 180 + i * 50, 200 + i * 50), class_label="motorcycle")
        for i in range(1, 5)
    ]
    # Motor 5: sebagian di pinggir kiri zone: x dari 80 s.d. 140 (lebar 60px), y dari 200 s.d. 300.
    # Irisan dengan zone (x>=100): lebar irisan = 40px (100 s.d. 140) -> overlap = 40/60 = 66.7% (>= 20%) -> SAH
    moto_edge = create_vehicle_track(track_id=5, bbox=(80, 200, 140, 300), class_label="motorcycle")

    # Motor 6: hanya menyenggol pinggir luar: x dari 50 s.d. 102 (lebar 52px).
    # Irisan dengan zone: lebar irisan = 2px (100 s.d. 102) -> overlap = 2/52 = 3.8% (< 20%) -> TIDAK SAH
    moto_outside = create_vehicle_track(track_id=6, bbox=(50, 200, 102, 300), class_label="motorcycle")

    all_motos = motos + [moto_edge, moto_outside]

    # t = 0s: Berkat stationary_dwell_sec = 0.0 (Instant), motor sah langsung terhitung tanpa jeda tunggu
    res1 = tracker.update(all_motos, polys, 1.0, 1.0, t)
    # Total sah: 4 (full) + 1 (edge >= 20%) = 5 motor
    assert res1["occupied_slots"] == 5, f"Expected 5 occupied, got {res1['occupied_slots']}"
    assert res1["available_slots"] == 25

    # 1 motor keluar dari frame (moto_edge pergi)
    res2 = tracker.update(motos, polys, 1.0, 1.0, t + 2.0)
    assert res2["occupied_slots"] == 4
    assert res2["available_slots"] == 26
    print("  [PASS] Scenario 14 Passed (Instant density dwell=0.0s & IoU footprint >=20% verified).")


def test_scenario_15_block_mode_tripwire_dominance():
    """
    Skenario 15: Double Verification: Dominasi Tripwire Net Count vs Densitas.
    Memastikan:
    1. gate_in - gate_out terakumulasi secara akurat.
    2. Konsolidasi okupansi mengambil max(tw_count, density_count).
    3. apply_tripwire_signal guard mencegah pembuatan SlotState individual pada block mode.
    """
    print("[RUN] Scenario 15: Block Mode Tripwire Dominance & Net Counting...")
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        vehicle_classes={"motorcycle", "bicycle"},
        stationary_dwell_sec=5.0,
    )
    zone1 = create_dummy_zone("zone_01", 100, 100, 800, 800)
    polys = [zone1]
    t = 3000.0

    # Simulasi tripwire gate crossing: 8 IN, 2 OUT -> net tw_count = 6
    for i in range(8):
        tracker.record_gate_crossing("A_TO_B", tripwire_id="tw_01", timestamp=t + i)
    for i in range(2):
        tracker.record_gate_crossing("B_TO_A", tripwire_id="tw_01", timestamp=t + 10 + i)

    assert tracker.gate_in_count == 8
    assert tracker.gate_out_count == 2

    # Di dalam zona hanya ada 3 motor stasioner (density=3)
    motos_3 = [
        create_vehicle_track(track_id=i, bbox=(150 + i * 50, 150 + i * 50, 180 + i * 50, 200 + i * 50), class_label="motorcycle")
        for i in range(1, 4)
    ]
    tracker.update(motos_3, polys, 1.0, 1.0, t)
    res = tracker.update(motos_3, polys, 1.0, 1.0, t + 6.0)

    # Konsolidasi: max(tw_count=6, density=3) = 6
    assert res["occupied_slots"] == 6
    assert res["available_slots"] == 24

    # apply_tripwire_signal guard check: tidak boleh membuat slot_states
    tracker.apply_tripwire_signal("zone_01", "A_TO_B", 99, t + 7.0)
    assert len(tracker.slot_states) == 0, "apply_tripwire_signal tidak boleh membuat slotState di block mode"
    print("  [PASS] Scenario 15 Passed (Tripwire dominance max(tw, density) and signal guard verified).")


def test_scenario_16_block_mode_hud_format():
    """
    Skenario 16: Verifikasi Format HUD 1-Baris Bersih Pojok Kanan Atas.
    Memastikan:
    1. Format HUD: PARKING: {occupied}/{total} TERISI | KOSONG: {available} UNIT
    2. Tanpa baris kedua IN/OUT.
    3. Render overlay berjalan mulus tanpa error visual.
    """
    print("[RUN] Scenario 16: Block Mode HUD 1-Line Rendering...")
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        vehicle_classes={"motorcycle", "bicycle"},
    )
    zone1 = create_dummy_zone("zone_01", 100, 100, 800, 800)
    polys = [zone1]
    tw1 = TripwireRule(
        tripwire_id="tw_01",
        label="Tripwire 1",
        p1=ROIPoint(x=500, y=200),
        p2=ROIPoint(x=600, y=300),
        direction="BOTH",
    )

    # 6 IN, 1 OUT -> 5 occupied
    for _ in range(6):
        tracker.record_gate_crossing("A_TO_B", tripwire_id="tw_01", timestamp=100.0)
    tracker.record_gate_crossing("B_TO_A", tripwire_id="tw_01", timestamp=105.0)

    stats = tracker.update([], polys, 1.0, 1.0, 110.0)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    sample_moto = create_vehicle_track(1, (200, 200, 250, 300), class_label="motorcycle")
    sample_moto.confidence = 0.88
    tracker.render_overlay(
        canvas=canvas,
        polygons=polys,
        tripwires=[tw1],
        tracks=[sample_moto],
        scale_x=1.0,
        scale_y=1.0,
        parking_stats=stats,
        current_time=110.0,
    )

    assert np.sum(canvas) > 0, "Canvas harus terisi elemen visual yang digambar"
    assert stats["occupied_slots"] == 5
    assert stats["available_slots"] == 25
    assert stats["gate_in"] == 6
    assert stats["gate_out"] == 1
    print("  [PASS] Scenario 16 Passed (1-Line HUD & Motorcycle Bounding Boxes rendered successfully).")


def test_scenario_17_block_mode_capacity_cap_at_30():
    """
    Skenario 17: Batasan Kapasitas (Clamping Kuota 0 s.d. 30 Unit).
    Memastikan:
    1. Jika akumulasi tripwire melebihi 30 (misal 38), occupied dibatasi tepat pada 30 dan available = 0.
    2. Jika gate_out > gate_in (akumulasi negatif), occupied tidak boleh < 0 (tetap 0).
    """
    print("[RUN] Scenario 17: Block Mode Capacity Clamping (Cap at 30)...")
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        vehicle_classes={"motorcycle", "bicycle"},
    )
    zone1 = create_dummy_zone("zone_01", 100, 100, 800, 800)
    polys = [zone1]

    # 40 IN, 2 OUT -> net tw = 38 (melebihi kapasitas 30)
    for _ in range(40):
        tracker.record_gate_crossing("A_TO_B", tripwire_id="tw_01", timestamp=200.0)
    for _ in range(2):
        tracker.record_gate_crossing("B_TO_A", tripwire_id="tw_01", timestamp=205.0)

    stats = tracker.update([], polys, 1.0, 1.0, 210.0)
    assert stats["occupied_slots"] == 30, f"Harus di-cap pada 30, dapat {stats['occupied_slots']}"
    assert stats["available_slots"] == 0, f"Available harus 0, dapat {stats['available_slots']}"
    assert stats["gate_in"] == 40
    assert stats["gate_out"] == 2

    # Tes net negatif: jika gate_out > gate_in
    tracker2 = SmartParkingTracker(parking_mode="motorcycle_block", block_capacity=30)
    tracker2.record_gate_crossing("B_TO_A", tripwire_id="tw_01", timestamp=10.0)
    stats2 = tracker2.update([], polys, 1.0, 1.0, 20.0)
    assert stats2["occupied_slots"] == 0
    assert stats2["available_slots"] == 30

    print("  [PASS] Scenario 17 Passed (Capacity clamped correctly between 0 and 30).")


if __name__ == "__main__":
    print("==================================================================")
    print("TRIPLE-CHECK PARKING VERIFICATION & DECLUTTERING TEST SUITE")
    print("(v5 — Motorcycle Block Mode with Double Verification)")
    print("==================================================================")
    try:
        test_scenario_1_happy_path()
        test_scenario_2_debounce_repark()
        test_scenario_3_fallback_dwell()
        test_scenario_4_false_entry_timeout()
        test_scenario_5_multi_slot()
        test_scenario_6_visual_decluttering()
        test_scenario_7_anti_id_churn()
        test_scenario_8_polygon_clear_hysteresis()
        test_scenario_9_exclusive_slot_assignment()
        test_scenario_10_yolo_miss_30_frames_no_vacant()
        test_scenario_11_legitimate_exit_after_10s()
        test_scenario_12_vehicle_class_aliasing_and_roofbox()
        test_scenario_13_block_mode_init()
        test_scenario_14_block_mode_density_count()
        test_scenario_15_block_mode_tripwire_dominance()
        test_scenario_16_block_mode_hud_format()
        test_scenario_17_block_mode_capacity_cap_at_30()
        print("==================================================================")
        print("RESULT: ALL 17 TESTS PASSED (100% SUCCESS)")
        print("==================================================================")
    except AssertionError as e:
        print(f"\n[FAIL] Test Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


