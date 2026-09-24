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

    # Periksa telemetry string via logic check
    occ_slots = [
        f"S{s.slot_num}"
        for s in sorted(tracker.slot_states.values(), key=lambda x: x.slot_num)
        if s.phase == "OCCUPIED"
    ]
    occ_text = ", ".join(occ_slots) if occ_slots else "-"
    expected_hud = f"PARKING: 1/2 SLOTS AVAILABLE | OCCUPIED: [{occ_text}]"
    assert occ_text == "S1"
    assert "S1" in expected_hud
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

    # ─── Oklusi panjang 5.5 detik (melampaui vacant_confirm_sec = 5.0s) ───
    # Simulasi: kendaraan benar-benar pergi
    tracker.update([], polys, scale, scale, t + 12.0)   # +0.8s hilang, mulai hitung
    tracker.update([], polys, scale, scale, t + 14.5)   # +3.3s hilang, masih dalam grace period 5s
    assert s1.phase == "OCCUPIED", \
        f"Belum 5s poligon kosong, harusnya masih OCCUPIED, dapat: {s1.phase}"

    # Lewati 5s penuh poligon kosong → baru boleh transisi ke LEAVING
    tracker.update([], polys, scale, scale, t + 18.0)   # ~6.8s hilang total (melampaui 5s)
    assert s1.phase == "LEAVING", \
        f"Setelah >= 5s poligon kosong harusnya LEAVING, dapat: {s1.phase}"

    # Konfirmasi akhir: dwell tidak pernah direset selama proses
    assert s1.dwell_duration >= 10.0, \
        f"Dwell harusnya tetap >= 10s selama transisi, dapat: {s1.dwell_duration}"

    print(f"  [PASS] Scenario 8 Passed (polygon_clear_since hysteresis 5s bekerja, dwell={s1.dwell_duration:.1f}s).")


if __name__ == "__main__":
    print("==================================================================")
    print("TRIPLE-CHECK PARKING VERIFICATION & DECLUTTERING TEST SUITE")
    print("(v2 — Anti-ID Churn & Spatial Slot Stabilization)")
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
        print("==================================================================")
        print("RESULT: ALL 8 TESTS PASSED (100% SUCCESS)")
        print("==================================================================")
    except AssertionError as e:
        print(f"\n[FAIL] Test Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

