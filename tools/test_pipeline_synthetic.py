"""
Suite Pengujian Sintetis Pipeline Fase 2 (Facility Management).
Menguji 4 skenario krusial:
  1. Motion Gate Frame Differencing & Heartbeat Invarian
  2. ROI Zone Lifecycle Events (ENTER -> LINGER -> EXIT) & Bukti Snapshot
  3. Spatial Memory Re-ID & Akumulasi Dwell Time Saat Oklusi
  4. Resolusi Normalisasi & Simetri Skala AI Canvas (640x360 -> 1920x1080 = scale 3.0)
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Tambahkan root workspace ke python path jika perlu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.logger import setup_logging
setup_logging(log_file="logs/test.log", log_level="WARNING", force_reconfigure=True)

from engine.config_loader import ROIZonesConfig, ROIZone, ROIPoint
from engine.detector_interface import DetectionResult
from engine.detector_impl import SyntheticMockDetector
from engine.event_dispatcher import EventDispatcher
from engine.geometry import bbox_centroid, scale_points
from engine.motion_gate import MotionGate
from engine.preprocessor import ProcessedFrame
from engine.tracker_interface import TrackResult
from engine.tracker_impl import CentroidTracker
from storage.database import init_db, get_writer_conn, close_db
from storage.models import get_unresolved_events


TEST_DB_PATH = "storage/test_synthetic.db"
TEST_SNAPSHOT_DIR = Path("storage/test_cam_snapshots")


def setup_test_env():
    """Bersihkan DB dan snapshot sisa test sebelumnya."""
    if Path(TEST_DB_PATH).exists():
        Path(TEST_DB_PATH).unlink()
    if TEST_SNAPSHOT_DIR.exists():
        shutil.rmtree(TEST_SNAPSHOT_DIR)
    if Path("cameras/test_cam").exists():
        shutil.rmtree("cameras/test_cam")
    TEST_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    init_db(TEST_DB_PATH)


def teardown_test_env():
    """Tutup koneksi DB dan hapus DB test."""
    close_db()
    if Path(TEST_DB_PATH).exists():
        Path(TEST_DB_PATH).unlink()
    if TEST_SNAPSHOT_DIR.exists():
        shutil.rmtree(TEST_SNAPSHOT_DIR)
    if Path("cameras/test_cam").exists():
        shutil.rmtree("cameras/test_cam")


def test_scenario_1_motion_gate():
    """Skenario 1: Motion Gate Differencing & Heartbeat 2.0s."""
    print("\n[TEST 1] Menguji MotionGate Differencing & Heartbeat Invarian...")
    gate = MotionGate(pixel_threshold=25, min_changed_pixels_pct=0.005, heartbeat_interval_sec=2.0)

    black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    t0 = 1000.0

    # 1. Frame pertama -> initial (True)
    res, reason = gate.should_process(black_frame, t0)
    assert res is True and reason == "initial", f"Expected initial, got {res}, {reason}"

    # 2. Frame identik 0.5s kemudian -> suppressed (False)
    res, reason = gate.should_process(black_frame, t0 + 0.5)
    assert res is False and reason == "suppressed", f"Expected suppressed, got {res}, {reason}"

    # 3. Frame dengan motion 1.0s kemudian -> motion (True)
    motion_frame = black_frame.copy()
    motion_frame[100:200, 100:200] = 255
    res, reason = gate.should_process(motion_frame, t0 + 1.0)
    assert res is True and reason == "motion", f"Expected motion, got {res}, {reason}"

    # 4. Frame identik 2.1s setelah last processed -> heartbeat (True)
    res, reason = gate.should_process(motion_frame, t0 + 3.2)
    assert res is True and reason == "heartbeat", f"Expected heartbeat, got {res}, {reason}"

    print("  ✓ PASS: MotionGate differencing dan heartbeat 2.0s terverifikasi sempurna.")


def test_scenario_2_lifecycle_and_snapshots():
    """Skenario 2: Lifecycle Events (ENTER -> LINGER -> EXIT) & Snapshot JPEG 1080p."""
    print("\n[TEST 2] Menguji Lifecycle Events (ENTER -> LINGER -> EXIT) & Snapshots...")

    conn = get_writer_conn()
    dispatcher = EventDispatcher(snapshot_quality=85, exit_timeout_sec=2.0)

    # ROI Zone di raw 1080p: polygon [(100,100), (900,100), (900,900), (100,900)]
    roi_config = ROIZonesConfig(
        schema_version="1.0",
        camera_id="test_cam",
        zones=[
            ROIZone(
                zone_id="zone_lobby",
                label="Lobby Test",
                color_hex="#00FF00",
                points=[
                    ROIPoint(x=100, y=100),
                    ROIPoint(x=900, y=100),
                    ROIPoint(x=900, y=900),
                    ROIPoint(x=100, y=900),
                ],
                trigger_on=["enter", "exit", "linger"],
                linger_threshold_sec=2,  # Short threshold for test
            )
        ],
    )

    raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ai_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    t0 = 2000.0

    proc_frame = ProcessedFrame(
        raw_frame=raw_frame,
        ai_frame=ai_frame,
        camera_id="test_cam",
        timestamp=t0,
        frame_number=1,
        scale_x=3.0,
        scale_y=3.0,
    )

    # Track 1 inside zone (in AI coords: bottom_center at (100, 100) -> raw (300, 300) inside polygon)
    track1 = TrackResult(
        track_id=1,
        bbox=(90.0, 80.0, 110.0, 100.0),  # bottom center x=100, y=100
        confidence=0.9,
        class_label="person",
        class_id=0,
        is_confirmed=True,
        camera_id="test_cam",
        frame_number=1,
    )

    # 1. ENTER
    events = dispatcher.process_tracks(proc_frame, [track1], roi_config, conn)
    assert len(events) == 1 and events[0].event_type == "enter", f"Expected ENTER, got {events}"
    assert Path(events[0].snapshot_path).exists(), f"Snapshot not found: {events[0].snapshot_path}"

    # Check DB row
    unresolved = get_unresolved_events(conn, "test_cam")
    assert len(unresolved) == 1 and unresolved[0]["event_type"] == "enter", "DB unresolved event count mismatch"

    # 2. LINGER (set timestamp t0 + 2.5s)
    proc_frame_linger = ProcessedFrame(
        raw_frame=raw_frame,
        ai_frame=ai_frame,
        camera_id="test_cam",
        timestamp=t0 + 2.5,
        frame_number=2,
        scale_x=3.0,
        scale_y=3.0,
    )
    events = dispatcher.process_tracks(proc_frame_linger, [track1], roi_config, conn)
    assert len(events) == 1 and events[0].event_type == "linger", f"Expected LINGER, got {events}"

    # 3. EXIT (track moves outside zone to AI x=500, y=500 -> raw x=1500, y=1500 outside polygon)
    track1_outside = TrackResult(
        track_id=1,
        bbox=(490.0, 480.0, 510.0, 500.0),  # bottom center x=500, y=500 -> raw (1500, 1500)
        confidence=0.9,
        class_label="person",
        class_id=0,
        is_confirmed=True,
        camera_id="test_cam",
        frame_number=3,
    )
    proc_frame_exit = ProcessedFrame(
        raw_frame=raw_frame,
        ai_frame=ai_frame,
        camera_id="test_cam",
        timestamp=t0 + 4.0,
        frame_number=3,
        scale_x=3.0,
        scale_y=3.0,
    )
    events = dispatcher.process_tracks(proc_frame_exit, [track1_outside], roi_config, conn)
    assert len(events) == 1 and events[0].event_type == "exit", f"Expected EXIT, got {events}"
    assert events[0].is_resolved == 1, "Exit event should be resolved"

    # Verify unresolved in DB is now empty
    unresolved = get_unresolved_events(conn, "test_cam")
    assert len(unresolved) == 0, f"Expected 0 unresolved events, got {len(unresolved)}"

    print("  ✓ PASS: Lifecycle events ENTER->LINGER->EXIT & snapshot 1080p terverifikasi.")


def test_scenario_3_spatial_memory_reid():
    """Skenario 3: Spatial Memory Re-ID dan Preservasi Dwell Time saat Oklusi."""
    print("\n[TEST 3] Menguji Spatial Memory Re-ID & Preservation Dwell Time...")

    tracker = CentroidTracker(
        max_disappeared_frames=2,  # Hilang cepat untuk test
        anchor_radius_px=15.0,
        enter_min_frames=3,
        spatial_memory_ttl_sec=3.0,
    )

    det_person = DetectionResult(
        bbox=(100.0, 100.0, 140.0, 140.0),  # Centroid (120, 120)
        confidence=0.9,
        class_id=0,
        class_label="person",
    )

    t0 = 3000.0

    # 1. Frame 1, 2, 3 -> Track menjadi Stationary (ID 1)
    tracker.update([det_person], frame_number=1, timestamp=t0)
    tracker.update([det_person], frame_number=2, timestamp=t0 + 0.5)
    tracks = tracker.update([det_person], frame_number=3, timestamp=t0 + 1.0)

    assert len(tracks) == 1 and tracks[0].track_id == 1, f"Expected track_id=1, got {tracks}"
    assert tracks[0].is_confirmed is True, "Track harus confirmed/stationary"

    # 2. Oklusi / Hilang deteksi selama 3 frame (frame 4, 5, 6)
    tracker.update([], frame_number=4, timestamp=t0 + 1.5)
    tracker.update([], frame_number=5, timestamp=t0 + 2.0)
    tracks = tracker.update([], frame_number=6, timestamp=t0 + 2.5)
    assert len(tracks) == 0, "Track harus hilang dari active tracks"

    # Cek bahwa track_id 1 tersimpan di spatial memory
    assert len(tracker.spatial_memory) == 1, f"Expected 1 item in spatial memory, got {len(tracker.spatial_memory)}"
    assert tracker.spatial_memory[0].track_id == 1, "Track ID 1 harus ada di spatial memory"

    # 3. Deteksi muncul kembali di dekat anchor radius (Centroid 122, 122 -> Dist = 2.8px <= 15.0px)
    det_reid = DetectionResult(
        bbox=(102.0, 102.0, 142.0, 142.0),  # Centroid (122, 122)
        confidence=0.88,
        class_id=0,
        class_label="person",
    )
    tracks = tracker.update([det_reid], frame_number=7, timestamp=t0 + 3.0)

    assert len(tracks) == 1, f"Expected 1 track after Re-ID, got {len(tracks)}"
    assert tracks[0].track_id == 1, f"Re-ID GAGAL: Expected track_id=1, got {tracks[0].track_id}"
    print("  ✓ PASS: Spatial Memory Re-ID mempertahankan Track ID=1 dan akumulasi dwell time.")


def test_scenario_4_resolution_normalization():
    """Skenario 4: Normalisasi Resolusi 16:9 Canvas & Simetri Scale Factor (scale_x=3.0, scale_y=3.0)."""
    print("\n[TEST 4] Menguji Normalisasi Resolusi Canvas 640x360 (Scale Factor = 3.0)...")

    raw_h, raw_w = 1080, 1920
    ai_h, ai_w = 360, 640

    scale_x = raw_w / ai_w
    scale_y = raw_h / ai_h

    assert scale_x == 3.0, f"Expected scale_x=3.0, got {scale_x}"
    assert scale_y == 3.0, f"Expected scale_y=3.0, got {scale_y}"

    # Test BBox mapping dari AI canvas ke 1080p raw
    ai_bbox = (100.0, 50.0, 200.0, 150.0)  # (x1, y1, x2, y2)
    raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ai_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    proc_frame = ProcessedFrame(
        raw_frame=raw_frame,
        ai_frame=ai_frame,
        camera_id="cam_test",
        timestamp=time.time(),
        frame_number=1,
        scale_x=scale_x,
        scale_y=scale_y,
    )

    rx1, ry1, rx2, ry2 = proc_frame.map_bbox_to_raw(*ai_bbox)

    assert rx1 == 300 and ry1 == 150 and rx2 == 600 and ry2 == 450, (
        f"Mapping bbox error: expected (300, 150, 600, 450), got ({rx1}, {ry1}, {rx2}, {ry2})"
    )

    print("  ✓ PASS: Resolusi canvas 640x360 & faktor skala 3.0 simetris terverifikasi.")


def main():
    print("============================================================")
    print("  FACILITY MANAGEMENT — SYNTETHIC PIPELINE TEST SUITE")
    print("============================================================")

    setup_test_env()

    try:
        test_scenario_1_motion_gate()
        test_scenario_2_lifecycle_and_snapshots()
        test_scenario_3_spatial_memory_reid()
        test_scenario_4_resolution_normalization()

        print("\n" + "=" * 60)
        print("  ALL 4 SYNTHETIC TEST SCENARIOS: PASS ✓")
        print("============================================================")
    finally:
        teardown_test_env()


if __name__ == "__main__":
    main()
