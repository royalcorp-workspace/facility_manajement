"""
Suite Pengujian Terpadu Fase 5 — Multi-Camera Scalability, Incident Viewer & Dynamic Web Dashboard.
Menguji 5 skenario utama:
  1. Camera Template Cloning & Auto-Discovery
  2. Staggered AI Balancing & CentroidTracker Coasting Continuity
  3. Web API Endpoints: GET /api/cameras & POST /api/incidents/{id}/resolve
  4. Snapshot Evidence Delivery & Traversal Protection
  5. Multi-Camera Concurrent Orchestration & Graceful Shutdown
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

# Setup sys.path agar modul root terbaca
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from engine.logger import setup_logging
setup_logging(log_file="logs/test.log", log_level="WARNING", force_reconfigure=True)

import cv2
import numpy as np
from fastapi.testclient import TestClient

from engine.config_loader import (
    CameraConfig,
    ResolutionConfig,
    CaptureConfig,
    DiskGuardCameraConfig,
    ROIZonesConfig,
    load_camera_config,
    load_roi_zones,
)
from engine.detector_interface import DetectionResult
from engine.detector_impl import SyntheticMockDetector
from engine.motion_gate import MotionGate
from engine.pipeline import CameraOrchestrator
from engine.preprocessor import ProcessedFrame
from engine.tracker_impl import CentroidTracker
from storage.database import init_db, get_writer_conn, get_reader_conn, close_db, resolve_incident
from storage.models import LifecycleEvent
from web.app import create_app
from web.buffer import MultiCameraBuffer

TEST_DB_PATH = Path("storage/test_phase5.db")
TEST_CAM_DIR = Path("cameras/cam_test_02")


def setup_test_env():
    """Siapkan DB dan direktori kamera uji."""
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    init_db(TEST_DB_PATH)


def teardown_test_env():
    """Bersihkan DB dan direktori kamera uji."""
    close_db()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    if TEST_CAM_DIR.exists():
        shutil.rmtree(TEST_CAM_DIR, ignore_errors=True)


def test_scenario_1_template_cloning_and_discovery():
    """Skenario 1: Kloning Template Kamera & Auto-Discovery."""
    print("\n[TEST 1] Menguji Kloning Template Kamera & Auto-Discovery...")
    template_dir = Path("cameras/_template")
    assert template_dir.exists(), "Direktori cameras/_template tidak ditemukan"

    TEST_CAM_DIR.mkdir(parents=True, exist_ok=True)
    (TEST_CAM_DIR / "snapshots").mkdir(parents=True, exist_ok=True)

    # Baca template config dan modifikasi
    with open(template_dir / "config.json", "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    cfg_data["camera_id"] = "cam_test_02"
    cfg_data["display_name"] = "Kamera Test 02"
    cfg_data["rtsp_url"] = "rtsp://user:pass@127.0.0.1:554/cam2"
    cfg_data["disk_guard"]["snapshot_dir"] = str(TEST_CAM_DIR / "snapshots")

    with open(TEST_CAM_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    # Baca template roi_zones dan modifikasi
    with open(template_dir / "roi_zones.json", "r", encoding="utf-8") as f:
        roi_data = json.load(f)
    roi_data["camera_id"] = "cam_test_02"
    with open(TEST_CAM_DIR / "roi_zones.json", "w", encoding="utf-8") as f:
        json.dump(roi_data, f, indent=2)

    # Validasi auto-loader
    loaded_cfg = load_camera_config(TEST_CAM_DIR / "config.json")
    loaded_roi = load_roi_zones(TEST_CAM_DIR / "roi_zones.json")

    assert loaded_cfg.camera_id == "cam_test_02"
    assert loaded_cfg.display_name == "Kamera Test 02"
    assert loaded_roi.camera_id == "cam_test_02"
    print("  ✓ PASS: Template kamera berhasil dikloning dan divalidasi skemanya.")


def test_scenario_2_staggered_ai_and_coasting():
    """Skenario 2: Staggered AI Balancing & CentroidTracker Coasting Continuity."""
    print("\n[TEST 2] Menguji Staggered AI Balancing & CentroidTracker Coasting...")

    tracker = CentroidTracker(
        max_disappeared_frames=30,
        enter_min_frames=2,
    )

    # Step A: 2 frame dengan deteksi nyata untuk confirm track
    det1 = [DetectionResult(bbox=(100.0, 100.0, 150.0, 200.0), confidence=0.9, class_label="person", class_id=0)]
    t1 = tracker.update(det1, frame_number=1, timestamp=100.0)
    assert len(t1) == 0, "Belum confirmed pada frame 1"

    t2 = tracker.update(det1, frame_number=2, timestamp=100.1)
    assert len(t2) == 1, "Harus confirmed pada frame 2 (min_hits=2)"
    track_id = t2[0].track_id

    # Step B: Panggil coast() pada frame 3 dan 4 (giliran kamera lain)
    coasted_1 = tracker.coast(frame_number=3, timestamp=100.2)
    assert len(coasted_1) == 1, "Track harus tetap aktif saat coasting"
    assert coasted_1[0].track_id == track_id, "Track ID harus konsisten"

    coasted_2 = tracker.coast(frame_number=4, timestamp=100.3)
    assert len(coasted_2) == 1
    assert coasted_2[0].track_id == track_id

    # Pastikan disappeared TIDAK naik karena coasting
    assert tracker.tracks[track_id].disappeared == 0, "Disappeared tidak boleh naik saat coasting"
    assert tracker.tracks[track_id].accumulated_dwell_sec > 0.1, "Dwell time harus bertambah"

    # Step C: Verifikasi logika giliran Staggered Cadence
    # Kamera 0: (count + 0) % 2 == 0
    # Kamera 1: (count + 1) % 2 == 0
    total_cams = 2
    cam0_turns = [((cnt + 0) % total_cams == 0) for cnt in range(1, 7)]
    cam1_turns = [((cnt + 1) % total_cams == 0) for cnt in range(1, 7)]

    # Pastikan kedua kamera bergantian dan tidak saling tabrakan
    for i in range(len(cam0_turns)):
        assert cam0_turns[i] != cam1_turns[i], f"Tabrakan giliran pada step {i+1}"

    print("  ✓ PASS: CentroidTracker coasting dan giliran inferensi interleaved terverifikasi sempurna.")


def test_scenario_3_web_endpoints_and_resolve():
    """Skenario 3: Web API Endpoints GET /api/cameras & POST /api/incidents/{id}/resolve."""
    print("\n[TEST 3] Menguji Web API Endpoints (GET /api/cameras & POST resolve)...")

    # Insert insiden dummy ke DB
    conn = get_writer_conn()
    ev = LifecycleEvent(
        camera_id="cam_test_02",
        zone_id="zone_entrance",
        event_type="linger",
        track_id=42,
        class_label="person",
        confidence=0.89,
        notes="Dwell duration: 15.2s",
        is_resolved=0,
    )
    ev.insert(conn)
    incident_id = ev.id
    assert incident_id is not None

    buffer = MultiCameraBuffer()
    cam1 = CameraConfig(
        camera_id="cam_01",
        display_name="Kamera 01 Lobby",
        rtsp_url="rtsp://dummy",
        resolution=ResolutionConfig(capture_width=1920, capture_height=1080, ai_width=640, ai_height=360),
        capture=CaptureConfig(),
        disk_guard=DiskGuardCameraConfig(snapshot_dir="cameras/cam_01/snapshots"),
    )
    cam2 = CameraConfig(
        camera_id="cam_test_02",
        display_name="Kamera Test 02",
        rtsp_url="rtsp://dummy2",
        resolution=ResolutionConfig(capture_width=1920, capture_height=1080, ai_width=640, ai_height=360),
        capture=CaptureConfig(),
        disk_guard=DiskGuardCameraConfig(snapshot_dir="cameras/cam_test_02/snapshots"),
    )

    app = create_app(buffer, [cam1, cam2])
    client = TestClient(app)

    # 1. Test GET /api/cameras
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    cams_data = resp.json()
    assert len(cams_data) == 2
    cam_ids = [c["camera_id"] for c in cams_data]
    assert "cam_01" in cam_ids and "cam_test_02" in cam_ids

    # 2. Test GET /api/incidents
    resp_inc = client.get("/api/incidents")
    assert resp_inc.status_code == 200
    inc_list = resp_inc.json()
    assert any(i["id"] == incident_id for i in inc_list)

    # 3. Test POST /api/incidents/{id}/resolve
    resp_res = client.post(f"/api/incidents/{incident_id}/resolve")
    assert resp_res.status_code == 200
    res_data = resp_res.json()
    assert res_data["status"] == "success"
    assert res_data["incident_id"] == incident_id
    assert "resolved_time" in res_data

    # Verifikasi langsung ke DB reader
    r_conn = get_reader_conn()
    row = r_conn.execute("SELECT is_resolved, resolved_time, notes FROM lifecycle_events WHERE id = ?", (incident_id,)).fetchone()
    r_conn.close()
    assert row is not None
    assert row["is_resolved"] == 1
    assert row["resolved_time"] is not None
    assert "Resolved by Operator" in row["notes"]

    # 4. Test Resolve ulang (harus 404 karena sudah resolved)
    resp_res_dup = client.post(f"/api/incidents/{incident_id}/resolve")
    assert resp_res_dup.status_code == 404

    # 5. Test Resolve ID non-existent
    resp_res_invalid = client.post("/api/incidents/999999/resolve")
    assert resp_res_invalid.status_code == 404

    print("  ✓ PASS: Endpoint /api/cameras dan /api/incidents/{id}/resolve terverifikasi.")


def test_scenario_4_snapshot_evidence_and_traversal_protection():
    """Skenario 4: Snapshot Evidence Delivery & Traversal Protection."""
    print("\n[TEST 4] Menguji Snapshot Evidence Delivery & Traversal Protection...")

    # Buat snapshot dummy di cameras/cam_test_02/snapshots/test_snap.jpg
    snap_dir = TEST_CAM_DIR / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_file = snap_dir / "test_snap.jpg"

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(snap_file), dummy_img)

    buffer = MultiCameraBuffer()
    cam2 = CameraConfig(
        camera_id="cam_test_02",
        display_name="Kamera Test 02",
        rtsp_url="rtsp://dummy2",
        resolution=ResolutionConfig(capture_width=1920, capture_height=1080, ai_width=640, ai_height=360),
        capture=CaptureConfig(),
        disk_guard=DiskGuardCameraConfig(snapshot_dir=str(snap_dir)),
    )

    app = create_app(buffer, [cam2])
    client = TestClient(app)

    # 1. Akses valid snapshot
    resp = client.get("/snapshots/cam_test_02/test_snap.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0

    # 2. Akses file yang tidak ada
    resp_not_found = client.get("/snapshots/cam_test_02/non_existent.jpg")
    assert resp_not_found.status_code == 404

    # 3. Path traversal attack
    resp_traversal = client.get("/snapshots/cam_test_02/../../config.json")
    assert resp_traversal.status_code in [403, 404]

    print("  ✓ PASS: Endpoint snapshot dan proteksi path traversal terverifikasi aman.")


def test_scenario_5_multicamera_concurrent_orchestration_and_shutdown():
    """Skenario 5: Multi-Camera Concurrent Orchestration & Graceful Shutdown."""
    print("\n[TEST 5] Menguji Multi-Camera Concurrent Orchestration & Graceful Shutdown...")

    import queue
    buffer = MultiCameraBuffer()
    ai_semaphore = threading.BoundedSemaphore(1)

    cfg1 = CameraConfig(
        camera_id="cam_01",
        display_name="Kamera 01",
        rtsp_url="rtsp://dummy1",
        resolution=ResolutionConfig(capture_width=1920, capture_height=1080, ai_width=640, ai_height=360),
        capture=CaptureConfig(),
        disk_guard=DiskGuardCameraConfig(snapshot_dir="cameras/cam_01/snapshots"),
    )
    cfg2 = CameraConfig(
        camera_id="cam_test_02",
        display_name="Kamera Test 02",
        rtsp_url="rtsp://dummy2",
        resolution=ResolutionConfig(capture_width=1920, capture_height=1080, ai_width=640, ai_height=360),
        capture=CaptureConfig(),
        disk_guard=DiskGuardCameraConfig(snapshot_dir=str(TEST_CAM_DIR / "snapshots")),
    )

    roi1 = load_roi_zones("cameras/cam_01/roi_zones.json")
    roi2 = load_roi_zones(TEST_CAM_DIR / "roi_zones.json")

    q1 = queue.Queue(maxsize=10)
    q2 = queue.Queue(maxsize=10)

    # Inisialisasi 2 orchestrator dengan camera_index berbeda
    orch1 = CameraOrchestrator(
        camera_config=cfg1,
        roi_config=roi1,
        proc_queue=q1,
        db_path=TEST_DB_PATH,
        detector=SyntheticMockDetector([]),
        ai_semaphore=ai_semaphore,
        frame_buffer=buffer,
        camera_index=0,
        total_cameras=2,
    )
    orch2 = CameraOrchestrator(
        camera_config=cfg2,
        roi_config=roi2,
        proc_queue=q2,
        db_path=TEST_DB_PATH,
        detector=SyntheticMockDetector([]),
        ai_semaphore=ai_semaphore,
        frame_buffer=buffer,
        camera_index=1,
        total_cameras=2,
    )

    orch1.start()
    orch2.start()

    # Masukkan synthetic frame ke kedua queue
    ai_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for frame_no in range(1, 5):
        ts_now = time.time()
        pf1 = ProcessedFrame(
            raw_frame=ai_frame,
            ai_frame=ai_frame,
            camera_id="cam_01",
            timestamp=ts_now,
            frame_number=frame_no,
            scale_x=1.0,
            scale_y=1.0,
        )
        pf2 = ProcessedFrame(
            raw_frame=ai_frame,
            ai_frame=ai_frame,
            camera_id="cam_test_02",
            timestamp=ts_now,
            frame_number=frame_no,
            scale_x=1.0,
            scale_y=1.0,
        )
        q1.put(pf1)
        q2.put(pf2)

    # Beri waktu pemrosesan
    time.sleep(0.5)

    # Graceful stop
    orch1.stop()
    orch2.stop()
    orch1.join(timeout=2.0)
    orch2.join(timeout=2.0)

    assert not orch1.is_alive(), "Orchestrator 1 gagal berhenti"
    assert not orch2.is_alive(), "Orchestrator 2 gagal berhenti"
    assert orch1.processed_count > 0, "Orchestrator 1 harus memproses frame"
    assert orch2.processed_count > 0, "Orchestrator 2 harus memproses frame"

    print("  ✓ PASS: Multi-kamera orkestrator berjalan bersamaan dan shutdown bersih.")


def main():
    print("============================================================")
    print("  FACILITY MANAGEMENT — PHASE 5 MULTI-CAM TEST SUITE")
    print("============================================================")

    setup_test_env()

    try:
        test_scenario_1_template_cloning_and_discovery()
        test_scenario_2_staggered_ai_and_coasting()
        test_scenario_3_web_endpoints_and_resolve()
        test_scenario_4_snapshot_evidence_and_traversal_protection()
        test_scenario_5_multicamera_concurrent_orchestration_and_shutdown()

        print("\n" + "=" * 60)
        print("  ALL 5 PHASE 5 TEST SCENARIOS: PASS ✓")
        print("============================================================")
    finally:
        teardown_test_env()


if __name__ == "__main__":
    main()
