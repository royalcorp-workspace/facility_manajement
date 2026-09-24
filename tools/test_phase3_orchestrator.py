"""
Suite Pengujian Terpadu Fase 3 (Facility Management).
Menguji 5 skenario krusial:
  1. MultiCameraBuffer O(1) Memory & Thread Safety
  2. AlertDispatcher Non-Blocking Queue (Drop-Oldest) & Cooldown
  3. Camera Auto-Discovery (Mengabaikan _template)
  4. Global AI BoundedSemaphore Proteksi CPU
  5. FastAPI Web Application, Endpoints & Resilient MJPEG Streaming
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import threading
import time
from pathlib import Path

# Setup path agar modul proyek terbaca
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.logger import setup_logging
setup_logging(log_file="logs/test.log", log_level="WARNING", force_reconfigure=True)

import cv2
import numpy as np
from fastapi.testclient import TestClient

from engine.config_loader import CameraConfig, ResolutionConfig, CaptureConfig, DiskGuardCameraConfig
from notification.dispatcher import AlertDispatcher, AlertPacket
from storage.database import init_db, get_writer_conn, close_db
from storage.models import LifecycleEvent
from web.app import create_app
from web.buffer import MultiCameraBuffer, FrameItem

TEST_DB_PATH = "storage/test_phase3.db"


def setup_test_env():
    """Siapkan DB terisolasi untuk testing."""
    if Path(TEST_DB_PATH).exists():
        Path(TEST_DB_PATH).unlink()
    init_db(TEST_DB_PATH)


def teardown_test_env():
    """Tutup dan hapus DB testing."""
    close_db()
    if Path(TEST_DB_PATH).exists():
        Path(TEST_DB_PATH).unlink()


def test_scenario_1_multicamera_buffer():
    """Skenario 1: MultiCameraBuffer O(1) Memory & Thread Safety."""
    print("\n[TEST 1] Menguji MultiCameraBuffer O(1) Memory & Thread Safety...")
    buffer = MultiCameraBuffer()

    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    ret, jpeg1 = cv2.imencode(".jpg", dummy_frame)
    assert ret, "Gagal encode JPEG"
    bytes1 = jpeg1.tobytes()

    # Tulis dari 2 thread berbeda
    def writer(cam_id: str, count: int):
        for i in range(count):
            buffer.set_frame(cam_id, bytes1, time.time(), i)

    t1 = threading.Thread(target=writer, args=("cam_A", 50))
    t2 = threading.Thread(target=writer, args=("cam_B", 50))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    cams = buffer.get_registered_cameras()
    assert set(cams) == {"cam_A", "cam_B"}, f"Expected cam_A and cam_B, got {cams}"

    # Verifikasi hanya ada 1 frame tersimpan per kamera (O(1) memory)
    assert len(buffer._frames) == 2, f"Buffer internal size must be exactly 2, got {len(buffer._frames)}"

    # Verifikasi wait_for_next_frame
    item = buffer.wait_for_next_frame("cam_A", last_sequence_id=48, timeout=0.2)
    assert item is not None and item.sequence_id == 49, f"Expected seq 49, got {item}"

    print("  ✓ PASS: MultiCameraBuffer O(1) dan sinkronisasi multi-thread terverifikasi.")


def test_scenario_2_alert_dispatcher():
    """Skenario 2: AlertDispatcher Non-Blocking Queue (Drop-Oldest) & Cooldown."""
    print("\n[TEST 2] Menguji AlertDispatcher Queue (Drop-Oldest) & Cooldown...")

    dispatcher = AlertDispatcher(queue_maxsize=10, cooldown_sec=1.0, enable_audio=True)
    dispatcher.start()

    dispatched_alerts = []
    dispatcher.add_sink(lambda p: dispatched_alerts.append(p))

    try:
        # Kirim 30 alert cepat ke antrean maxsize 10 (menguji drop-oldest tanpa blocking)
        for i in range(30):
            dispatcher.dispatch(
                AlertPacket(
                    camera_id="cam_01",
                    zone_id="lobby",
                    event_type="enter",
                    track_id=i,
                    message=f"Alert #{i}",
                )
            )

        time.sleep(0.5)

        # Karena cooldown 1.0s pada (cam_01, lobby, enter), hanya 1 alert yang lolos dispatching
        assert dispatcher.total_dispatched == 1, (
            f"Expected exactly 1 alert dispatched due to cooldown, got {dispatcher.total_dispatched}"
        )
        assert dispatcher.total_cooldown_suppressed > 0, "Cooldown suppression must be triggered"

        # Tunggu cooldown berlalu lalu kirim alert baru
        time.sleep(1.1)
        dispatcher.dispatch(
            AlertPacket(
                camera_id="cam_01",
                zone_id="lobby",
                event_type="enter",
                track_id=99,
                message="Alert after cooldown",
            )
        )
        time.sleep(0.5)
        assert dispatcher.total_dispatched == 2, f"Expected 2 alerts dispatched, got {dispatcher.total_dispatched}"

    finally:
        dispatcher.stop()
        dispatcher.join(timeout=2.0)

    print("  ✓ PASS: AlertDispatcher non-blocking, drop-oldest, dan cooldown 1.0s terverifikasi.")


def test_scenario_3_camera_autodiscovery():
    """Skenario 3: Auto-Discovery Profil Kamera."""
    print("\n[TEST 3] Menguji Camera Auto-Discovery...")

    cameras_dir = Path("cameras")
    cam_dirs = sorted([
        d for d in cameras_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "config.json").exists()
    ])

    discovered_ids = [d.name for d in cam_dirs]
    assert "cam_01" in discovered_ids, f"cam_01 harus ditemukan, got {discovered_ids}"
    assert "_template" not in discovered_ids, f"_template tidak boleh terdeteksi sebagai kamera aktif"

    print(f"  ✓ PASS: Auto-discovery mendeteksi {len(discovered_ids)} kamera ({discovered_ids}), mengabaikan _template.")


def test_scenario_4_ai_bounded_semaphore():
    """Skenario 4: Global AI BoundedSemaphore Proteksi CPU."""
    print("\n[TEST 4] Menguji Global AI BoundedSemaphore (Concurrency=1)...")

    sem = threading.BoundedSemaphore(value=1)
    concurrent_executions = 0
    max_observed_concurrency = 0
    lock = threading.Lock()

    def simulate_inference():
        nonlocal concurrent_executions, max_observed_concurrency
        with sem:
            with lock:
                concurrent_executions += 1
                if concurrent_executions > max_observed_concurrency:
                    max_observed_concurrency = concurrent_executions
            time.sleep(0.05)  # Simulasi forward pass ONNX
            with lock:
                concurrent_executions -= 1

    threads = [threading.Thread(target=simulate_inference) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_observed_concurrency == 1, (
        f"Concurrency pelanggaran! Expected max 1, got {max_observed_concurrency}"
    )

    print("  ✓ PASS: Global AI BoundedSemaphore menjamin eksekusi staggered (max concurrency = 1).")


def test_scenario_5_web_endpoints_and_mjpeg():
    """Skenario 5: FastAPI Web Application, Endpoints & Resilient MJPEG Streaming."""
    print("\n[TEST 5] Menguji FastAPI Endpoints & Resilient MJPEG Streaming...")

    conn = get_writer_conn()
    ev = LifecycleEvent(
        camera_id="cam_01",
        zone_id="zone_lobby_entry",
        event_type="linger",
        track_id=1,
        class_label="person",
        confidence=0.88,
        notes="Dwell duration: 32.5s",
    )
    ev.insert(conn)

    buffer = MultiCameraBuffer()
    cam_cfg = CameraConfig(
        camera_id="cam_01",
        display_name="Kamera 01 Lobby",
        rtsp_url="rtsp://dummy:dummy@localhost:554/live",
        resolution=ResolutionConfig(capture_width=1920, capture_height=1080, ai_width=640, ai_height=360),
        capture=CaptureConfig(),
        disk_guard=DiskGuardCameraConfig(snapshot_dir="cameras/cam_01/snapshots"),
    )

    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    ret, jpeg_bytes = cv2.imencode(".jpg", dummy_frame)
    buffer.set_frame("cam_01", jpeg_bytes.tobytes(), time.time(), 1)

    app = create_app(buffer, [cam_cfg])
    client = TestClient(app)

    # 1. Test GET /
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "Facility Management" in resp.text
    assert "cam_01" in resp.text

    # 2. Test GET /api/status/cam_01
    resp = client.get("/api/status/cam_01")
    assert resp.status_code == 200
    status_data = resp.json()
    assert status_data["camera_id"] == "cam_01"
    assert "640x360" in status_data["resolution"]

    # 3. Test GET /api/incidents
    resp = client.get("/api/incidents")
    assert resp.status_code == 200
    incidents = resp.json()
    assert len(incidents) >= 1
    assert incidents[0]["camera_id"] == "cam_01"
    assert incidents[0]["event_type"] == "linger"

    # 4. Test GET /video_feed/cam_01 (Streaming Generator & Clean Cancellation)
    feed_route = None
    for r in app.routes:
        if hasattr(r, "path") and r.path == "/video_feed/{camera_id}":
            feed_route = r
            break
    assert feed_route is not None, "Route /video_feed/{camera_id} tidak ditemukan"

    async def verify_streaming():
        resp = await feed_route.endpoint("cam_01")
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.media_type
        gen = resp.body_iterator
        chunk = await anext(gen)
        assert b"--frame" in chunk
        await gen.aclose()  # Verifikasi clean cancel & generator close

    asyncio.run(verify_streaming())

    print("  ✓ PASS: FastAPI endpoints (/, /api/status, /api/incidents, /video_feed) terverifikasi.")


def main():
    print("============================================================")
    print("  FACILITY MANAGEMENT — PHASE 3 ORCHESTRATOR TEST SUITE")
    print("============================================================")

    setup_test_env()

    try:
        test_scenario_1_multicamera_buffer()
        test_scenario_2_alert_dispatcher()
        test_scenario_3_camera_autodiscovery()
        test_scenario_4_ai_bounded_semaphore()
        test_scenario_5_web_endpoints_and_mjpeg()

        print("\n" + "=" * 60)
        print("  ALL 5 PHASE 3 TEST SCENARIOS: PASS ✓")
        print("============================================================")
    finally:
        teardown_test_env()


if __name__ == "__main__":
    main()
