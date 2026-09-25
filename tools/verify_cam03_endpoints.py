"""
tools/verify_cam03_endpoints.py
===============================
Audit & Verifikasi Ketersediaan & Routing Endpoint cam_03 pada FastAPI Web Hub.
Menguji:
1. MultiCameraBuffer registration & fallback canvas untuk cam_03
2. GET /video_feed/cam_03 streaming endpoint
3. GET /api/status/cam_03 JSON telemetry (FPS, active tracks, motorcycle_block parking)
4. GET /api/zones/cam_03 dan GET /api/zones?cam=cam_03 ROI configuration
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Set UTF-8 encoding untuk konsol Windows
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from engine.config_loader import load_camera_config, load_roi_zones
from engine.smart_parking import SmartParkingTracker
from web.app import create_app
from web.buffer import MultiCameraBuffer, generate_fallback_canvas


def run_audit():
    print("=" * 70)
    print(" AUDIT ENDPOINT & ROUTING cam_03 PADA FASTAPI WEB HUB")
    print("=" * 70)

    api_key = os.getenv("API_KEY", "facility-royal-2026")
    headers = {"X-API-Key": api_key}

    # 1. Setup konfigurasi kamera cam_03
    cam03_cfg_path = PROJECT_ROOT / "cameras" / "cam_03" / "config.json"
    cam03_roi_path = PROJECT_ROOT / "cameras" / "cam_03" / "roi_zones.json"

    assert cam03_cfg_path.exists(), "cameras/cam_03/config.json tidak ditemukan!"
    assert cam03_roi_path.exists(), "cameras/cam_03/roi_zones.json tidak ditemukan!"

    cam_cfg = load_camera_config(cam03_cfg_path)
    roi_cfg = load_roi_zones(cam03_roi_path)
    print(f"[OK] Konfigurasi cam_03 berhasil dimuat: {cam_cfg.display_name}")

    # 2. Verifikasi MultiCameraBuffer
    print("\n--- [1] AUDIT MULTI-CAMERA BUFFER ---")
    buffer = MultiCameraBuffer()
    buffer.register_camera("cam_03")

    registered = buffer.get_registered_cameras()
    assert "cam_03" in registered, f"cam_03 gagal terdaftar di buffer: {registered}"
    print(f"  [PASS] MultiCameraBuffer berhasil meregistrasi: {registered}")

    initial_frame = buffer.get_latest_frame("cam_03")
    assert initial_frame is not None, "Initial fallback frame cam_03 kosong!"
    assert len(initial_frame.jpeg_bytes) > 100, "Ukuran initial fallback frame terlalu kecil!"
    print(f"  [PASS] Fallback canvas cam_03 tersedia ({len(initial_frame.jpeg_bytes)} bytes)")

    # 3. Setup mock orchestrator dengan SmartParkingTracker motorcycle_block
    pt = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        dwell_threshold_sec=10.0,
        stationary_dwell_sec=0.0,
        block_exclusion_x_1080p=720,
    )
    # Simulasikan ada 8 motor terdeteksi dan ter-latch
    pt._last_stable_occupied = 8
    pt.gate_in_count = 12
    pt.gate_out_count = 4

    mock_orchestrator = MagicMock()
    mock_orchestrator.processed_count = 1520
    mock_orchestrator.motion_detected_count = 450
    mock_orchestrator.is_alive.return_value = True
    mock_orchestrator.fps = 19.8
    mock_orchestrator.tracker.tracks = {1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
    mock_orchestrator.parking_tracker = pt

    orchestrators = {"cam_03": mock_orchestrator}

    app = create_app(
        buffer=buffer,
        camera_configs=[cam_cfg],
        orchestrators=orchestrators,
    )
    client = TestClient(app)

    # 4. Verifikasi GET /video_feed/cam_03
    print("\n--- [2] AUDIT ENDPOINT STREAMING MJPEG ---")
    # 4a. Cek autentikasi (tanpa API Key -> 401)
    resp_unauth = client.get("/video_feed/cam_03")
    assert resp_unauth.status_code == 401, f"Expected 401, got {resp_unauth.status_code}"
    print("  [PASS] GET /video_feed/cam_03 menolak akses tanpa API Key (401 Unauthorized)")

    # 4b. Cek streaming generator
    import asyncio
    # Panggil route endpoint langsung
    route_handler = None
    for route in app.routes:
        if route.path == "/video_feed/{camera_id}":
            route_handler = route.endpoint
            break
    assert route_handler is not None, "Endpoint /video_feed/{camera_id} tidak terdaftar di FastAPI app!"

    # Uji StreamingResponse yang dihasilkan
    streaming_resp = asyncio.run(route_handler("cam_03"))
    assert streaming_resp.media_type == "multipart/x-mixed-replace; boundary=frame"

    # Verifikasi frame pertama dari generator
    async def get_first_frame():
        gen = streaming_resp.body_iterator
        first = await gen.__anext__()
        await gen.aclose()
        return first

    first_chunk = asyncio.run(get_first_frame())
    assert len(first_chunk) > 0, "Chunk stream kosong!"
    assert b"--frame" in first_chunk, "Boundary frame tidak ditemukan di chunk!"
    assert b"image/jpeg" in first_chunk, "Content-Type JPEG tidak ada di chunk!"
    print("  [PASS] GET /video_feed/cam_03 route terdaftar dan melayani feed multipart/x-mixed-replace secara atomic")

    # 5. Verifikasi GET /api/status/cam_03
    print("\n--- [3] AUDIT TELEMETRI STATUS cam_03 ---")
    resp_status = client.get("/api/status/cam_03", headers=headers)
    assert resp_status.status_code == 200, f"Expected 200, got {resp_status.status_code}"
    data = resp_status.json()

    print(f"  [PAYLOAD] camera_id: {data.get('camera_id')}")
    print(f"  [PAYLOAD] status: {data.get('status')}")
    print(f"  [PAYLOAD] fps: {data.get('fps')}")
    print(f"  [PAYLOAD] active_tracks: {data.get('active_tracks')}")
    print(f"  [PAYLOAD] resolution: {data.get('resolution')}")
    print(f"  [PAYLOAD] parking: {data.get('parking')}")

    assert data.get("camera_id") == "cam_03"
    assert data.get("status") == "active"
    assert data.get("fps") == 19.8
    assert data.get("active_tracks") == 3
    assert data.get("parking") is not None
    assert data["parking"].get("parking_mode") == "motorcycle_block"
    assert data["parking"].get("total_slots") == 30
    assert data["parking"].get("occupied_slots") == 8
    assert data["parking"].get("available_slots") == 22
    assert data["parking"].get("gate_in") == 12
    assert data["parking"].get("gate_out") == 4
    print("  [PASS] GET /api/status/cam_03 menyajikan metrik FPS, active tracks, dan status parkir blok motor secara akurat")

    # 6. Verifikasi GET /api/zones/cam_03 & GET /api/zones?cam=cam_03
    print("\n--- [4] AUDIT ENDPOINT ZONA ROI cam_03 ---")
    # 6a. GET /api/zones/cam_03
    resp_z1 = client.get("/api/zones/cam_03", headers=headers)
    assert resp_z1.status_code == 200
    z1_data = resp_z1.json()
    assert "polygons" in z1_data and len(z1_data["polygons"]) > 0
    print(f"  [PASS] GET /api/zones/cam_03 -> {len(z1_data['polygons'])} polygon(s), {len(z1_data.get('tripwires', []))} tripwire(s)")

    # 6b. GET /api/zones?cam=cam_03
    resp_z2 = client.get("/api/zones?cam=cam_03", headers=headers)
    assert resp_z2.status_code == 200
    z2_data = resp_z2.json()
    assert z2_data == z1_data
    print("  [PASS] GET /api/zones?cam=cam_03 -> Output identik dan terverifikasi")

    print("\n" + "=" * 70)
    print(" SEMUA AUDIT ROUTING & BUFFER cam_03 SELESAI & LULUS 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_audit()
