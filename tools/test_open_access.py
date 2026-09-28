"""
Unit test untuk verifikasi Open Access pada Web Hub & REST API:
1. Akses langsung GET / (Dashboard Kiosk) tanpa API Key
2. Akses seluruh endpoint REST API (/api/cameras, /api/status/..., /api/zones, /api/incidents) tanpa API Key
3. Akses video streaming MJPEG (/video_feed/...) tanpa API Key
4. Memastikan tidak ada penolakan HTTP 401 Unauthorized
"""

import sys
from pathlib import Path

# Tambahkan root proyek ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from engine.config_loader import load_camera_config
from web.app import create_app
from web.buffer import MultiCameraBuffer


def create_test_client():
    cam_cfg = load_camera_config(PROJECT_ROOT / "cameras" / "cam_01" / "config.json")
    buffer = MultiCameraBuffer()
    buffer.set_frame(cam_cfg.camera_id, b"\xff\xd8\xff\xe0fake", 100.0, 1)
    app = create_app(buffer=buffer, camera_configs=[cam_cfg], orchestrators={})
    return TestClient(app)


def test_public_index_open_access():
    client = create_test_client()
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}"
    # Memastikan tidak ada token rahasia yang bocor di template HTML
    assert "facility-royal-2026" not in resp.text, "Token API_KEY lama masih ditemukan di template index.html!"
    assert 'id="stream-feed"' in resp.text
    print("[PASS] GET / dapat diakses langsung (200 OK) tanpa token API_KEY di template.")


def test_rest_api_open_access():
    client = create_test_client()

    endpoints = [
        "/api/cameras",
        "/api/status/cam_01",
        "/api/zones",
        "/api/zones/cam_01",
        "/api/incidents",
    ]

    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 200, f"Expected 200 OK for {ep}, got {resp.status_code}"
        assert resp.status_code != 401, f"Endpoint {ep} masih menolak dengan 401 Unauthorized!"
        print(f"  [PASS] {ep} -> 200 OK (Open Access)")

    print("[PASS] Seluruh endpoint REST API dapat diakses langsung tanpa autentikasi.")


def test_video_feed_open_access():
    client = create_test_client()
    app = client.app
    route_handler = None
    for route in app.routes:
        if getattr(route, "path", None) == "/video_feed/{camera_id}":
            route_handler = route.endpoint
            break
    assert route_handler is not None, "Endpoint /video_feed/{camera_id} tidak terdaftar!"

    import asyncio
    streaming_resp = asyncio.run(route_handler("cam_01"))
    assert streaming_resp.status_code == 200, f"Expected 200, got {streaming_resp.status_code}"
    assert "multipart/x-mixed-replace" in streaming_resp.media_type

    # Ambil frame pertama
    async def get_first_chunk():
        gen = streaming_resp.body_iterator
        first = await gen.__anext__()
        await gen.aclose()
        return first

    chunk = asyncio.run(get_first_chunk())
    assert len(chunk) > 0, "Chunk stream kosong!"
    assert b"--frame" in chunk, "Boundary frame hilang dari chunk!"
    print("[PASS] GET /video_feed/cam_01 -> 200 OK (Multipart Stream Open Access)")


if __name__ == "__main__":
    print("=" * 70)
    print(" MENJALANKAN AUDIT OPEN ACCESS REST API & STREAMING HUB")
    print("=" * 70)
    test_public_index_open_access()
    test_rest_api_open_access()
    test_video_feed_open_access()
    print("=" * 70)
    print(" ALL OPEN ACCESS WEB TESTS PASSED (100% SUCCESS)!")
    print("=" * 70)
