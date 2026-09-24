"""
Unit test untuk verifikasi autentikasi API Key pada endpoint REST & Video Feed:
1. Header X-API-Key
2. Query Parameter ?api_key=...
3. Penolakan 401 Unauthorized jika tidak ada atau salah
4. Akses publik index / dan static assets
"""

import os
import sys
from pathlib import Path

# Tambahkan root proyek ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from engine.config_loader import load_camera_config
from web.app import create_app
from web.buffer import MultiCameraBuffer


def create_test_client():
    os.environ["API_KEY"] = "facility-royal-2026"
    cam_cfg = load_camera_config(Path(__file__).resolve().parent.parent / "cameras" / "cam_01" / "config.json")
    buffer = MultiCameraBuffer()
    buffer.set_frame(cam_cfg.camera_id, b"\xff\xd8\xff\xe0fake", 100.0, 1)
    app = create_app(buffer=buffer, camera_configs=[cam_cfg], orchestrators={})
    return TestClient(app)


def test_public_index_accessible():
    client = create_test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "facility-royal-2026" in resp.text
    print("[PASS] GET / dapat diakses langsung dan menyertakan token API_KEY di template.")


def test_missing_api_key_rejected():
    client = create_test_client()

    for path in ["/api/cameras", "/api/status/cam_01", "/api/zones", "/video_feed/cam_01"]:
        resp = client.get(path)
        assert resp.status_code == 401, f"Expected 401 for {path}, got {resp.status_code}"
        assert resp.json() == {"detail": "Invalid or missing API Key"}
    print("[PASS] Akses tanpa API Key berhasil ditolak dengan HTTP 401 Unauthorized.")


def test_invalid_api_key_rejected():
    client = create_test_client()

    # Via Header
    resp = client.get("/api/cameras", headers={"X-API-Key": "wrong-secret"})
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid or missing API Key"}

    # Via Query Param
    resp = client.get("/api/status/cam_01?api_key=wrong-secret")
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid or missing API Key"}

    print("[PASS] Akses dengan API Key salah berhasil ditolak dengan HTTP 401 Unauthorized.")


def test_valid_header_accepted():
    client = create_test_client()
    resp = client.get("/api/cameras", headers={"X-API-Key": "facility-royal-2026"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = client.get("/api/zones", headers={"X-API-Key": "facility-royal-2026"})
    assert resp.status_code == 200
    print("[PASS] Akses dengan header 'X-API-Key: facility-royal-2026' berhasil 200 OK.")


def test_valid_query_param_accepted():
    client = create_test_client()
    resp = client.get("/api/cameras?api_key=facility-royal-2026")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = client.get("/api/status/cam_01?api_key=facility-royal-2026")
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == "cam_01"

    print("[PASS] Akses dengan query parameter '?api_key=facility-royal-2026' berhasil 200 OK.")


if __name__ == "__main__":
    test_public_index_accessible()
    test_missing_api_key_rejected()
    test_invalid_api_key_rejected()
    test_valid_header_accepted()
    test_valid_query_param_accepted()
    print("\nALL API KEY AUTHENTICATION TESTS PASSED (100% SUCCESS)!")
