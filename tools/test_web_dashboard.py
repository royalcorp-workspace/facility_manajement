"""
Facility Management — Test Suite: Web Dashboard 100vh Kiosk Video Stream Focus
Memvalidasi integrasi FastAPI, Jinja2 template rendering, static file delivery, dan tokens CSS 100vh Kiosk.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Tambahkan root workspace
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from engine.config_loader import load_camera_config
from web.app import create_app
from web.buffer import MultiCameraBuffer


def test_dashboard_kiosk_mode():
    print("==================================================================")
    print("  FACILITY MANAGEMENT — WEB DASHBOARD 100VH KIOSK MODE VALIDATION")
    print("==================================================================")

    # 1. Setup minimal app
    cam_cfg = load_camera_config(PROJECT_ROOT / "cameras" / "cam_01" / "config.json")
    buffer = MultiCameraBuffer()
    buffer.set_frame(cam_cfg.camera_id, b"\xff\xd8\xff\xe0fake", time.time(), 1)
    app = create_app(buffer=buffer, camera_configs=[cam_cfg])
    client = TestClient(app)

    # 2. Test GET / (HTML Template rendering)
    print("\n[1/5] Testing GET / (HTML Template Rendering)...")
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "Facility Management — Live Surveillance Hub" in resp.text
    assert "/static/css/dashboard.css" in resp.text
    assert "camera-switcher" in resp.text
    assert "stream-feed" in resp.text
    assert "main-viewport" in resp.text
    assert "camera-hud" in resp.text
    print("  ✓ GET / rendered valid 100vh Kiosk HTML with single video stream focus.")

    # 3. Test GET /static/css/dashboard.css
    print("\n[2/5] Testing GET /static/css/dashboard.css...")
    resp_css = client.get("/static/css/dashboard.css")
    assert resp_css.status_code == 200, f"Expected 200, got {resp_css.status_code}"
    assert "text/css" in resp_css.headers.get("content-type", "")
    css_text = resp_css.text
    print("  ✓ dashboard.css successfully served via FastAPI static mount.")

    # 4. Validate Viewport Lock and 100vh Kiosk CSS Rules
    print("\n[3/5] Validating 100vh Viewport Lock & Video Stream Focus Rules...")
    kiosk_rules = [
        "height: 100vh",
        "max-height: 100vh",
        "overflow: hidden",
        "--navbar-height: 44px",
        "--bg-app: #f8fafc",
        "--bg-viewport: #000000",
        "object-fit: contain",
        ".camera-hud",
    ]
    for rule in kiosk_rules:
        assert rule in css_text, f"Kiosk CSS rule missing: {rule}"
    print("  ✓ 100vh viewport lock, zero scrollbar, and object-fit: contain verified.")

    # 5. Test Backward Compatibility with /static/css/style.css
    print("\n[4/5] Testing /static/css/style.css backward-compatibility forwarder...")
    resp_old_css = client.get("/static/css/style.css")
    assert resp_old_css.status_code == 200
    assert "dashboard.css" in resp_old_css.text
    print("  ✓ style.css properly forwards to dashboard.css.")

    # 6. Test Core API Endpoints
    print("\n[5/5] Testing Core Web API Endpoints (/api/cameras, /api/status/cam_01)...")
    # 5a. Akses tanpa API key wajib 401
    resp_unauth = client.get("/api/cameras")
    assert resp_unauth.status_code == 401, f"Expected 401 unauthenticated, got {resp_unauth.status_code}"

    # 5b. Akses terautentikasi (header / query param)
    api_key = "facility-royal-2026"
    resp_cams = client.get("/api/cameras", headers={"X-API-Key": api_key})
    assert resp_cams.status_code == 200
    assert isinstance(resp_cams.json(), list)

    resp_status = client.get(f"/api/status/cam_01?api_key={api_key}")
    assert resp_status.status_code == 200
    assert resp_status.json().get("camera_id") == "cam_01"
    print("  [PASS] API endpoints protected with API Key (401 unauthenticated, 200 authenticated).")


    print("\n==================================================================")
    print("  ALL WEB DASHBOARD 100VH KIOSK TESTS PASSED! [OK]")
    print("==================================================================")


if __name__ == "__main__":
    test_dashboard_kiosk_mode()
