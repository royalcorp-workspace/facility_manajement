from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from engine.config_loader import CameraConfig
from engine.logger import get_logger
from storage.database import get_reader_conn
from web.buffer import MultiCameraBuffer, generate_fallback_canvas

logger = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def verify_api_key(request: Request) -> str:
    """
    Dependency keamanan API Key:
    1. Membaca API Key dari HTTP Header 'X-API-Key'.
    2. Fallback membaca dari Query Parameter '?api_key=...' (atau '?x-api-key=...').
    3. Jika tidak ada atau tidak cocok, lempar HTTP 401 Unauthorized.
    """
    expected_key = os.getenv("API_KEY", "facility-royal-2026")
    provided_key = request.headers.get("X-API-Key")
    if not provided_key:
        provided_key = request.query_params.get("api_key") or request.query_params.get("x-api-key")

    if not provided_key or provided_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return provided_key


def create_app(
    buffer: MultiCameraBuffer,
    camera_configs: List[CameraConfig],
    orchestrators: Optional[Dict[str, any]] = None,
) -> FastAPI:

    app = FastAPI(title="Facility Management Vision Hub", version="0.3.0")

    # Static files & Jinja2 Templates
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    cam_map = {cfg.camera_id: cfg for cfg in camera_configs}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        api_key = os.getenv("API_KEY", "facility-royal-2026")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "cameras": camera_configs,
                "api_key": api_key,
            },
        )

    @app.get("/video_feed/{camera_id}", dependencies=[Depends(verify_api_key)])
    async def video_feed(camera_id: str):
        if camera_id not in cam_map and camera_id not in buffer.get_registered_cameras():
            raise HTTPException(status_code=404, detail=f"Kamera '{camera_id}' tidak ditemukan")

        async def mjpeg_generator():
            last_seq = -1
            consecutive_idle_cycles = 0
            try:
                while True:
                    # Ambil frame via run_in_executor untuk mencegah blocking event loop
                    frame = await asyncio.to_thread(
                        buffer.wait_for_next_frame,
                        camera_id=camera_id,
                        last_sequence_id=last_seq,
                        timeout=0.5,
                    )

                    if frame is None or frame.sequence_id == last_seq:
                        consecutive_idle_cycles += 1
                        # Jika kamera reconnecting / putus >= 2 detik (4 siklus x 0.5s), sajikan fallback canvas informatif
                        if consecutive_idle_cycles >= 4:
                            consecutive_idle_cycles = 0
                            cfg_cam = cam_map.get(camera_id)
                            w = cfg_cam.resolution.ai_width if cfg_cam else 640
                            h = cfg_cam.resolution.ai_height if cfg_cam else 360
                            fallback_bytes = generate_fallback_canvas(
                                camera_id=camera_id,
                                message="CONNECTING / RECONNECTING...",
                                width=w,
                                height=h,
                            )
                            header = (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(fallback_bytes)).encode() + b"\r\n\r\n"
                            )
                            yield header + fallback_bytes + b"\r\n"

                        await asyncio.sleep(0.02)
                        continue

                    # Frame aktif valid diterima
                    consecutive_idle_cycles = 0
                    last_seq = frame.sequence_id
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame.jpeg_bytes)).encode() + b"\r\n\r\n"
                    )
                    yield header + frame.jpeg_bytes + b"\r\n"

                    # Yield control singkat
                    await asyncio.sleep(0.01)

            except (GeneratorExit, asyncio.CancelledError):
                logger.debug(f"[Streaming] Klien memutus koneksi feed: {camera_id}")
                return

        return StreamingResponse(
            mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status/{camera_id}", dependencies=[Depends(verify_api_key)])
    async def get_camera_status(camera_id: str):
        cfg = cam_map.get(camera_id)
        if not cfg:
            raise HTTPException(status_code=404, detail="Kamera tidak ditemukan")

        orchestrator = orchestrators.get(camera_id) if orchestrators else None

        processed = orchestrator.processed_count if orchestrator else 0
        motion = orchestrator.motion_detected_count if orchestrator else 0
        status_val = "active" if (orchestrator and orchestrator.is_alive()) else "inactive"
        fps_val = round(orchestrator.fps, 1) if (orchestrator and hasattr(orchestrator, "fps")) else 0.0
        active_tracks_count = len(orchestrator.tracker.tracks) if (orchestrator and hasattr(orchestrator, "tracker") and hasattr(orchestrator.tracker, "tracks")) else 0

        parking_info = None
        if orchestrator and hasattr(orchestrator, "parking_tracker"):
            pt = orchestrator.parking_tracker
            is_block = getattr(pt, "parking_mode", "slot") == "motorcycle_block"
            if is_block:
                occ = getattr(pt, "occupied_slots", 0)
                if hasattr(pt, "_last_stable_occupied") and pt._last_stable_occupied is not None:
                    occ = pt._last_stable_occupied
                avail = max(0, pt.total_slots - occ)
                parking_info = {
                    "parking_mode": "motorcycle_block",
                    "total_slots": pt.total_slots,
                    "occupied_slots": occ,
                    "available_slots": avail,
                    "gate_in": pt.gate_in_count,
                    "gate_out": pt.gate_out_count,
                    "stationary_units_count": len(getattr(pt, "_stationary_motor_units", {})),
                    "slots": {},
                }
            else:
                occ = sum(1 for s in pt.slot_states.values() if s.occupied)
                avail = max(0, pt.total_slots - occ)
                parking_info = {
                    "parking_mode": "slot",
                    "total_slots": pt.total_slots,
                    "occupied_slots": occ,
                    "available_slots": avail,
                    "gate_in": pt.gate_in_count,
                    "gate_out": pt.gate_out_count,
                    "slots": {
                        s_id: {
                            "phase": s.phase,
                            "occupied": s.occupied,
                            "track_id": s.track_id,
                            "dwell": round(s.dwell_duration, 1),
                        }
                        for s_id, s in pt.slot_states.items()
                    },
                }

        return {
            "camera_id": camera_id,
            "display_name": cfg.display_name,
            "status": status_val,
            "fps": fps_val,
            "active_tracks": active_tracks_count,
            "processed_count": processed,
            "motion_detected_count": motion,
            "resolution": f"{cfg.resolution.capture_width}x{cfg.resolution.capture_height} -> {cfg.resolution.ai_width}x{cfg.resolution.ai_height}",
            "parking": parking_info,
        }

    @app.get("/api/cameras", dependencies=[Depends(verify_api_key)])
    async def get_cameras():
        """Daftar seluruh kamera aktif dan telemetrinya."""
        result = []
        for cfg in camera_configs:
            orchestrator = orchestrators.get(cfg.camera_id) if orchestrators else None
            processed = orchestrator.processed_count if orchestrator else 0
            motion = orchestrator.motion_detected_count if orchestrator else 0
            status_val = "active" if (orchestrator and orchestrator.is_alive()) else "inactive"
            fps_val = round(orchestrator.fps, 1) if (orchestrator and hasattr(orchestrator, "fps")) else 0.0
            active_tracks_count = len(orchestrator.tracker.tracks) if (orchestrator and hasattr(orchestrator, "tracker") and hasattr(orchestrator.tracker, "tracks")) else 0

            parking_info = None
            if orchestrator and hasattr(orchestrator, "parking_tracker"):
                pt = orchestrator.parking_tracker
                is_block = getattr(pt, "parking_mode", "slot") == "motorcycle_block"
                if is_block:
                    occ = getattr(pt, "occupied_slots", 0)
                    if hasattr(pt, "_last_stable_occupied") and pt._last_stable_occupied is not None:
                        occ = pt._last_stable_occupied
                    avail = max(0, pt.total_slots - occ)
                    parking_info = {
                        "parking_mode": "motorcycle_block",
                        "total_slots": pt.total_slots,
                        "occupied_slots": occ,
                        "available_slots": avail,
                        "gate_in": pt.gate_in_count,
                        "gate_out": pt.gate_out_count,
                        "stationary_units_count": len(getattr(pt, "_stationary_motor_units", {})),
                        "slots": {},
                    }
                else:
                    occ = sum(1 for s in pt.slot_states.values() if s.occupied)
                    avail = max(0, pt.total_slots - occ)
                    parking_info = {
                        "parking_mode": "slot",
                        "total_slots": pt.total_slots,
                        "occupied_slots": occ,
                        "available_slots": avail,
                        "gate_in": pt.gate_in_count,
                        "gate_out": pt.gate_out_count,
                    }

            result.append({
                "camera_id": cfg.camera_id,
                "display_name": cfg.display_name,
                "status": status_val,
                "fps": fps_val,
                "active_tracks": active_tracks_count,
                "processed_count": processed,
                "motion_detected_count": motion,
                "resolution": f"{cfg.resolution.capture_width}x{cfg.resolution.capture_height} -> {cfg.resolution.ai_width}x{cfg.resolution.ai_height}",
                "parking": parking_info,
            })
        return result

    @app.get("/api/zones", dependencies=[Depends(verify_api_key)])
    async def get_all_zones(cam: Optional[str] = None):
        """Dapatkan seluruh definisi zona ROI (poligon & tripwire) untuk seluruh kamera atau spesifik via ?cam=."""
        import json
        if cam:
            roi_path = Path(f"cameras/{cam}/roi_zones.json")
            if not roi_path.exists():
                raise HTTPException(status_code=404, detail=f"Zona untuk kamera '{cam}' tidak ditemukan")
            try:
                return json.loads(roi_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Gagal membaca konfigurasi zona: {e}")

        result = {}
        for cfg in camera_configs:
            roi_path = Path(f"cameras/{cfg.camera_id}/roi_zones.json")
            if roi_path.exists():
                try:
                    result[cfg.camera_id] = json.loads(roi_path.read_text(encoding="utf-8"))
                except Exception:
                    result[cfg.camera_id] = {"polygons": [], "tripwires": []}
            else:
                result[cfg.camera_id] = {"polygons": [], "tripwires": []}
        return result

    @app.get("/api/zones/{camera_id}", dependencies=[Depends(verify_api_key)])
    async def get_camera_zones(camera_id: str):
        """Dapatkan definisi zona ROI spesifik untuk kamera yang diminta."""
        import json
        roi_path = Path(f"cameras/{camera_id}/roi_zones.json")
        if not roi_path.exists():
            raise HTTPException(status_code=404, detail=f"Zona untuk kamera '{camera_id}' tidak ditemukan")
        try:
            return json.loads(roi_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal membaca konfigurasi zona: {e}")

    @app.get("/api/incidents", dependencies=[Depends(verify_api_key)])
    async def get_incidents():
        """Dapatkan log insiden/lifecycle events terakhir dari SQLite database."""
        try:
            conn = get_reader_conn()
            cursor = conn.execute(
                "SELECT id, camera_id, zone_id, event_type, track_id, class_label, "
                "confidence, timestamp, snapshot_path, is_resolved, resolved_time, notes "
                "FROM lifecycle_events ORDER BY id DESC LIMIT 50"
            )
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"[API] Gagal mengambil insiden: {e}")
            return []

    @app.post("/api/incidents/{incident_id}/resolve", dependencies=[Depends(verify_api_key)])
    async def resolve_incident_endpoint(incident_id: int):
        """Tandai insiden sebagai telah ditangani oleh operator (thread-safe)."""
        from storage.database import resolve_incident
        from datetime import datetime, timezone

        success = await asyncio.to_thread(resolve_incident, incident_id=incident_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Insiden ID {incident_id} tidak ditemukan atau sudah diselesaikan.",
            )
        return {
            "status": "success",
            "incident_id": incident_id,
            "resolved_time": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/snapshots/{camera_id}/{filename}", dependencies=[Depends(verify_api_key)])
    async def get_snapshot(camera_id: str, filename: str):
        """Menyajikan file snapshot JPEG bukti secara aman."""
        base_dir = Path(f"cameras/{camera_id}/snapshots").resolve()
        file_path = (base_dir / filename).resolve()

        # Validasi path traversal
        if not str(file_path).startswith(str(base_dir)):
            raise HTTPException(status_code=403, detail="Akses ditolak")

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Snapshot tidak ditemukan")

        content = file_path.read_bytes()
        return Response(content=content, media_type="image/jpeg")

    return app
