from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env sebelum import engine apapun ────────────────────────────────────
load_dotenv(Path(".env"), override=False)

# ── Muat Settings Global & Inisialisasi Logging Terpusat ───────────────────────
def load_global_settings() -> dict:
    """Muat konfigurasi global dari configs/settings.json."""
    settings_path = Path("configs/settings.json")
    if settings_path.exists():
        try:
            return json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Gagal membaca settings.json: {e}")
    return {}


from engine.logger import (
    setup_logging,
    get_logger,
    render_banner,
    render_preflight_table,
    log_error,
    log_session_start,
    log_session_end,
)

_settings = load_global_settings()
_log_cfg = _settings.get("logging", {})
_log_file = os.environ.get("LOG_FILE", _log_cfg.get("file", "logs/app.log"))
_log_level = os.environ.get("LOG_LEVEL", _settings.get("app", {}).get("log_level", "INFO")).upper()
_max_bytes = int(_log_cfg.get("max_bytes", 5 * 1024 * 1024))
_backup_count = int(_log_cfg.get("backup_count", 3))
_console_color = bool(_log_cfg.get("console_color", True))

setup_logging(
    log_level=_log_level,
    file_log_level="DEBUG",
    log_file=_log_file,
    max_bytes=_max_bytes,
    backup_count=_backup_count,
    console_color=_console_color,
)
logger = get_logger("MasterOrchestrator")


class WebServerThread(threading.Thread):
    """Thread terkelola untuk menjalankan ASGI server (Uvicorn)."""

    def __init__(self, app, host: str = "0.0.0.0", port: int = 8070) -> None:
        super().__init__(name="WebServerThread", daemon=True)
        import uvicorn
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self) -> None:
        logger.debug(f"[WebHub] Menjalankan Web Streaming Hub pada http://{self.server.config.host}:{self.server.config.port}")
        self.server.run()

    def stop(self) -> None:
        logger.debug("[WebHub] Menghentikan Web Streaming Hub...")
        self.server.should_exit = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Facility Management — Master Orchestrator (Fase 4 Production)")
    parser.add_argument("--host", default=None, help="Host binding Web Streaming Hub (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port binding Web Streaming Hub (default: 8070)")
    parser.add_argument("--dry-run", action="store_true", help="Uji startup seluruh subsistem selama 3s lalu shutdown bersih")
    args = parser.parse_args()

    start_time = time.time()
    mode_str = "DRY-RUN AKTIF" if args.dry_run else "Production"
    log_session_start("Facility Management", "0.4.0")
    banner_card = render_banner(
        title="FACILITY MANAGEMENT",
        version="0.4.0",
        details={
            "Mode": mode_str,
            "Target Web": f"http://{args.host or '0.0.0.0'}:{args.port or 8070}",
        },
    )
    print(banner_card)
    logger.info(f"Facility Management v0.4.0 — Master Orchestrator Starting... [MODE: {mode_str}]")

    # ── Step 1: Audit Schema (wajib lulus sebelum engine aktif) ───────────────
    logger.debug("Menjalankan schema audit...")
    from tools.audit_schema import run_audit
    audit_result = run_audit(silent=True)
    if audit_result != 0:
        log_error(
            logger,
            "Schema audit GAGAL — engine tidak akan dijalankan.",
            mitigation_hint="Periksa berkas cameras/*/config.json dan roi_zones.json via tools/audit_schema.py",
        )
        sys.exit(1)
    logger.debug("Schema audit: PASS ✓")

    # ── Step 2: Inisialisasi Database ─────────────────────────────────────────
    logger.debug("Menginisialisasi database (SQLite WAL)...")
    from storage.database import init_db, get_writer_conn, close_db
    db_path = os.environ.get("DB_PATH", "storage/facility.db")
    init_db(db_path)
    logger.debug("Database: READY ✓")

    # ── Step 3: Muat Konfigurasi & Auto-Discovery Kamera ──────────────────────
    from engine.config_loader import load_camera_config, load_roi_zones, CameraConfig, ROIZonesConfig
    cameras_dir = Path("cameras")
    cam_dirs = sorted([
        d for d in cameras_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "config.json").exists()
    ])

    if not cam_dirs:
        logger.warning("Tidak ada kamera terkonfigurasi. Keluar.")
        sys.exit(0)

    camera_configs: list[CameraConfig] = []
    roi_configs: dict[str, ROIZonesConfig] = {}
    for cam_dir in cam_dirs:
        cfg = load_camera_config(cam_dir / "config.json")
        roi = load_roi_zones(cam_dir / "roi_zones.json")
        camera_configs.append(cfg)
        roi_configs[cfg.camera_id] = roi
        logger.debug(f"Kamera dimuat via auto-discovery: {cfg.safe_repr()}")

    # ── Step 4: Daftarkan Kamera ke Registry ──────────────────────────────────
    from storage.models import CameraRegistry
    conn = get_writer_conn()
    for cfg in camera_configs:
        reg = CameraRegistry(camera_id=cfg.camera_id, display_name=cfg.display_name)
        reg.upsert(conn)
    logger.debug(f"Camera registry: {len(camera_configs)} kamera terdaftar ✓")

    # ── Step 5: Muat Pengaturan Global ────────────────────────────────────────
    settings = load_global_settings()
    web_cfg = settings.get("web", {})
    alert_cfg = settings.get("alert", {})
    concurrency_cfg = settings.get("concurrency", {})

    web_host = args.host or os.environ.get("WEB_HOST", web_cfg.get("host", "0.0.0.0"))
    web_port = args.port or int(os.environ.get("WEB_PORT", web_cfg.get("port", 8070)))
    live_quality = int(web_cfg.get("live_jpeg_quality", 75))

    cooldown_sec = float(alert_cfg.get("cooldown_sec", 5.0))
    queue_maxsize = int(alert_cfg.get("queue_maxsize", 30))
    enable_audio = bool(alert_cfg.get("enable_audio", True))

    max_ai_conc = int(concurrency_cfg.get("max_concurrent_ai_inferences", 1))

    # ── Step 6: Inisialisasi Hubungan Terpusat (Buffer, Alert, Semaphore) ─────
    from web.buffer import MultiCameraBuffer
    from notification.dispatcher import AlertDispatcher
    from notification import set_global_dispatcher

    frame_buffer = MultiCameraBuffer()

    alert_dispatcher = AlertDispatcher(
        queue_maxsize=queue_maxsize,
        cooldown_sec=cooldown_sec,
        enable_audio=enable_audio,
    )
    set_global_dispatcher(alert_dispatcher)
    alert_dispatcher.start()
    logger.debug("AlertDispatcher daemon: STARTED ✓")

    ai_semaphore = threading.BoundedSemaphore(value=max_ai_conc)
    logger.debug(f"Global AI BoundedSemaphore (concurrency={max_ai_conc}): READY ✓")

    # ── Step 7: Start DiskGuard Daemon ────────────────────────────────────────
    from engine.disk_guard import DiskGuard
    disk_guard = DiskGuard(
        camera_configs=camera_configs,
        global_threshold_pct=float(os.environ.get("DISK_GUARD_THRESHOLD_PCT", "80")),
        global_threshold_gb=float(os.environ.get("DISK_GUARD_THRESHOLD_GB", "5")),
        base_dir=".",
    )
    disk_guard.start()
    logger.debug("DiskGuard daemon: STARTED ✓")

    # ── Step 8: Start Full Pipeline per Kamera ────────────────────────────────
    from engine.capture import create_capture_pipeline
    from engine.preprocessor import create_preprocessor_pipeline
    from engine.pipeline import CameraOrchestrator

    workers: list[threading.Thread] = []
    orchestrator_map: dict[str, CameraOrchestrator] = {}

    total_cams = len(camera_configs)
    for idx, cfg in enumerate(camera_configs):
        capture_worker, frame_queue = create_capture_pipeline(cfg)
        preproc_worker, proc_queue = create_preprocessor_pipeline(cfg, frame_queue)
        orchestrator = CameraOrchestrator(
            camera_config=cfg,
            roi_config=roi_configs[cfg.camera_id],
            proc_queue=proc_queue,
            db_path=db_path,
            ai_semaphore=ai_semaphore,
            frame_buffer=frame_buffer,
            alert_dispatcher=alert_dispatcher,
            live_jpeg_quality=live_quality,
            camera_index=idx,
            total_cameras=total_cams,
        )

        capture_worker.start()
        preproc_worker.start()
        orchestrator.start()

        workers.extend([capture_worker, preproc_worker])
        orchestrator_map[cfg.camera_id] = orchestrator

        logger.debug(
            f"[{cfg.camera_id}] Full Pipeline (Capture + Preproc + Vision Engine + Buffer) aktif "
            f"({cfg.resolution.ai_width}x{cfg.resolution.ai_height}) [Index: {idx}/{total_cams}]"
        )

    # ── Step 9: Start Web Streaming Hub Server (FastAPI) ─────────────────────
    from web.app import create_app
    web_app = create_app(frame_buffer, camera_configs, orchestrator_map)
    web_server = WebServerThread(app=web_app, host=web_host, port=web_port)
    web_server.start()
    logger.debug(f"[WebHub] Menjalankan Web Streaming Hub pada http://{web_host}:{web_port}")

    # ── Compact Pre-Flight Checklist Table ───────────────────────────────────
    cam_summary = ", ".join([f"{c.camera_id} ({c.resolution.ai_width}x{c.resolution.ai_height})" for c in camera_configs])
    preflight_checks = {
        "Schema Audit": f"{len(camera_configs)}/{len(camera_configs)} Kamera Valid ✓",
        "Basis Data (WAL)": f"{db_path} (WAL Mode OK)",
        "Profil Kamera": cam_summary,
        "Resource Guard": f"Semaphore AI ({max_ai_conc}), DiskGuard (<{int(disk_guard.global_threshold_pct)}%/{int(disk_guard.global_threshold_gb)}GB)",
        "Web Streaming Hub": f"http://{web_host}:{web_port} (Active)",
    }
    if total_cams > 1:
        preflight_checks["Multi-Cam Balancing"] = f"Staggered AI Cadence (1/{total_cams} Round-Robin) ✓"
    print(render_preflight_table(preflight_checks))
    logger.info(f"Facility Management Hub v0.4.0 aktif pada http://{web_host}:{web_port} ({len(camera_configs)} kamera online). Tekan Ctrl+C untuk shutdown.")

    # ── Step 10: Graceful Shutdown Handler & Dry-Run Logic ────────────────────
    shutdown_event = threading.Event()

    def _handle_shutdown(signum, frame):
        logger.info(f"Signal {signum} diterima — memulai graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    if args.dry_run:
        logger.info("[DRY-RUN] Membiarkan subsistem berjalan selama 3 detik untuk verifikasi...")
        time.sleep(3.0)
        logger.info("[DRY-RUN] Waktu verifikasi selesai — memicu shutdown otomatis...")
        shutdown_event.set()

    # ── Main Event Loop ───────────────────────────────────────────────────────
    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_event.set()

    # ── Berjenjang Shutdown Sequence ──────────────────────────────────────────
    logger.debug("Shutdown sequence dimulai...")

    web_server.stop()

    for orchestrator in orchestrator_map.values():
        orchestrator.stop()

    for worker in workers:
        if hasattr(worker, "stop"):
            worker.stop()

    alert_dispatcher.stop()
    disk_guard.stop()

    for orchestrator in orchestrator_map.values():
        orchestrator.join(timeout=3.0)

    for worker in workers:
        worker.join(timeout=3.0)

    alert_dispatcher.join(timeout=2.0)
    frame_buffer.clear()

    try:
        for cfg in camera_configs:
            reg = CameraRegistry(camera_id=cfg.camera_id, display_name=cfg.display_name, status="inactive")
            reg.upsert(conn)
    except Exception as e:
        logger.debug(f"Gagal memperbarui status registry saat shutdown: {e}")

    close_db()
    uptime_sec = round(time.time() - start_time, 1)
    logger.info(f"Facility Management: shutdown selesai (Durasi operasional: {uptime_sec}s).")
    log_session_end("Graceful Shutdown Complete (shutdown selesai)")

    if args.dry_run:
        print("\n[DRY-RUN] Seluruh subsistem beroperasi normal. Keluar bersih (Exit Code 0).")
        sys.exit(0)


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    main()
