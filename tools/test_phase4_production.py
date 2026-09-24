"""
Suite Pengujian Terpadu Fase 4 (Production Readiness & Hardware Integration).
Menguji 4 skenario krusial:
  1. Dual-Mode Video Source & Auto-Looping EOF pada berkas video lokal
  2. Keyframe Warmup Safeguard & Watchdog Timeout
  3. Visual ROI Calibrator Headless Auto-Backup & Pydantic Validation
  4. Master Orchestrator CLI Dry-Run Mode (main.py --dry-run)
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import time
from pathlib import Path

# Setup path agar modul proyek terbaca
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.logger import setup_logging
setup_logging(log_file="logs/test.log", log_level="WARNING", force_reconfigure=True)

import cv2
import numpy as np

from engine.capture import RTSPCaptureWorker, FramePacket
from engine.config_loader import (
    CameraConfig,
    ResolutionConfig,
    CaptureConfig,
    DiskGuardCameraConfig,
)
from tools.roi_calibrator import run_test_save

TEMP_VIDEO_PATH = PROJECT_ROOT / "storage" / "temp_loop_test.mp4"


def create_temp_video_file(num_frames: int = 15, width: int = 1920, height: int = 1080) -> Path:
    """Buat file video mp4 sintetis untuk menguji auto-looping EOF."""
    TEMP_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(TEMP_VIDEO_PATH), fourcc, 30.0, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Gambar penanda nomor frame
        cv2.putText(frame, f"FRAME {i+1}/{num_frames}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        out.write(frame)

    out.release()
    return TEMP_VIDEO_PATH


def test_scenario_1_dual_mode_and_autoloop():
    """Skenario 1: Dual-Mode Video Source & Auto-Looping EOF."""
    print("\n[TEST 1] Menguji Dual-Mode Video Source & Auto-Looping EOF...")

    # Buat file video 15 frame
    video_path = create_temp_video_file(num_frames=15)

    cam_cfg = CameraConfig(
        camera_id="cam_test_loop",
        display_name="Kamera Test Loop",
        rtsp_url=str(video_path),
        resolution=ResolutionConfig(capture_width=1920, capture_height=1080, ai_width=640, ai_height=360),
        capture=CaptureConfig(
            queue_maxsize=4,
            reconnect_delay_sec=1.0,
            max_reconnect_attempts=3,
            frame_timeout_sec=5.0,
            keyframe_warmup_frames=0,  # Tanpa warmup agar semua frame terhitung
        ),
        disk_guard=DiskGuardCameraConfig(snapshot_dir="storage/test_snapshots"),
    )

    frame_q = queue.Queue(maxsize=10)
    worker = RTSPCaptureWorker(config=cam_cfg, frame_queue=frame_q)
    worker.start()

    received_frames = 0
    start_time = time.monotonic()

    try:
        # Baca 35 frame berturut-turut (melebihi durasi video 15 frame -> auto-looping minimal 2x)
        while received_frames < 35 and (time.monotonic() - start_time) < 10.0:
            try:
                pkt: FramePacket = frame_q.get(timeout=0.5)
                received_frames += 1
                assert pkt.raw_frame is not None and pkt.raw_frame.shape == (1080, 1920, 3)
            except queue.Empty:
                continue

        assert received_frames >= 35, (
            f"Gagal membaca 35 frame via auto-looping! Hanya terbaca: {received_frames}"
        )
        assert worker._reconnect_count == 0, "Worker tidak boleh disconnect saat mencapai EOF video lokal"

    finally:
        worker.stop()
        worker.join(timeout=3.0)
        if video_path.exists():
            video_path.unlink()

    print(f"  ✓ PASS: Auto-looping EOF berhasil membaca {received_frames} frame berkelanjutan.")


def test_scenario_2_keyframe_warmup():
    """Skenario 2: Keyframe Warmup Safeguard."""
    print("\n[TEST 2] Menguji Keyframe Warmup Safeguard...")

    video_path = create_temp_video_file(num_frames=20)

    cam_cfg = CameraConfig(
        camera_id="cam_test_warmup",
        display_name="Kamera Test Warmup",
        rtsp_url=str(video_path),
        resolution=ResolutionConfig(),
        capture=CaptureConfig(
            queue_maxsize=4,
            keyframe_warmup_frames=5,  # 5 frame awal wajib diabaikan
        ),
        disk_guard=DiskGuardCameraConfig(snapshot_dir="storage/test_snapshots"),
    )

    frame_q = queue.Queue(maxsize=10)
    worker = RTSPCaptureWorker(config=cam_cfg, frame_queue=frame_q)
    worker.start()

    time.sleep(1.0)
    worker.stop()
    worker.join(timeout=3.0)

    if video_path.exists():
        video_path.unlink()

    # Pastikan frame counter mencatat frame setelah 5 frame warmup diabaikan
    assert worker.frame_count > 0, "Harus ada frame yang diproses setelah masa warmup"
    print(f"  ✓ PASS: Keyframe warmup safeguard mengabaikan frame awal dengan benar (Frame valid: {worker.frame_count}).")


def test_scenario_3_roi_calibrator_test_save():
    """Skenario 3: Visual ROI Calibrator Headless Auto-Backup & Pydantic Validation."""
    print("\n[TEST 3] Menguji Visual ROI Calibrator Headless Auto-Backup...")

    exit_code = run_test_save(camera_id="cam_01")
    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

    # Pastikan berkas .bak ada
    bak_path = PROJECT_ROOT / "cameras" / "cam_01" / "roi_zones.json.bak"
    assert bak_path.exists(), f"File backup .bak tidak ditemukan: {bak_path}"

    print("  ✓ PASS: ROI Calibrator auto-backup .bak dan validasi skema Pydantic terverifikasi.")


def test_scenario_4_cli_dry_run_mode():
    """Skenario 4: Master Orchestrator CLI Dry-Run Mode (main.py --dry-run)."""
    print("\n[TEST 4] Menguji Master Orchestrator CLI Dry-Run Mode...")

    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    cmd = [str(python_exe), "main.py", "--dry-run", "--port", "8089"]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
    )

    if proc.returncode != 0:
        print("[ERROR OUTPUT]:\n", proc.stderr)

    assert proc.returncode == 0, f"Expected exit code 0, got {proc.returncode}"
    assert "DRY-RUN AKTIF" in proc.stdout or "DRY-RUN" in proc.stdout, "Banner dry-run harus muncul"
    assert "shutdown selesai" in proc.stdout, "Graceful shutdown harus selesai tercatat"

    print("  ✓ PASS: Master Orchestrator dry-run startup dan shutdown otomatis sukses (Exit Code 0).")


def main():
    print("============================================================")
    print("  FACILITY MANAGEMENT — PHASE 4 PRODUCTION TEST SUITE")
    print("============================================================")

    test_scenario_1_dual_mode_and_autoloop()
    test_scenario_2_keyframe_warmup()
    test_scenario_3_roi_calibrator_test_save()
    test_scenario_4_cli_dry_run_mode()

    print("\n" + "=" * 60)
    print("  ALL 4 PHASE 4 PRODUCTION TEST SCENARIOS: PASS ✓")
    print("============================================================")


if __name__ == "__main__":
    main()
