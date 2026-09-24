"""
Modul Video Capture untuk Facility Management.
Mendukung Dual-Mode Video Source (RTSP live & berkas video lokal dengan auto-looping),
keyframe warmup safeguard, watchdog capture timeout, dan drop-on-full queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from engine.config_loader import CameraConfig
from engine.logger import get_logger, log_error

logger = get_logger(__name__)


@dataclass
class FramePacket:
    """Kontainer data frame raw tertangkap dari kamera."""

    raw_frame: np.ndarray
    camera_id: str
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0


class RTSPCaptureWorker(threading.Thread):
    """
    Worker capture thread dengan dukungan Dual-Mode (RTSP stream & File simulasi),
    auto-looping EOF, keyframe warmup discard, dan watchdog timeout.
    """

    def __init__(self, config: CameraConfig, frame_queue: queue.Queue) -> None:
        super().__init__(name=f"capture-{config.camera_id}", daemon=True)
        self.config = config
        self.frame_queue = frame_queue
        self._stop_event = threading.Event()
        self._frame_count: int = 0
        self._reconnect_count: int = 0
        self._is_local_file: bool = False

        # Parameter capture dari config
        self.frame_timeout_sec: float = getattr(config.capture, "frame_timeout_sec", 8.0)
        self.keyframe_warmup_frames: int = getattr(config.capture, "keyframe_warmup_frames", 5)

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and self.is_alive()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def run(self) -> None:
        cam_id = self.config.camera_id
        source = self.config.rtsp_url
        cfg = self.config.capture

        # Deteksi sumber: apakah berkas lokal (.mp4, .avi, etc) atau URL live RTSP/HTTP
        self._is_local_file = Path(source).exists() or not (
            source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://")
        )
        mode_str = "BERKAS LOKAL (Auto-Looping)" if self._is_local_file else "RTSP LIVE"
        logger.debug(f"[{cam_id}] Capture worker dimulai dalam mode {mode_str}.")

        while not self._stop_event.is_set():
            if self._reconnect_count >= cfg.max_reconnect_attempts:
                logger.error(
                    f"[{cam_id}] Maksimum reconnect ({cfg.max_reconnect_attempts}x) "
                    f"tercapai. Worker berhenti."
                )
                break

            cap = self._open_capture(source, cam_id)
            if cap is None:
                self._wait_reconnect(cam_id, cfg.reconnect_delay_sec)
                continue

            self._reconnect_count = 0
            logger.debug(f"[{cam_id}] Stream terhubung ✓ ({mode_str})")

            # Keyframe warmup safeguard: lewati N frame awal
            warmup_discarded = 0
            last_frame_time = time.monotonic()

            while not self._stop_event.is_set():
                ret, frame = cap.read()

                # Penanganan EOF untuk berkas video lokal
                if (not ret or frame is None) and self._is_local_file:
                    logger.debug(f"[{cam_id}] EOF tercapai pada video lokal — me-reset posisi ke awal (auto-looping)...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                if not ret or frame is None:
                    logger.warning(f"[{cam_id}] Frame read gagal atau koneksi putus — reconnect...")
                    break

                now = time.monotonic()

                # Watchdog capture timeout safeguard
                if (now - last_frame_time) > self.frame_timeout_sec:
                    logger.warning(f"[{cam_id}] Watchdog timeout: tidak ada frame selama {self.frame_timeout_sec}s — reconnecting...")
                    break
                last_frame_time = now

                # Keyframe Warmup Safeguard
                if warmup_discarded < self.keyframe_warmup_frames:
                    warmup_discarded += 1
                    continue

                self._frame_count += 1
                packet = FramePacket(
                    raw_frame=frame,
                    camera_id=cam_id,
                    frame_number=self._frame_count,
                )
                self._enqueue(packet)

                # Untuk file lokal, tambahkan jeda fps alami agar tidak menghabiskan CPU 100%
                if self._is_local_file:
                    time.sleep(0.033)  # ~30 FPS

            cap.release()

            if not self._stop_event.is_set():
                self._reconnect_count += 1
                self._wait_reconnect(cam_id, cfg.reconnect_delay_sec)

        logger.debug(f"[{cam_id}] Capture worker selesai. Total frame: {self._frame_count}")

    def _open_capture(self, source: str, cam_id: str) -> Optional[cv2.VideoCapture]:
        logger.debug(f"[{cam_id}] Membuka video capture source...")

        if self._is_local_file:
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            log_error(
                logger,
                f"[{cam_id}] Gagal membuka stream video (attempt #{self._reconnect_count + 1})",
                mitigation_hint="Periksa IP, port 554, atau kredensial kamera di .env",
            )
            cap.release()
            return None

        return cap

    def _enqueue(self, packet: FramePacket) -> None:
        """Memasukkan frame ke queue dengan kebijakan drop-on-full O(1)."""
        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(packet)
            except queue.Full:
                pass

    def _wait_reconnect(self, cam_id: str, delay_sec: float) -> None:
        backoff = min(delay_sec * (1.5 ** min(self._reconnect_count, 4)), 30.0)
        logger.info(f"[{cam_id}] Reconnect dalam {backoff:.1f} detik... (attempt #{self._reconnect_count})")
        self._stop_event.wait(timeout=backoff)


def create_capture_pipeline(config: CameraConfig) -> tuple[RTSPCaptureWorker, queue.Queue]:
    frame_queue: queue.Queue = queue.Queue(maxsize=config.capture.queue_maxsize)
    worker = RTSPCaptureWorker(config=config, frame_queue=frame_queue)
    return worker, frame_queue
