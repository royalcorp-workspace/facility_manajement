from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from engine.capture import FramePacket
from engine.config_loader import CameraConfig, ResolutionConfig
from engine.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONTAINER: ProcessedFrame
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProcessedFrame:
    """
    Frame setelah preprocessing — dua buffer independen.

    raw_frame   : Digunakan untuk display MJPEG / snapshot (full-res)
    ai_frame    : Digunakan sebagai input inferensi AI (resize-down, uint8 BGR)
    scale_x/y   : Faktor skala untuk memetakan bounding box AI → koordinat raw
    """

    raw_frame: np.ndarray          # Shape: (H_capture, W_capture, 3) BGR
    ai_frame: np.ndarray           # Shape: (H_ai, W_ai, 3) BGR
    camera_id: str
    timestamp: float
    frame_number: int
    scale_x: float                 # W_capture / W_ai
    scale_y: float                 # H_capture / H_ai
    is_warmup: bool = False

    def map_bbox_to_raw(
        self,
        x1: float, y1: float, x2: float, y2: float,
    ) -> tuple[int, int, int, int]:
        """
        Konversi bounding box dari koordinat AI canvas → koordinat raw frame.
        Berguna untuk menggambar overlay pada stream display.
        """
        return (
            int(x1 * self.scale_x),
            int(y1 * self.scale_y),
            int(x2 * self.scale_x),
            int(y2 * self.scale_y),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSOR WORKER
# ═══════════════════════════════════════════════════════════════════════════════


class PreprocessorWorker(threading.Thread):
    """
    Thread worker yang mengkonsumsi FramePacket dari capture queue
    dan menghasilkan ProcessedFrame ke processing queue.

    Pipeline:
        RTSPCaptureWorker → [frame_queue] → PreprocessorWorker → [proc_queue] → Detector (Fase 2)
    """

    def __init__(
        self,
        config: CameraConfig,
        frame_queue: queue.Queue,
        proc_queue: queue.Queue,
        frame_timeout: float = 1.0,
        initial_warmup_frames: int = 30,
    ) -> None:
        super().__init__(name=f"preproc-{config.camera_id}", daemon=True)
        self.config = config
        self.frame_queue = frame_queue
        self.proc_queue = proc_queue
        self._frame_timeout = frame_timeout
        self._stop_event = threading.Event()
        self._processed_count: int = 0
        self.initial_warmup_frames = initial_warmup_frames
        self._res = config.resolution

    # ── Public API ────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def frame_count(self) -> int:
        return self._processed_count

    # ── Internal ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        cam_id = self.config.camera_id
        logger.debug(f"[{cam_id}] Preprocessor worker dimulai.")

        while not self._stop_event.is_set():
            try:
                packet: FramePacket = self.frame_queue.get(timeout=self._frame_timeout)
            except queue.Empty:
                continue

            try:
                processed = self._preprocess(packet)
                self._emit(processed)
                self._processed_count += 1
            except Exception as exc:
                logger.error(f"[{cam_id}] Error preprocessing frame #{packet.frame_number}: {exc}")

        logger.debug(f"[{cam_id}] Preprocessor selesai. Total diproses: {self._processed_count}")

    def _preprocess(self, packet: FramePacket) -> ProcessedFrame:
        """
        Resize raw_frame → ai_frame secara independen.
        Kedua buffer menjaga data terpisah (no shared memory).
        """
        raw = packet.raw_frame
        res = self._res

        # Resize ke AI canvas — INTER_LINEAR: trade-off kecepatan/kualitas optimal
        ai_frame = cv2.resize(
            raw,
            (res.ai_width, res.ai_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Skala referensi koordinat capture (1080p) ke AI canvas
        ref_w = float(res.capture_width or raw.shape[1])
        ref_h = float(res.capture_height or raw.shape[0])
        scale_x = ref_w / res.ai_width   
        scale_y = ref_h / res.ai_height   

        is_warmup = (self._processed_count < self.initial_warmup_frames)
        return ProcessedFrame(
            raw_frame=raw,
            ai_frame=ai_frame,
            camera_id=packet.camera_id,
            timestamp=packet.timestamp,
            frame_number=packet.frame_number,
            scale_x=scale_x,
            scale_y=scale_y,
            is_warmup=is_warmup,
        )

    def _emit(self, processed: ProcessedFrame) -> None:
        """
        Kirim ke proc_queue dengan Drop-on-Full (sama seperti capture queue).
        Detector Fase 2 boleh lambat — frame baru selalu menimpa yang lama.
        """
        try:
            self.proc_queue.put_nowait(processed)
        except queue.Full:
            try:
                self.proc_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.proc_queue.put_nowait(processed)
            except queue.Full:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def create_preprocessor_pipeline(
    config: CameraConfig,
    frame_queue: queue.Queue,
    initial_warmup_frames: int = 30,
) -> tuple[PreprocessorWorker, queue.Queue]:
    """
    Buat preprocessor worker dan output queue.

    Returns:
        (worker, proc_queue) — worker belum di-start.
    """
    proc_queue: queue.Queue = queue.Queue(maxsize=1)  # Drop-on-Full
    worker = PreprocessorWorker(
        config=config,
        frame_queue=frame_queue,
        proc_queue=proc_queue,
        initial_warmup_frames=initial_warmup_frames,
    )
    return worker, proc_queue
