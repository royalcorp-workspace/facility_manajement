"""
Modul Motion Gate untuk Facility Management Vision Engine.
Melakukan micro-diffing antar frame AI (640x360) untuk menekan beban inferensi DNN.
Menjamin eksekusi periodik melalui mekanisme heartbeat invarian.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    """
    Frame Differencing Engine dengan Heartbeat Invarian.

    Menentukan apakah frame AI perlu dilewatkan ke model DNN atau di-bypass.
    """

    def __init__(
        self,
        pixel_threshold: int = 25,
        min_changed_pixels_pct: float = 0.005,
        heartbeat_interval_sec: float = 2.0,
    ) -> None:
        """
        Inisialisasi MotionGate.

        Args:
            pixel_threshold: Ambang batas perbedaan absolut nilai gray (0-255).
            min_changed_pixels_pct: Persentase minimal piksel yang berubah terhadap total piksel.
            heartbeat_interval_sec: Interval maksimal (detik) tanpa inferensi sebelum heartbeat dipicu.
        """
        self.pixel_threshold = pixel_threshold
        self.min_changed_pixels_pct = min_changed_pixels_pct
        self.heartbeat_interval_sec = heartbeat_interval_sec

        self._prev_gray: np.ndarray | None = None
        self._last_processed_time: float = 0.0

    def reset(self) -> None:
        """Reset state internal MotionGate (misal saat koneksi kamera terputus)."""
        self._prev_gray = None
        self._last_processed_time = 0.0

    def should_process(self, ai_frame: np.ndarray, timestamp: float) -> tuple[bool, str]:
        """
        Evaluasi apakah frame AI perlu diinferensi.

        Args:
            ai_frame: Frame canvas AI (BGR, 640x360).
            timestamp: Timestamp POSIX saat ini (detik).

        Returns:
            Tuple (should_process, reason):
            - (True, "initial"): Frame pertama setelah reset.
            - (True, "motion"): Gerakan terdeteksi di atas threshold.
            - (True, "heartbeat"): Heartbeat 2.0s tercapai.
            - (False, "suppressed"): Frame statis & heartbeat belum tercapai.
        """
        # Konversi ke Grayscale untuk frame diff
        if len(ai_frame.shape) == 3 and ai_frame.shape[2] == 3:
            gray = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = ai_frame

        # Frame pertama: selalu diproses
        if self._prev_gray is None:
            self._prev_gray = gray
            self._last_processed_time = timestamp
            return True, "initial"

        # Cek Heartbeat
        elapsed_sec = timestamp - self._last_processed_time
        if elapsed_sec >= self.heartbeat_interval_sec:
            self._prev_gray = gray
            self._last_processed_time = timestamp
            return True, "heartbeat"

        # Frame Differencing
        frame_diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = gray.shape[0] * gray.shape[1]
        changed_pct = changed_pixels / total_pixels

        if changed_pct >= self.min_changed_pixels_pct:
            self._prev_gray = gray
            self._last_processed_time = timestamp
            return True, "motion"

        # Frame statis dan heartbeat belum tercapai
        return False, "suppressed"
