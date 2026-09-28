from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(
        self,
        pixel_threshold: int = 25,
        min_changed_pixels_pct: float = 0.005,
        heartbeat_interval_sec: float = 2.0,
    ) -> None:
        self.pixel_threshold = pixel_threshold
        self.min_changed_pixels_pct = min_changed_pixels_pct
        self.heartbeat_interval_sec = heartbeat_interval_sec

        self._prev_gray: np.ndarray | None = None
        self._last_processed_time: float = 0.0

    def reset(self) -> None:
        self._prev_gray = None
        self._last_processed_time = 0.0

    def should_process(self, ai_frame: np.ndarray, timestamp: float) -> tuple[bool, str]:
        if len(ai_frame.shape) == 3 and ai_frame.shape[2] == 3:
            gray = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = ai_frame

        if self._prev_gray is None:
            self._prev_gray = gray
            self._last_processed_time = timestamp
            return True, "initial"

        elapsed_sec = timestamp - self._last_processed_time
        if elapsed_sec >= self.heartbeat_interval_sec:
            self._prev_gray = gray
            self._last_processed_time = timestamp
            return True, "heartbeat"

        frame_diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = gray.shape[0] * gray.shape[1]
        changed_pct = changed_pixels / total_pixels

        if changed_pct >= self.min_changed_pixels_pct:
            self._prev_gray = gray
            self._last_processed_time = timestamp
            return True, "motion"

        return False, "suppressed"
