from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FrameItem:

    jpeg_bytes: bytes
    timestamp: float
    sequence_id: int


class MultiCameraBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frames: Dict[str, FrameItem] = {}

    def set_frame(
        self,
        camera_id: str,
        jpeg_bytes: bytes,
        timestamp: float,
        sequence_id: int,
    ) -> None:

        item = FrameItem(
            jpeg_bytes=jpeg_bytes,
            timestamp=timestamp,
            sequence_id=sequence_id,
        )
        with self._condition:
            self._frames[camera_id] = item
            self._condition.notify_all()

    def get_latest_frame(self, camera_id: str) -> Optional[FrameItem]:
        with self._condition:
            return self._frames.get(camera_id)

    def wait_for_next_frame(
        self,
        camera_id: str,
        last_sequence_id: int,
        timeout: float = 1.0,
    ) -> Optional[FrameItem]:

        end_time = time.monotonic() + timeout
        with self._condition:
            while True:
                item = self._frames.get(camera_id)
                if item is not None and item.sequence_id != last_sequence_id:
                    return item

                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    return item

                # Tunggu notifikasi frame baru
                self._condition.wait(timeout=min(remaining, 0.5))

    def get_registered_cameras(self) -> List[str]:
        with self._condition:
            return list(self._frames.keys())

    def clear(self) -> None:
        with self._condition:
            self._frames.clear()
            self._condition.notify_all()
