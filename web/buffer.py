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


def generate_fallback_canvas(
    camera_id: str,
    message: str = "CONNECTING / RECONNECTING...",
    width: int = 640,
    height: int = 360,
    quality: int = 70,
) -> bytes:
    """
    Menghasilkan canvas JPEG informatif saat stream kamera belum siap atau terputus/reconnecting.
    Mencegah pemutusan koneksi HTTP streaming client.
    """
    try:
        import cv2
        import numpy as np
        from datetime import datetime

        # Dark high-contrast background (#0f172a / BGR: 42, 23, 15)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (42, 23, 15)

        # Frame border cyan halus
        cv2.rectangle(img, (2, 2), (width - 3, height - 3), (230, 216, 0), 1)

        # Header: Camera ID badge
        header_text = f"[{camera_id.upper()}] LIVE STREAM"
        cv2.putText(
            img,
            header_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 216, 0),
            2,
            cv2.LINE_AA,
        )

        # Center: Warning / Status message
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        tx = max(20, (width - text_size[0]) // 2)
        ty = (height // 2)
        cv2.putText(
            img,
            message,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),  # Amber warning
            2,
            cv2.LINE_AA,
        )

        # Sub-text: Wait advice
        sub_text = "Retrying RTSP connection automatically. Standby..."
        sub_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        stx = max(20, (width - sub_size[0]) // 2)
        cv2.putText(
            img,
            sub_text,
            (stx, ty + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # Footer: Timestamp
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            img,
            f"HUB TIME: {ts_str}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ret:
            return buf.tobytes()
    except Exception:
        pass
    # Fallback minimal 1x1 JPEG jika cv2/numpy terkendala
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08"
        b"\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13"
        b"\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff"
        b"\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01"
        b"\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
        b"\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
    )


class MultiCameraBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frames: Dict[str, FrameItem] = {}

    def register_camera(
        self,
        camera_id: str,
        initial_frame: Optional[FrameItem] = None,
    ) -> None:
        """
        Daftarkan camera_id ke buffer sejak auto-discovery aktif.
        Jika initial_frame tidak diberikan, fallback canvas otomatis dibuatkan.
        """
        with self._condition:
            if camera_id not in self._frames:
                if initial_frame is None:
                    fb_bytes = generate_fallback_canvas(camera_id, "INITIALIZING STREAM...")
                    initial_frame = FrameItem(
                        jpeg_bytes=fb_bytes,
                        timestamp=time.time(),
                        sequence_id=0,
                    )
                self._frames[camera_id] = initial_frame
                self._condition.notify_all()

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
