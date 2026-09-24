from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from engine.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlertPacket:

    camera_id: str
    zone_id: str
    event_type: str  
    track_id: Optional[int] = None
    class_label: Optional[str] = None
    dwell_sec: float = 0.0
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    snapshot_path: Optional[str] = None


class AlertDispatcher(threading.Thread):
    def __init__(
        self,
        queue_maxsize: int = 30,
        cooldown_sec: float = 5.0,
        enable_audio: bool = True,
    ) -> None:
        super().__init__(name="AlertDispatcherWorker", daemon=True)
        self.queue_maxsize = queue_maxsize
        self.cooldown_sec = cooldown_sec
        self.enable_audio = enable_audio

        self._queue: queue.Queue[AlertPacket] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._last_alert_time: Dict[Tuple[str, str, str], float] = {}
        self._sinks: List[Callable[[AlertPacket], None]] = []

        self.total_dispatched: int = 0
        self.total_dropped: int = 0
        self.total_cooldown_suppressed: int = 0

    def add_sink(self, sink_func: Callable[[AlertPacket], None]) -> None:
        self._sinks.append(sink_func)

    def dispatch(self, alert: AlertPacket) -> None:
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.total_dropped += 1
            except queue.Empty:
                pass
            self._queue.put_nowait(alert)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                alert = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            now = time.time()
            key = (alert.camera_id, alert.zone_id, alert.event_type)
            last_time = self._last_alert_time.get(key, 0.0)

            # Evaluasi Cooldown
            if (now - last_time) < self.cooldown_sec:
                self.total_cooldown_suppressed += 1
                logger.debug(
                    f"[AlertDispatcher] Alert ditekan oleh cooldown ({alert.camera_id}:{alert.event_type})"
                )
                continue

            self._last_alert_time[key] = now
            self.total_dispatched += 1

            # Log terstruktur
            if alert.event_type in ("LINE_CROSSING", "BARRIER_BREACH"):
                cross_logger = get_logger("CROSSING")
                cross_logger.info(f"[{alert.camera_id}:{alert.zone_id}] {alert.message} ✓")
            elif alert.event_type == "WALKWAY_VIOLATION":
                walk_logger = get_logger("WALKWAY")
                walk_logger.warning(f"[{alert.camera_id}:{alert.zone_id}] {alert.message}")
            elif alert.event_type == "CONGESTION_ALERT":
                congest_logger = get_logger("CONGEST")
                congest_logger.warning(f"[{alert.camera_id}:{alert.zone_id}] {alert.message}")
            else:
                log_msg = (
                    f"[{alert.camera_id}:{alert.zone_id}] {alert.event_type.upper()} "
                    f"Track #{alert.track_id} ({alert.class_label or 'object'}) "
                    f"Dwell: {alert.dwell_sec:.1f}s - {alert.message}"
                )
                if alert.event_type == "linger":
                    logger.warning(log_msg)
                else:
                    logger.info(log_msg)

            # Audio Alert dengan Silent Fallback (tidak boleh crash jika headless/tanpa audio driver)
            if self.enable_audio and alert.event_type in ("linger", "enter"):
                self._trigger_audio_fallback(alert.event_type)

            # Teruskan ke sink eksternal
            for sink in self._sinks:
                try:
                    sink(alert)
                except Exception as exc:
                    logger.error(f"[AlertDispatcher] Error pada sink alert: {exc}", exc_info=True)

    def _trigger_audio_fallback(self, event_type: str) -> None:
        try:
            import winsound

            freq = 1200 if event_type == "linger" else 800
            duration = 300 if event_type == "linger" else 150
            winsound.Beep(freq, duration)
        except Exception as e:
            # Lingkungan headless / Linux / audio service tidak aktif
            logger.debug(f"[AlertDispatcher] Audio silent fallback dipicu: {e}")
