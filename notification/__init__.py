"""
Package notification untuk Facility Management.
Menyediakan AlertDispatcher non-blocking, AlertPacket, dan helper send_alert.
"""

from __future__ import annotations

import logging
from typing import Optional

from notification.dispatcher import AlertDispatcher, AlertPacket

logger = logging.getLogger(__name__)

_global_dispatcher: Optional[AlertDispatcher] = None


def set_global_dispatcher(dispatcher: AlertDispatcher) -> None:
    """Set instance global AlertDispatcher."""
    global _global_dispatcher
    _global_dispatcher = dispatcher


def get_global_dispatcher() -> Optional[AlertDispatcher]:
    """Dapatkan instance global AlertDispatcher."""
    return _global_dispatcher


def send_alert(
    camera_id: str,
    event_type: str,
    message: str,
    zone_id: str = "default",
    track_id: Optional[int] = None,
    class_label: Optional[str] = None,
    dwell_sec: float = 0.0,
    snapshot_path: Optional[str] = None,
    **kwargs,
) -> None:
    """Kirim notifikasi alert via AlertDispatcher non-blocking."""
    packet = AlertPacket(
        camera_id=camera_id,
        zone_id=zone_id,
        event_type=event_type,
        track_id=track_id,
        class_label=class_label,
        dwell_sec=dwell_sec,
        message=message,
        snapshot_path=snapshot_path,
    )
    if _global_dispatcher is not None:
        _global_dispatcher.dispatch(packet)
    else:
        logger.info(f"[ALERT FALLBACK] [{camera_id}:{zone_id}] {event_type}: {message}")


__all__ = ["AlertDispatcher", "AlertPacket", "send_alert", "set_global_dispatcher", "get_global_dispatcher"]
