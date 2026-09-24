from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ── Event Types ─────────────────────────────────────────────────────────────
EVENT_ENTER = "enter"
EVENT_LINGER = "linger"
EVENT_EXIT = "exit"
EVENT_LINE_CROSSING = "LINE_CROSSING"
EVENT_WALKWAY_VIOLATION = "WALKWAY_VIOLATION"
EVENT_CONGESTION_ALERT = "CONGESTION_ALERT"


@dataclass
class LifecycleEvent:

    camera_id: str
    zone_id: str
    event_type: str                      
    timestamp: str = field(default_factory=_now_iso)
    track_id: Optional[int] = None
    class_label: Optional[str] = None
    confidence: Optional[float] = None
    snapshot_path: Optional[str] = None
    is_resolved: int = 0                 
    resolved_time: Optional[str] = None   
    notes: Optional[str] = None
    id: Optional[int] = None              

    _INSERT_SQL = """
        INSERT INTO lifecycle_events
            (camera_id, zone_id, event_type, track_id, class_label,
             confidence, timestamp, snapshot_path,
             is_resolved, resolved_time, notes)
        VALUES
            (:camera_id, :zone_id, :event_type, :track_id, :class_label,
             :confidence, :timestamp, :snapshot_path,
             :is_resolved, :resolved_time, :notes)
    """

    _RESOLVE_SQL = """
        UPDATE lifecycle_events
        SET is_resolved = 1,
            resolved_time = :resolved_time,
            notes = COALESCE(:notes, notes)
        WHERE id = :id
    """

    def insert(self, conn: sqlite3.Connection) -> int:
        """INSERT event ke DB, return rowid."""
        cursor = conn.execute(
            self._INSERT_SQL,
            {
                "camera_id": self.camera_id,
                "zone_id": self.zone_id,
                "event_type": self.event_type,
                "track_id": self.track_id,
                "class_label": self.class_label,
                "confidence": self.confidence,
                "timestamp": self.timestamp,
                "snapshot_path": self.snapshot_path,
                "is_resolved": self.is_resolved,
                "resolved_time": self.resolved_time,
                "notes": self.notes,
            },
        )
        self.id = cursor.lastrowid
        return self.id

    def resolve(self, conn: sqlite3.Connection, notes: Optional[str] = None) -> None:
        if self.id is None:
            raise ValueError("Event belum di-INSERT ke database (id=None).")
        resolved_time = _now_iso()
        conn.execute(
            self._RESOLVE_SQL,
            {"id": self.id, "resolved_time": resolved_time, "notes": notes},
        )
        self.is_resolved = 1
        self.resolved_time = resolved_time
        self.notes = notes or self.notes



@dataclass
class CameraRegistry:

    camera_id: str
    display_name: str
    status: str = "inactive"            
    last_seen: Optional[str] = None

    _UPSERT_SQL = """
        INSERT INTO camera_registry (camera_id, display_name, status, last_seen)
        VALUES (:camera_id, :display_name, :status, :last_seen)
        ON CONFLICT(camera_id) DO UPDATE SET
            display_name = excluded.display_name,
            status       = excluded.status,
            last_seen    = excluded.last_seen
    """

    _UPDATE_STATUS_SQL = """
        UPDATE camera_registry
        SET status = :status, last_seen = :last_seen
        WHERE camera_id = :camera_id
    """

    def upsert(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            self._UPSERT_SQL,
            {
                "camera_id": self.camera_id,
                "display_name": self.display_name,
                "status": self.status,
                "last_seen": self.last_seen or _now_iso(),
            },
        )

    def update_status(self, conn: sqlite3.Connection, status: str) -> None:
        self.status = status
        self.last_seen = _now_iso()
        conn.execute(
            self._UPDATE_STATUS_SQL,
            {
                "camera_id": self.camera_id,
                "status": self.status,
                "last_seen": self.last_seen,
            },
        )

def get_unresolved_events(
    conn: sqlite3.Connection,
    camera_id: Optional[str] = None,
    limit: int = 100,
) -> list[sqlite3.Row]:

    if camera_id:
        return conn.execute(
            "SELECT * FROM lifecycle_events "
            "WHERE is_resolved = 0 AND camera_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (camera_id, limit),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM lifecycle_events "
        "WHERE is_resolved = 0 "
        "ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
