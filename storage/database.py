from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from engine.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = Path("storage/facility.db")

_PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA foreign_keys=ON;",
    "PRAGMA synchronous=NORMAL;",  
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA mmap_size=134217728;",  
]


_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS lifecycle_events (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id        TEXT    NOT NULL,
        zone_id          TEXT    NOT NULL,
        event_type       TEXT    NOT NULL,
        track_id         INTEGER,
        class_label      TEXT,
        confidence       REAL,
        timestamp        TEXT    NOT NULL,
        snapshot_path    TEXT,
        is_resolved      INTEGER NOT NULL DEFAULT 0,
        resolved_time    TEXT,
        notes            TEXT
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_lifecycle_camera_id
        ON lifecycle_events (camera_id, timestamp DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_lifecycle_unresolved
        ON lifecycle_events (is_resolved, camera_id)
        WHERE is_resolved = 0;
    """,
    """
    CREATE TABLE IF NOT EXISTS camera_registry (
        camera_id    TEXT PRIMARY KEY,
        display_name TEXT,
        status       TEXT NOT NULL DEFAULT 'inactive',
        last_seen    TEXT
    );
    """,
]

_writer_conn: sqlite3.Connection | None = None
_writer_lock = Lock()
_db_path: Path = DEFAULT_DB_PATH


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    global _writer_conn, _db_path

    _db_path = Path(db_path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    with _writer_lock:
        if _writer_conn is not None:
            logger.warning("init_db dipanggil lebih dari sekali — diabaikan.")
            return

        logger.debug(f"Menginisialisasi database: {_db_path}")
        conn = sqlite3.connect(
            str(_db_path),
            isolation_level=None,        
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row   

        for pragma in _PRAGMAS:
            conn.execute(pragma)

        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        if mode != "wal":
            logger.warning(f"journal_mode adalah '{mode}', bukan 'wal'. Cek permissions.")
        else:
            logger.debug("SQLite WAL mode aktif ✓")

        # Buat tabel
        for ddl in _DDL_STATEMENTS:
            conn.execute(ddl)

        _writer_conn = conn
        logger.debug("Database siap.")


def get_writer_conn() -> sqlite3.Connection:
    if _writer_conn is None:
        raise RuntimeError(
            "Database belum diinisialisasi. Panggil init_db() terlebih dahulu."
        )
    return _writer_conn


def get_writer_lock() -> Lock:
    """Dapatkan mutex lock untuk operasi tulis ke writer connection."""
    return _writer_lock



def get_reader_conn() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("Database belum diinisialisasi.")

    conn = sqlite3.connect(
        f"file:{_db_path}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


def close_db() -> None:
    global _writer_conn
    with _writer_lock:
        if _writer_conn is not None:
            _writer_conn.close()
            _writer_conn = None
            logger.debug("Database connection ditutup.")


def resolve_incident(
    incident_id: int,
    resolved_by: str = "Operator",
    notes: Optional[str] = None,
) -> bool:
    """
    Tandai insiden sebagai telah diselesaikan secara thread-safe menggunakan writer lock.
    Mengembalikan True jika insiden berhasil diselesaikan, False jika tidak ditemukan / sudah resolved.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    note_append = f"Resolved by {resolved_by}" if not notes else f"Resolved by {resolved_by}: {notes}"

    with _writer_lock:
        conn = get_writer_conn()
        cursor = conn.execute(
            """
            UPDATE lifecycle_events
            SET is_resolved = 1,
                resolved_time = :resolved_time,
                notes = CASE
                          WHEN notes IS NULL OR notes = '' THEN :note_append
                          ELSE notes || ' | ' || :note_append
                        END
            WHERE id = :id AND is_resolved = 0
            """,
            {
                "id": incident_id,
                "resolved_time": now_iso,
                "note_append": note_append,
            },
        )
        return cursor.rowcount > 0

