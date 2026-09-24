from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from engine.config_loader import CameraConfig
from engine.logger import get_logger

logger = get_logger(__name__)
    
DEFAULT_THRESHOLD_PCT: float = 80.0
DEFAULT_THRESHOLD_GB: float = 5.0
DEFAULT_POLL_INTERVAL: float = 60.0


class DiskGuard(threading.Thread):
    def __init__(
        self,
        camera_configs: list[CameraConfig],
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL,
        global_threshold_pct: float = DEFAULT_THRESHOLD_PCT,
        global_threshold_gb: float = DEFAULT_THRESHOLD_GB,
        base_dir: str | Path = ".",
    ) -> None:
        super().__init__(name="disk-guard", daemon=True)
        self.camera_configs = camera_configs
        self.poll_interval = poll_interval_sec
        self.global_threshold_pct = global_threshold_pct
        self.global_threshold_gb = global_threshold_gb
        self.base_dir = Path(base_dir).resolve()
        self._stop_event = threading.Event()
        self._total_purged: int = 0

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def total_purged(self) -> int:
        return self._total_purged
    def run(self) -> None:
        logger.debug(
            f"[DiskGuard] Dimulai — interval={self.poll_interval}s, "
            f"threshold_pct={self.global_threshold_pct}%, "
            f"threshold_gb={self.global_threshold_gb}GB"
        )

        while not self._stop_event.is_set():
            try:
                self._check_all()
            except Exception as exc:
                logger.error(f"[DiskGuard] Error saat pengecekan: {exc}", exc_info=True)

            self._stop_event.wait(timeout=self.poll_interval)

        logger.debug(f"[DiskGuard] Berhenti. Total file dipurge: {self._total_purged}")

    def _check_all(self) -> None:
        disk_pct = self._get_disk_usage_pct()
        if disk_pct is not None:
            logger.debug(f"[DiskGuard] Disk usage: {disk_pct:.1f}%")
            if disk_pct >= self.global_threshold_pct:
                logger.warning(
                    f"[DiskGuard] Disk usage {disk_pct:.1f}% >= threshold "
                    f"{self.global_threshold_pct}% — trigger purge semua kamera"
                )
                for cfg in self.camera_configs:
                    self._purge_camera(cfg, reason="disk_usage_global")
                return  

        for cfg in self.camera_configs:
            snapshot_dir = self.base_dir / cfg.disk_guard.snapshot_dir
            if not snapshot_dir.exists():
                continue

            size_gb = self._get_dir_size_gb(snapshot_dir)
            logger.debug(f"[{cfg.camera_id}] Snapshot size: {size_gb:.3f} GB / {cfg.disk_guard.max_size_gb} GB")

            if size_gb >= cfg.disk_guard.max_size_gb:
                logger.warning(
                    f"[{cfg.camera_id}] Snapshot {size_gb:.3f} GB >= "
                    f"limit {cfg.disk_guard.max_size_gb} GB — trigger purge"
                )
                self._purge_camera(cfg, reason="snapshot_size")

    def _purge_camera(self, cfg: CameraConfig, reason: str) -> None:
        snapshot_dir = self.base_dir / cfg.disk_guard.snapshot_dir
        if not snapshot_dir.exists():
            logger.debug(f"[{cfg.camera_id}] Snapshot dir tidak ada: {snapshot_dir}")
            return

        files = sorted(
            [f for f in snapshot_dir.iterdir() if f.is_file()],
            key=lambda f: f.stat().st_mtime,
        )

        if not files:
            return

        n_to_purge = max(1, int(len(files) * cfg.disk_guard.purge_oldest_pct / 100.0))
        to_purge = files[:n_to_purge]

        purged_count = 0
        purged_size_bytes = 0
        for f in to_purge:
            try:
                purged_size_bytes += f.stat().st_size
                f.unlink()
                purged_count += 1
            except OSError as exc:
                logger.warning(f"[{cfg.camera_id}] Gagal menghapus {f.name}: {exc}")

        self._total_purged += purged_count
        purged_mb = purged_size_bytes / (1024 ** 2)

        logger.info(
            f"[{cfg.camera_id}] DiskGuard purge selesai — "
            f"dihapus {purged_count} file ({purged_mb:.2f} MB), "
            f"alasan: {reason}"
        )

        self._log_purge_event(cfg, purged_count, purged_mb, reason)

    def _log_purge_event(
        self, cfg: CameraConfig, count: int, size_mb: float, reason: str
    ) -> None:
        try:
            from storage.database import get_writer_conn
            from storage.models import LifecycleEvent

            conn = get_writer_conn()
            event = LifecycleEvent(
                camera_id=cfg.camera_id,
                zone_id="__disk_guard__",
                event_type="disk_purge",
                notes=f"reason={reason}, files={count}, size_mb={size_mb:.2f}",
            )
            event.insert(conn)
        except Exception as exc:
            logger.debug(f"[DiskGuard] Gagal log ke DB: {exc}")

    # ── Disk Metrics ──────────────────────────────────────────────────────────

    def _get_disk_usage_pct(self) -> Optional[float]:
        try:
            usage = shutil.disk_usage(self.base_dir)
            return (usage.used / usage.total) * 100.0
        except OSError as exc:
            logger.warning(f"[DiskGuard] Gagal baca disk usage: {exc}")
            return None

    def _get_dir_size_gb(self, directory: Path) -> float:
        try:
            total_bytes = sum(
                f.stat().st_size
                for f in directory.rglob("*")
                if f.is_file()
            )
            return total_bytes / (1024 ** 3)
        except OSError as exc:
            logger.warning(f"[DiskGuard] Gagal hitung ukuran {directory}: {exc}")
            return 0.0
