"""
engine/tracker_interface.py
============================
Abstrak kontrak TrackerBase — Interface untuk multi-object tracker.

STATUS: ACTIVE — Terintegrasi dengan CentroidTracker & Spatial Memory Re-ID.

Desain:
  - TrackerBase adalah ABC yang mendefinisikan kontrak wajib
  - Input: list[DetectionResult] dari detector
  - Output: list[TrackResult] dengan track_id persisten antar frame
  - Track ID bersifat unik dan persisten selama objek tertrack

Contoh penggunaan Fase 2:
    class ByteTracker(TrackerBase):
        def update(self, detections: list[DetectionResult], ...) -> list[TrackResult]: ...
        def reset(self) -> None: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from engine.detector_interface import DetectionResult


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONTRACT: TrackResult
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrackResult:
    """
    Hasil tracking satu objek dengan track ID persisten.

    track_id     : ID unik, persisten selama objek tertrack (tidak berubah antar frame)
    bbox         : Bounding box ter-smoothed oleh tracker (AI canvas coords)
    confidence   : Confidence skor (dari detector atau interpolasi tracker)
    class_label  : Label kelas objek
    class_id     : ID kelas numerik
    age          : Jumlah frame sejak track ini pertama kali terlihat
    is_confirmed : True jika track sudah melewati minimum hit threshold
    camera_id    : ID kamera asal
    frame_number : Nomor frame saat ini
    """

    track_id: int
    bbox: tuple[float, float, float, float]    # (x1, y1, x2, y2) AI canvas coords
    confidence: float
    class_label: str
    class_id: int
    age: int = 0
    is_confirmed: bool = True
    camera_id: str = ""
    frame_number: int = 0
    prev_centroid: Optional[tuple[float, float]] = None

    @property
    def current_centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @classmethod
    def from_detection(cls, det: DetectionResult, track_id: int, **kwargs) -> "TrackResult":
        """Buat TrackResult dari DetectionResult (untuk tracker sederhana)."""
        return cls(
            track_id=track_id,
            bbox=det.bbox,
            confidence=det.confidence,
            class_label=det.class_label,
            class_id=det.class_id,
            camera_id=det.camera_id,
            frame_number=det.frame_number,
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS: TrackerBase
# ═══════════════════════════════════════════════════════════════════════════════


class TrackerBase(ABC):
    """
    Kontrak abstrak untuk semua multi-object tracker.

    Implementasi Fase 2 WAJIB mengoverride semua @abstractmethod.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            max_age      : Jumlah frame maksimum track dipertahankan tanpa deteksi
            min_hits     : Minimum deteksi sebelum track dianggap "confirmed"
            iou_threshold: Threshold IoU untuk assignment deteksi ke track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._frame_count: int = 0

    @abstractmethod
    def update(
        self,
        detections: list[DetectionResult],
        frame_number: int = 0,
    ) -> list[TrackResult]:
        """
        Update state tracker dengan deteksi frame saat ini.

        Args:
            detections  : Hasil deteksi dari detector untuk frame ini
            frame_number: Nomor frame (untuk logging dan TrackResult)

        Returns:
            List TrackResult — satu per objek yang sedang di-track.
            Hanya mengembalikan track yang "active" (tidak lost/deleted).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset state tracker (semua track dihapus).
        Dipanggil saat reconnect kamera atau restart engine.
        """
        ...

    def coast(
        self,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> list[TrackResult]:
        """
        Lanjutkan tracking pada frame di mana inferensi DNN dilewati (staggered AI coasting).
        Mempertahankan posisi bbox dan dwell tracking tanpa menaikkan disappeared count.
        """
        return []

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_age={self.max_age}, "
            f"min_hits={self.min_hits}, "
            f"iou_threshold={self.iou_threshold})"
        )
