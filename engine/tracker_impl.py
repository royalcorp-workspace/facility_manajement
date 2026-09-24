"""
Implementasi Multi-Object Tracker dengan Spatial Memory & Re-ID untuk Facility Management.
Menggunakan Centroid & IoU Matching dengan Transient/Stationary State Machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.detector_interface import DetectionResult
from engine.geometry import bbox_centroid, bbox_iou
from engine.tracker_interface import TrackerBase, TrackResult


@dataclass
class SpatialMemoryItem:
    """Entry memori spasial untuk Re-ID objek yang mengalami oklusi sementara."""

    track_id: int
    class_label: str
    class_id: int
    spatial_anchor: Tuple[float, float]  # Centroid (cx, cy)
    accumulated_dwell_sec: float
    last_seen_time: float


@dataclass
class TrackState:
    """State internal pelacakan untuk satu objek."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_label: str
    class_id: int
    age: int = 0
    hits: int = 0
    disappeared: int = 0
    status: str = "transient"  # "transient" | "stationary" | "occluded"
    spatial_anchor: Tuple[float, float] = (0.0, 0.0)
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    accumulated_dwell_sec: float = 0.0
    prev_centroid: Optional[Tuple[float, float]] = None

    @property
    def centroid(self) -> Tuple[float, float]:
        return bbox_centroid(self.bbox)

    def to_track_result(self, frame_number: int = 0, camera_id: str = "") -> TrackResult:
        return TrackResult(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.confidence,
            class_label=self.class_label,
            class_id=self.class_id,
            age=self.age,
            is_confirmed=(self.hits >= 3 or self.status == "stationary"),
            camera_id=camera_id,
            frame_number=frame_number,
            prev_centroid=self.prev_centroid,
        )


class CentroidTracker(TrackerBase):
    """
    Centroid + IoU Tracker dengan Spatial Anchor Memory Re-ID.
    """

    def __init__(
        self,
        max_disappeared_frames: int = 30,
        max_distance_px: float = 60.0,
        anchor_radius_px: float = 15.0,
        enter_min_frames: int = 3,
        spatial_memory_ttl_sec: float = 3.0,
        iou_threshold: float = 0.3,
    ) -> None:
        super().__init__(max_age=max_disappeared_frames, min_hits=enter_min_frames, iou_threshold=iou_threshold)
        self.max_disappeared_frames = max_disappeared_frames
        self.max_distance_px = max_distance_px
        self.anchor_radius_px = anchor_radius_px
        self.enter_min_frames = enter_min_frames
        self.spatial_memory_ttl_sec = spatial_memory_ttl_sec

        self._next_track_id: int = 1
        self.tracks: Dict[int, TrackState] = {}
        self.spatial_memory: List[SpatialMemoryItem] = []

    def reset(self) -> None:
        """Reset state tracker sepenuhnya."""
        self._next_track_id = 1
        self.tracks.clear()
        self.spatial_memory.clear()
        self._frame_count = 0

    def coast(
        self,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> List[TrackResult]:
        """
        Lanjutkan tracking pada frame di mana inferensi DNN dilewati (staggered AI coasting).
        Mempertahankan posisi bbox dan dwell tracking tanpa menaikkan disappeared count.
        """
        self._frame_count += 1
        now = timestamp if timestamp is not None else time.time()

        # Bersihkan spatial memory yang kedaluwarsa
        self.spatial_memory = [
            mem for mem in self.spatial_memory
            if (now - mem.last_seen_time) <= self.spatial_memory_ttl_sec
        ]

        # Perbarui age dan dwell time untuk track aktif tanpa menaikkan disappeared
        for track in self.tracks.values():
            track.age += 1
            if now > track.last_seen_time:
                dwell_increment = now - track.last_seen_time
                track.accumulated_dwell_sec += dwell_increment
                track.last_seen_time = now

        return [
            t.to_track_result(frame_number)
            for t in self.tracks.values()
            if t.hits >= self.enter_min_frames or t.status == "stationary"
        ]

    def update(
        self,
        detections: List[DetectionResult],
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> List[TrackResult]:
        """
        Update state pelacakan dengan deteksi baru.
        """
        self._frame_count += 1
        now = timestamp if timestamp is not None else time.time()

        # Clean up expired spatial memory entries
        self.spatial_memory = [
            mem for mem in self.spatial_memory
            if (now - mem.last_seen_time) <= self.spatial_memory_ttl_sec
        ]

        active_track_ids = list(self.tracks.keys())

        # 1. Jika tidak ada deteksi baru, naikkan count disappeared
        if len(detections) == 0:
            for track_id in active_track_ids:
                track = self.tracks[track_id]
                track.disappeared += 1
                track.age += 1
                if track.disappeared >= self.max_disappeared_frames:
                    self._archive_to_spatial_memory(track, now)
                    del self.tracks[track_id]
            return [t.to_track_result(frame_number) for t in self.tracks.values() if t.hits >= self.enter_min_frames]

        # 2. Match active tracks dengan deteksi via IoU + Centroid Distance
        if len(self.tracks) == 0:
            # Tidak ada track aktif, coba Re-ID dari spatial memory dulu
            for det in detections:
                det_centroid = bbox_centroid(det.bbox)
                reidentified_mem = self._try_reid_spatial_memory(det_centroid, det.class_label, now)

                if reidentified_mem:
                    # Restorasi track ID dan dwell time lama
                    track = TrackState(
                        track_id=reidentified_mem.track_id,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_label=det.class_label,
                        class_id=det.class_id,
                        age=1,
                        hits=self.enter_min_frames,  # Langsung stationary/confirmed
                        disappeared=0,
                        status="stationary",
                        spatial_anchor=det_centroid,
                        first_seen_time=now - reidentified_mem.accumulated_dwell_sec,
                        last_seen_time=now,
                        accumulated_dwell_sec=reidentified_mem.accumulated_dwell_sec,
                    )
                    self.tracks[track.track_id] = track
                else:
                    # Track baru
                    track_id = self._next_track_id
                    self._next_track_id += 1
                    track = TrackState(
                        track_id=track_id,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_label=det.class_label,
                        class_id=det.class_id,
                        age=1,
                        hits=1,
                        disappeared=0,
                        status="transient",
                        spatial_anchor=det_centroid,
                        first_seen_time=now,
                        last_seen_time=now,
                        accumulated_dwell_sec=0.0,
                    )
                    self.tracks[track_id] = track

            return [t.to_track_result(frame_number) for t in self.tracks.values() if t.hits >= self.enter_min_frames]

        # Cost matrix matching: Matriks jarak centroid + (1.0 - IoU)
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]
        det_centroids = [bbox_centroid(d.bbox) for d in detections]

        num_tracks = len(track_ids)
        num_dets = len(detections)

        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            track_box = self.tracks[tid].bbox
            tc = track_centroids[i]
            for j, det in enumerate(detections):
                dist = np.hypot(tc[0] - det_centroids[j][0], tc[1] - det_centroids[j][1])
                iou = bbox_iou(track_box, det.bbox)
                # Gabungan distance norm + IoU penalty
                cost_matrix[i, j] = dist + (1.0 - iou) * 100.0

        # Greedy Assignment
        assigned_tracks = set()
        assigned_dets = set()

        if num_tracks > 0 and num_dets > 0:
            flat_indices = np.argsort(cost_matrix, axis=None)
            for flat_idx in flat_indices:
                row = flat_idx // num_dets
                col = flat_idx % num_dets
                if row in assigned_tracks or col in assigned_dets:
                    continue

                dist = np.hypot(
                    track_centroids[row][0] - det_centroids[col][0],
                    track_centroids[row][1] - det_centroids[col][1]
                )
                if dist <= self.max_distance_px:
                    assigned_tracks.add(row)
                    assigned_dets.add(col)

                    tid = track_ids[row]
                    det = detections[col]
                    track = self.tracks[tid]

                    # Update track state
                    track.prev_centroid = track.centroid
                    track.bbox = det.bbox
                    track.confidence = det.confidence
                    track.hits += 1
                    track.disappeared = 0
                    track.age += 1
                    track.last_seen_time = now
                    track.accumulated_dwell_sec = now - track.first_seen_time

                    # Transition Transient -> Stationary
                    if track.hits >= self.enter_min_frames:
                        track.status = "stationary"
                        track.spatial_anchor = bbox_centroid(det.bbox)

        # Unassigned Tracks -> naikkan disappeared
        for i, tid in enumerate(track_ids):
            if i not in assigned_tracks:
                track = self.tracks[tid]
                track.disappeared += 1
                track.age += 1
                if track.disappeared >= self.max_disappeared_frames:
                    self._archive_to_spatial_memory(track, now)
                    del self.tracks[tid]

        # Unassigned Detections -> Cek Re-ID dulu, jika gagal buat track baru
        for j in range(num_dets):
            if j not in assigned_dets:
                det = detections[j]
                det_centroid = det_centroids[j]
                reidentified_mem = self._try_reid_spatial_memory(det_centroid, det.class_label, now)

                if reidentified_mem:
                    track = TrackState(
                        track_id=reidentified_mem.track_id,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_label=det.class_label,
                        class_id=det.class_id,
                        age=1,
                        hits=self.enter_min_frames,
                        disappeared=0,
                        status="stationary",
                        spatial_anchor=det_centroid,
                        first_seen_time=now - reidentified_mem.accumulated_dwell_sec,
                        last_seen_time=now,
                        accumulated_dwell_sec=reidentified_mem.accumulated_dwell_sec,
                    )
                    self.tracks[track.track_id] = track
                else:
                    track_id = self._next_track_id
                    self._next_track_id += 1
                    track = TrackState(
                        track_id=track_id,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_label=det.class_label,
                        class_id=det.class_id,
                        age=1,
                        hits=1,
                        disappeared=0,
                        status="transient",
                        spatial_anchor=det_centroid,
                        first_seen_time=now,
                        last_seen_time=now,
                        accumulated_dwell_sec=0.0,
                    )
                    self.tracks[track_id] = track

        return [t.to_track_result(frame_number) for t in self.tracks.values() if t.hits >= self.enter_min_frames]

    def _archive_to_spatial_memory(self, track: TrackState, now: float) -> None:
        """Simpan track stationary yang hilang ke memori spasial untuk Re-ID."""
        if track.status == "stationary":
            self.spatial_memory.append(
                SpatialMemoryItem(
                    track_id=track.track_id,
                    class_label=track.class_label,
                    class_id=track.class_id,
                    spatial_anchor=track.spatial_anchor,
                    accumulated_dwell_sec=track.accumulated_dwell_sec,
                    last_seen_time=now,
                )
            )

    def _try_reid_spatial_memory(
        self,
        centroid: Tuple[float, float],
        class_label: str,
        now: float,
    ) -> Optional[SpatialMemoryItem]:
        """Cari apakah centroid dekat dengan spatial anchor di memori spasial."""
        best_match: Optional[SpatialMemoryItem] = None
        min_dist = self.anchor_radius_px

        for mem in self.spatial_memory:
            if mem.class_label != class_label:
                continue
            if (now - mem.last_seen_time) > self.spatial_memory_ttl_sec:
                continue

            dist = np.hypot(centroid[0] - mem.spatial_anchor[0], centroid[1] - mem.spatial_anchor[1])
            if dist <= min_dist:
                min_dist = dist
                best_match = mem

        if best_match:
            self.spatial_memory.remove(best_match)
            return best_match

        return None
