"""
engine/spatial_rules.py
=======================
Advanced Spatial Analytics Engine untuk Facility Management.
Mendukung:
1. Directional Tripwire Crossing (zero-dwell, signed determinant, vector dot-product, debouncing).
2. Polyline Barrier Crossing (multi-segment crossing).
3. Exclusion Mask Filter (suppression of motion, dwell, or all events).
4. Safe Walkway Compliance (K3 Monitoring, footpoint tracking, vehicle proximity latch).
5. Density / Area Congestion Monitor (quota limit & minimum dwell duration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from engine.config_loader import (
    BarrierRule,
    DensityRule,
    ExclusionMask,
    ROIZone,
    ROIZonesConfig,
    SafeWalkwayRule,
    TripwireRule,
)
from engine.geometry import (
    Point,
    bbox_bottom_center,
    bbox_centroid,
    check_line_crossing,
    check_polyline_crossing,
    euclidean_distance,
    point_in_polygon,
    scale_points,
)
from engine.logger import get_logger
from engine.tracker_interface import TrackResult

logger = get_logger("SpatialRules")

# Kategori kelas kendaraan untuk proximity filter pada walkway K3
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class SpatialEvent:
    """Hasil evaluasi event spasial tingkat lanjut."""

    event_type: str  # "LINE_CROSSING", "BARRIER_BREACH", "WALKWAY_VIOLATION", "CONGESTION_ALERT"
    rule_id: str
    rule_label: str
    camera_id: str
    track_id: Optional[int] = None
    class_label: Optional[str] = None
    confidence: Optional[float] = None
    direction: Optional[str] = None
    message: str = ""
    timestamp: float = 0.0
    rule: Optional[Any] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def notes(self) -> str:
        """Alias catatan event untuk database."""
        return self.message


class SpatialRuleEngine:
    """
    Mesin evaluasi analitik spasial terpadu berbasis aljabar vektor skalar murni.
    """

    def __init__(self) -> None:
        # Cache debouncing: (camera_id, rule_id, track_id) -> last_trigger_timestamp
        self._tripwire_debounce: Dict[Tuple[str, str, int], float] = {}
        self._barrier_debounce: Dict[Tuple[str, str, int], float] = {}

        # State Walkway: (camera_id, walkway_id, track_id) -> violation_start_timestamp
        self._walkway_violations: Dict[Tuple[str, str, int], float] = {}
        self._walkway_reported: Set[Tuple[str, str, int]] = set()

        # State Density: (camera_id, rule_id) -> congestion_start_timestamp
        self._density_dwell: Dict[Tuple[str, str], float] = {}
        self._density_reported: Set[Tuple[str, str]] = set()

    def evaluate_tripwires(
        self,
        camera_id: str,
        tracks: List[TrackResult],
        tripwires: List[TripwireRule],
        scale_x: float,
        scale_y: float,
        timestamp: float,
    ) -> List[SpatialEvent]:
        """Evaluasi crossing pada seluruh directional tripwire 2-titik aktif."""
        events: List[SpatialEvent] = []

        for tw in tripwires:
            if not tw.active:
                continue

            # Konversi titik tripwire dari capture resolution (1080p) ke AI canvas scale
            p1_ai = (tw.p1.x / scale_x, tw.p1.y / scale_y)
            p2_ai = (tw.p2.x / scale_x, tw.p2.y / scale_y)

            for trk in tracks:
                if not trk.is_confirmed:
                    continue

                # Filter target classes jika didefinisikan
                if tw.target_classes and trk.class_label not in tw.target_classes:
                    continue

                p_curr = trk.current_centroid
                p_prev = trk.prev_centroid or p_curr

                # Jika objek belum berpindah posisi, lewati
                if p_curr == p_prev:
                    continue

                crossed, actual_dir = check_line_crossing(
                    p_prev=p_prev,
                    p_curr=p_curr,
                    line_start=p1_ai,
                    line_end=p2_ai,
                    allowed_direction=tw.direction,
                )

                if crossed:
                    # Debouncing check per track & tripwire
                    debounce_key = (camera_id, tw.tripwire_id, trk.track_id)
                    last_time = self._tripwire_debounce.get(debounce_key, 0.0)

                    if (timestamp - last_time) >= tw.debounce_sec:
                        self._tripwire_debounce[debounce_key] = timestamp
                        msg = (
                            f"Track #{trk.track_id} ({trk.class_label}) melintasi "
                            f"Tripwire '{tw.label}' (Arah: {actual_dir})"
                        )
                        events.append(
                            SpatialEvent(
                                event_type="LINE_CROSSING",
                                rule_id=tw.tripwire_id,
                                rule_label=tw.label,
                                camera_id=camera_id,
                                track_id=trk.track_id,
                                class_label=trk.class_label,
                                confidence=trk.confidence,
                                direction=actual_dir,
                                message=msg,
                                timestamp=timestamp,
                                rule=tw,
                                details={
                                    "p_prev": p_prev,
                                    "p_curr": p_curr,
                                    "allowed_direction": tw.direction,
                                },
                            )
                        )

        return events

    def evaluate_barriers(
        self,
        camera_id: str,
        tracks: List[TrackResult],
        barriers: List[BarrierRule],
        scale_x: float,
        scale_y: float,
        timestamp: float,
    ) -> List[SpatialEvent]:
        """Evaluasi crossing pada seluruh polyline barrier multi-segmen aktif."""
        events: List[SpatialEvent] = []

        for bar in barriers:
            if not bar.active or len(bar.points) < 2:
                continue

            raw_pts = [(p.x, p.y) for p in bar.points]
            ai_pts = scale_points(raw_pts, scale_x, scale_y)

            for trk in tracks:
                if not trk.is_confirmed:
                    continue

                if bar.target_classes and trk.class_label not in bar.target_classes:
                    continue

                p_curr = trk.current_centroid
                p_prev = trk.prev_centroid or p_curr

                if p_curr == p_prev:
                    continue

                crossed, seg_idx, actual_dir = check_polyline_crossing(
                    p_prev=p_prev,
                    p_curr=p_curr,
                    polyline=ai_pts,
                    allowed_direction=bar.direction,
                )

                if crossed:
                    debounce_key = (camera_id, bar.barrier_id, trk.track_id)
                    last_time = self._barrier_debounce.get(debounce_key, 0.0)

                    if (timestamp - last_time) >= bar.debounce_sec:
                        self._barrier_debounce[debounce_key] = timestamp
                        msg = (
                            f"Track #{trk.track_id} ({trk.class_label}) menerobos "
                            f"Barrier '{bar.label}' segmen #{seg_idx} (Arah: {actual_dir})"
                        )
                        events.append(
                            SpatialEvent(
                                event_type="LINE_CROSSING",
                                rule_id=bar.barrier_id,
                                rule_label=bar.label,
                                camera_id=camera_id,
                                track_id=trk.track_id,
                                class_label=trk.class_label,
                                confidence=trk.confidence,
                                direction=actual_dir,
                                message=msg,
                                timestamp=timestamp,
                                rule=bar,
                                details={
                                    "segment_index": seg_idx,
                                    "allowed_direction": bar.direction,
                                },
                            )
                        )

        return events

    def filter_exclusion(
        self,
        track: TrackResult,
        exclusion_masks: List[ExclusionMask],
        scale_x: float,
        scale_y: float,
        filter_type: str = "all",
    ) -> bool:
        """
        Cek apakah objek berada di dalam exclusion mask.
        Returns:
            True jika objek berada di dalam mask yang mengabaikan `filter_type` (diabaikan),
            False jika objek valid untuk diproses.
        """
        if not exclusion_masks:
            return False

        footpoint = bbox_bottom_center(track.bbox)

        for mask in exclusion_masks:
            if not mask.active or len(mask.points) < 3:
                continue

            # Periksa kecocokan jenis pengabaian
            if filter_type not in mask.ignore_types and "all" not in mask.ignore_types:
                continue

            raw_pts = [(p.x, p.y) for p in mask.points]
            ai_pts = scale_points(raw_pts, scale_x, scale_y)

            if point_in_polygon(footpoint, ai_pts):
                return True

        return False

    def evaluate_walkways(
        self,
        camera_id: str,
        tracks: List[TrackResult],
        walkways: List[SafeWalkwayRule],
        scale_x: float,
        scale_y: float,
        timestamp: float,
    ) -> List[SpatialEvent]:
        """
        Evaluasi kepatuhan jalur pejalan kaki K3 (Safe Walkway Compliance).
        Memicu event jika 'person' berada di luar jalur aman melebihi batas waktu toleransi
        atau jika berada dalam radius berbahaya di dekat kendaraan yang aktif.
        """
        events: List[SpatialEvent] = []
        if not walkways:
            return events

        # Kelompokkan orang vs kendaraan
        person_tracks = [t for t in tracks if t.is_confirmed and t.class_label == "person"]
        vehicle_tracks = [t for t in tracks if t.is_confirmed and t.class_label in VEHICLE_CLASSES]

        for ww in walkways:
            if not ww.active or len(ww.points) < 3:
                continue

            raw_pts = [(p.x, p.y) for p in ww.points]
            ai_pts = scale_points(raw_pts, scale_x, scale_y)

            for person in person_tracks:
                footpoint = bbox_bottom_center(person.bbox)
                is_inside = point_in_polygon(footpoint, ai_pts)
                key = (camera_id, ww.walkway_id, person.track_id)

                if is_inside:
                    # Berada di dalam jalur aman -> reset status pelanggaran
                    self._walkway_violations.pop(key, None)
                    self._walkway_reported.discard(key)
                else:
                    # Berada di luar jalur aman!
                    if key not in self._walkway_violations:
                        self._walkway_violations[key] = timestamp

                    start_time = self._walkway_violations[key]
                    violation_duration = timestamp - start_time

                    # Cek Proximity Latch terhadap kendaraan
                    has_vehicle_hazard = False
                    hazard_dist = 9999.0
                    hazard_veh = None

                    if ww.vehicle_proximity_filter and vehicle_tracks:
                        p_center = bbox_centroid(person.bbox)
                        for veh in vehicle_tracks:
                            v_center = bbox_centroid(veh.bbox)
                            dist = euclidean_distance(p_center, v_center)
                            if dist <= ww.proximity_radius_px:
                                has_vehicle_hazard = True
                                if dist < hazard_dist:
                                    hazard_dist = dist
                                    hazard_veh = veh

                    # Pemicu event: Waktu toleransi terlewati ATAU bahaya tabrakan kendaraan
                    should_trigger = (
                        (violation_duration >= ww.violation_timeout_sec) or has_vehicle_hazard
                    )

                    if should_trigger and key not in self._walkway_reported:
                        self._walkway_reported.add(key)
                        if has_vehicle_hazard:
                            veh_name = hazard_veh.class_label if hazard_veh else "kendaraan"
                            msg = (
                                f"BAHAYA K3: Track #{person.track_id} (person) di luar jalur '{ww.label}' "
                                f"dalam jarak dekat dengan {veh_name} ({int(hazard_dist)}px)!"
                            )
                        else:
                            msg = (
                                f"PELANGGARAN K3: Track #{person.track_id} (person) di luar jalur '{ww.label}' "
                                f"selama {int(violation_duration)}s (Maks: {int(ww.violation_timeout_sec)}s)"
                            )

                        events.append(
                            SpatialEvent(
                                event_type="WALKWAY_VIOLATION",
                                rule_id=ww.walkway_id,
                                rule_label=ww.label,
                                camera_id=camera_id,
                                track_id=person.track_id,
                                class_label="person",
                                confidence=person.confidence,
                                message=msg,
                                timestamp=timestamp,
                                rule=ww,
                                details={
                                    "violation_duration_sec": violation_duration,
                                    "has_vehicle_hazard": has_vehicle_hazard,
                                    "hazard_distance_px": hazard_dist if has_vehicle_hazard else None,
                                },
                            )
                        )

        return events

    def evaluate_density(
        self,
        camera_id: str,
        tracks: List[TrackResult],
        density_rules: List[DensityRule],
        polygons: List[ROIZone],
        scale_x: float,
        scale_y: float,
        timestamp: float,
    ) -> List[SpatialEvent]:
        """
        Evaluasi kuota kepadatan / penumpukan objek (Density / Congestion Monitor).
        Memicu event jika jumlah objek di dalam zona poligon melebihi batas kuota
        selama minimal durasi min_dwell_sec.
        """
        events: List[SpatialEvent] = []
        if not density_rules or not polygons:
            return events

        # Map zone_id -> scaled polygon
        poly_map: Dict[str, Tuple[ROIZone, List[Point]]] = {}
        for poly in polygons:
            if poly.active and len(poly.points) >= 3:
                raw_pts = [(p.x, p.y) for p in poly.points]
                ai_pts = scale_points(raw_pts, scale_x, scale_y)
                poly_map[poly.zone_id] = (poly, ai_pts)

        for rule in density_rules:
            if not rule.active or rule.zone_ref not in poly_map:
                continue

            poly_zone, ai_pts = poly_map[rule.zone_ref]
            key = (camera_id, rule.rule_id)

            # Hitung jumlah objek confirmed di dalam poligon
            inside_tracks = [
                t for t in tracks
                if t.is_confirmed and point_in_polygon(bbox_bottom_center(t.bbox), ai_pts)
            ]
            current_count = len(inside_tracks)

            if current_count >= rule.max_allowed_objects:
                if key not in self._density_dwell:
                    self._density_dwell[key] = timestamp

                dwell_duration = timestamp - self._density_dwell[key]

                if dwell_duration >= rule.min_dwell_sec and key not in self._density_reported:
                    self._density_reported.add(key)
                    msg = (
                        f"KEMACETAN / KEPADATAN TINGGI: Area '{poly_zone.label}' terisi "
                        f"{current_count} objek (Batas Maksimal: {rule.max_allowed_objects}) "
                        f"selama {int(dwell_duration)}s"
                    )
                    events.append(
                        SpatialEvent(
                            event_type="CONGESTION_ALERT",
                            rule_id=rule.rule_id,
                            rule_label=rule.label,
                            camera_id=camera_id,
                            message=msg,
                            timestamp=timestamp,
                            rule=poly_zone,
                            details={
                                "zone_ref": rule.zone_ref,
                                "current_count": current_count,
                                "max_allowed": rule.max_allowed_objects,
                                "dwell_sec": dwell_duration,
                            },
                        )
                    )
            else:
                # Kepadatan normal kembali -> reset timer dan reported flag
                self._density_dwell.pop(key, None)
                self._density_reported.discard(key)

        return events

    def cleanup_tracks(self, active_track_ids: Set[int]) -> None:
        """Bersihkan cache pelacakan untuk objek yang sudah hilang dari stream."""
        to_del_walkway = [
            k for k in self._walkway_violations.keys()
            if k[2] not in active_track_ids
        ]
        for k in to_del_walkway:
            self._walkway_violations.pop(k, None)
            self._walkway_reported.discard(k)
