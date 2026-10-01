"""
engine.parking.motorcycle_block
===============================
Dedicated motorcycle block tracker for cam_03 (dense clustered parking).
Handles:
- Smart deduplication with child fragmentation filtering (IoS >= 0.85).
- Side-by-side relaxation (min_dx separation & wheel contact Y2 separation).
- Spatial Anchor Memory (multi-tier confidence: 0.22 retention, 0.30 acquisition).
- Moving Median Window (anti-flapping stability).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from engine.config_loader import ROIZone
from engine.geometry import bbox_iou
from engine.parking.base import StationaryMotorUnit
from engine.parking.spatial import bbox_ios, bbox_polygon_overlap_ratio
from engine.tracker_interface import TrackResult


def deduplicate_motorcycle_tracks(
    tracks: List[TrackResult],
    iou_thresh: float = 0.40,
    ios_thresh: float = 0.60,
    min_dx_px: float = 12.0,
    cumulative_overlap_thresh: float = 0.85,
    max_centroid_dist_px: Optional[float] = 28.0,
) -> List[TrackResult]:
    """
    Deduplikasi spasial cerdas untuk kendaraan roda dua (motorcycle / bicycle) padat/berhimpitan:
    1. Sort kandidat berdasarkan confidence tertinggi.
    2. Filter Kotak Anak: tolak jika IoS >= ios_thresh (0.60) -> pecahan anak (jok/stang/roda) dari motor yang sama.
    3. Relaksasi Berdampingan:
       - Jika IoU >= iou_thresh (0.40): tolak sebagai duplikat.
       - Proximity Guard: jika jarak Euclidean titik tengah dist(c1, c2) <= max_centroid_dist_px (28.0 px)
         tanpa separasi lateral fisik antar motor (dx < min_dx_px and dy2 < 8.0).
       - Jika ada overlap signifikan (IoU >= 0.20 atau IoS >= 0.30):
         Pertahankan sebagai 2 unit terpisah jika ada separasi sumbu-X (|cx_A - cx_B| >= min_dx_px)
         ATAU separasi titik tumpu roda (|y2_A - y2_B| >= 8.0).
         Tolak sebagai duplikat jika tidak ada separasi fisik (|cx_A - cx_B| < min_dx_px and |y2_A - y2_B| < 8.0).
    4. Multi-Box Cumulative Overlap Rejection: tolak jika >= cumulative_overlap_thresh (0.85) luas kotak tertutup
       oleh gabungan kotak yang sudah diterima, dengan Physical Contact Anchor Exception untuk rentang 0.70 s.d. 0.85.
    """
    sorted_tracks = sorted(tracks, key=lambda t: t.confidence, reverse=True)
    deduped: List[TrackResult] = []

    for cand in sorted_tracks:
        c_cx = float((cand.bbox[0] + cand.bbox[2]) / 2.0)
        c_cy = float((cand.bbox[1] + cand.bbox[3]) / 2.0)
        c_y2 = float(cand.bbox[3])
        is_dup = False

        # 1. Pairwise checks
        for acc in deduped:
            iou = bbox_iou(cand.bbox, acc.bbox)
            ios = bbox_ios(cand.bbox, acc.bbox)
            a_cx = float((acc.bbox[0] + acc.bbox[2]) / 2.0)
            a_cy = float((acc.bbox[1] + acc.bbox[3]) / 2.0)
            a_y2 = float(acc.bbox[3])

            dx = abs(c_cx - a_cx)
            dy2 = abs(c_y2 - a_y2)
            dist = math.hypot(c_cx - a_cx, c_cy - a_cy)

            # Filter Kotak Anak / Bersarang: jika IoS >= ios_thresh (0.60), kotak kecil adalah pecahan
            if ios >= ios_thresh:
                is_dup = True
                break

            # Jika IoU tinggi (>= iou_thresh 0.40), tolak sebagai duplikat
            if iou >= iou_thresh:
                is_dup = True
                break

            # Proximity Guard: jika jarak Euclidean titik tengah dist(c1, c2) <= max_centroid_dist_px (28.0 px)
            # tanpa separasi lateral fisik antar motor (dx < min_dx_px)
            if (
                max_centroid_dist_px is not None
                and dist <= max_centroid_dist_px
                and dx < min_dx_px
                and dy2 < 8.0
            ):
                is_dup = True
                break

            # Relaksasi Berdampingan:
            # Jika ada overlap signifikan dan tidak ada separasi fisik sumbu-X maupun tapak roda
            if (iou >= 0.20 or ios >= 0.30) and (dx < min_dx_px and dy2 < 8.0):
                is_dup = True
                break

        if is_dup:
            continue

        # 2. Multi-Box Cumulative Overlap Rejection
        cand_w = int(math.ceil(cand.bbox[2] - cand.bbox[0]))
        cand_h = int(math.ceil(cand.bbox[3] - cand.bbox[1]))
        if cand_w > 0 and cand_h > 0 and len(deduped) > 0:
            mask = np.zeros((cand_h, cand_w), dtype=np.uint8)
            for acc in deduped:
                ix1 = max(0, int(math.floor(acc.bbox[0] - cand.bbox[0])))
                iy1 = max(0, int(math.floor(acc.bbox[1] - cand.bbox[1])))
                ix2 = min(cand_w, int(math.ceil(acc.bbox[2] - cand.bbox[0])))
                iy2 = min(cand_h, int(math.ceil(acc.bbox[3] - cand.bbox[1])))
                if ix2 > ix1 and iy2 > iy1:
                    mask[iy1:iy2, ix1:ix2] = 1
            covered_ratio = float(np.sum(mask)) / float(cand_w * cand_h)
            if covered_ratio >= cumulative_overlap_thresh:
                is_dup = True
            elif covered_ratio >= 0.70:
                # Physical Contact Anchor Exception:
                # Jika overlap kumulatif berada pada 0.70 <= covered < cumulative_overlap_thresh (0.85),
                # pertahankan kandidat jika terdapat separasi fisik tapak ban
                # (|cx_cand - cx_acc| >= 16.0 px ATAU |y2_cand - y2_acc| >= 6.0 px)
                # terhadap seluruh kotak yang berkontribusi menutupi area kandidat.
                has_physical_separation = True
                for acc in deduped:
                    if bbox_ios(cand.bbox, acc.bbox) > 0.05 or bbox_iou(cand.bbox, acc.bbox) > 0.05:
                        a_cx = float((acc.bbox[0] + acc.bbox[2]) / 2.0)
                        a_y2 = float(acc.bbox[3])
                        if abs(c_cx - a_cx) < 16.0 and abs(c_y2 - a_y2) < 6.0:
                            has_physical_separation = False
                            break
                if not has_physical_separation:
                    is_dup = True

        if not is_dup:
            deduped.append(cand)

    return deduped


def is_valid_motorcycle_anatomy(
    bbox: Tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    max_w_1080p: float = 580.0,
    max_h_1080p: float = 380.0,
    min_w_1080p: float = 20.0,
    min_h_1080p: float = 25.0,
    max_area_1080p: float = 140000.0,
    min_area_360p: float = 800.0,
    min_hw_ratio: float = 0.35,
    max_hw_ratio: float = 2.25,
) -> bool:
    sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
    sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

    x1, y1, x2, y2 = bbox
    bw_curr = max(1.0, float(x2 - x1))
    bh_curr = max(1.0, float(y2 - y1))

    bw_1080p = bw_curr / sx
    bh_1080p = bh_curr / sy
    area_1080p = bw_1080p * bh_1080p
    area_curr = bw_curr * bh_curr
    hw_ratio = bh_1080p / bw_1080p

    # Filter luas minimum (800 px² pada 360p atau 7200 px² pada 1080p)
    if area_curr < min_area_360p and area_1080p < (min_area_360p * 9.0):
        return False

    if (
        bw_1080p > max_w_1080p
        or bh_1080p > max_h_1080p
        or bw_1080p < min_w_1080p
        or bh_1080p < min_h_1080p
        or area_1080p > max_area_1080p
        or hw_ratio < min_hw_ratio
        or hw_ratio > max_hw_ratio
    ):
        return False
    return True


class MotorcycleBlockTracker:
    def __init__(
        self,
        block_capacity: int = 30,
        stationary_dwell_sec: float = 0.0,
        block_exclusion_x_1080p: Optional[int] = None,
        vehicle_classes: Optional[Set[str]] = None,
    ) -> None:
        self.block_capacity = block_capacity
        self.stationary_dwell_sec = stationary_dwell_sec
        self._block_exclusion_x_1080p: Optional[int] = block_exclusion_x_1080p
        self.vehicle_classes = set(vehicle_classes) if vehicle_classes is not None else {"motorcycle", "bicycle"}

        self._stationary_motor_units: Dict[int, StationaryMotorUnit] = {}
        self._next_motor_unit_id: int = 1
        self._density_history: deque = deque(maxlen=25)
        self._last_density_time: Optional[float] = None
        self._last_stable_occupied: Optional[int] = None

    @property
    def total_slots(self) -> int:
        return self.block_capacity

    @property
    def occupied_slots(self) -> int:
        return self._last_stable_occupied if self._last_stable_occupied is not None else 0

    @property
    def available_slots(self) -> int:
        return max(0, self.total_slots - self.occupied_slots)

    def _block_stats(self, occupied: int) -> Dict[str, Any]:
        self._last_stable_occupied = occupied
        available = max(0, self.block_capacity - occupied)
        return {
            "total_slots": self.block_capacity,
            "occupied_slots": occupied,
            "available_slots": available,
            "slot_states": {},
            "parking_mode": "motorcycle_block",
        }

    def update(
        self,
        tracks: List[TrackResult],
        polygons: List[ROIZone],
        scale_x: float,
        scale_y: float,
        current_time: float,
        is_warmup: bool = False,
    ) -> Dict[str, Any]:
        active_zone = next((p for p in polygons if p.active), None)
        if active_zone is None:
            return self._block_stats(0)

        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        pts_scaled = np.array(
            [[int(pt.x * sx), int(pt.y * sy)] for pt in active_zone.points],
            dtype=np.int32,
        )

        vehicle_tracks = [
            t
            for t in tracks
            if (t.is_confirmed or is_warmup or getattr(t, "age", 0) >= 1) and (t.class_label in self.vehicle_classes)
        ]

        drum_excl_x_scaled: Optional[int] = None
        if self._block_exclusion_x_1080p is not None:
            drum_excl_x_scaled = int(self._block_exclusion_x_1080p * sx)

        scale_factor = max(1.0, sx / (1.0 / 3.0))
        anchor_match_dist = 10.0 * scale_factor

        candidate_tracks: List[TrackResult] = []
        for track in vehicle_tracks:
            rx1, ry1, rx2, ry2 = track.bbox
            cx = float((rx1 + rx2) / 2.0)
            cy = float((ry1 + ry2) / 2.0)
            wheel_y = float(ry2)
            wheel_pt = (int(cx), int(wheel_y))
            center_pt = (int(cx), int(cy))

            if drum_excl_x_scaled is not None and int(cx) < drum_excl_x_scaled:
                continue

            if not is_valid_motorcycle_anatomy(track.bbox, sx, sy):
                continue

            # Multi-Tier Thresholding:
            # - Ambang retensi motor terkunci (is_latched): conf >= 0.08
            # - Ambang akuisisi motor baru: conf >= 0.10
            is_near_latched = False
            for unit in self._stationary_motor_units.values():
                if unit.is_latched and math.hypot(cx - unit.centroid[0], cy - unit.centroid[1]) < anchor_match_dist:
                    is_near_latched = True
                    break

            min_conf = 0.08 if is_near_latched else 0.10
            if track.confidence < min_conf:
                continue

            # Kontak roda pada batas poligon & Penolakan Motor di Luar Garis
            d_wheel = float(cv2.pointPolygonTest(pts_scaled, wheel_pt, True))
            d_center = float(cv2.pointPolygonTest(pts_scaled, center_pt, True))
            overlap_ratio = bbox_polygon_overlap_ratio(track.bbox, pts_scaled)

            # Syarat mutlak: Titik tengah ATAU titik tapak ban bawah WAJIB di dalam poligon (>= 0.0 px)
            if d_wheel < 0.0 and d_center < 0.0:
                continue

            # Jika titik tumpu berada di luar poligon (d_wheel < 0.0 px),
            # proposal HANYA boleh dipertimbangkan jika memiliki persentase irisan luas overlap_ratio >= 0.25 (25%)
            if d_wheel < 0.0:
                if overlap_ratio < 0.25:
                    continue
            else:
                # Titik tumpu di dalam (d_wheel >= 0.0 px): wajib overlap wajar >= 0.10
                if overlap_ratio < 0.10:
                    continue

            candidate_tracks.append(track)

        valid_tracks = deduplicate_motorcycle_tracks(
            candidate_tracks,
            iou_thresh=0.40,
            ios_thresh=0.60,
            min_dx_px=6.0 * scale_factor,
            cumulative_overlap_thresh=0.85,
            max_centroid_dist_px=28.0,
        )

        UNIT_LATCH_SEC = 1.2
        UNIT_TTL_SEC = 12.0

        matched_unit_ids: Set[int] = set()

        for cand in sorted(valid_tracks, key=lambda t: t.confidence, reverse=True):
            cand_cx = float((cand.bbox[0] + cand.bbox[2]) / 2.0)
            cand_cy = float((cand.bbox[1] + cand.bbox[3]) / 2.0)

            best_unit_id = None
            best_dist = anchor_match_dist

            for uid, unit in self._stationary_motor_units.items():
                if uid in matched_unit_ids:
                    continue
                dist = math.hypot(cand_cx - unit.centroid[0], cand_cy - unit.centroid[1])
                if dist < best_dist:
                    best_dist = dist
                    best_unit_id = uid

            if best_unit_id is not None:
                unit = self._stationary_motor_units[best_unit_id]
                unit.bbox = cand.bbox
                unit.confidence = cand.confidence
                unit.centroid = (cand_cx, cand_cy)
                unit.last_seen_time = current_time
                unit.consecutive_hits += 1
                unit.missed_frames = 0
                if not unit.is_latched:
                    elapsed_visible = current_time - unit.first_seen_time
                    if elapsed_visible >= UNIT_LATCH_SEC or unit.consecutive_hits >= 2 or is_warmup:
                        unit.is_latched = True
                matched_unit_ids.add(best_unit_id)
            else:
                uid = self._next_motor_unit_id
                self._next_motor_unit_id += 1
                is_latched_init = is_warmup
                hits_init = 1
                self._stationary_motor_units[uid] = StationaryMotorUnit(
                    unit_id=uid,
                    bbox=cand.bbox,
                    confidence=cand.confidence,
                    centroid=(cand_cx, cand_cy),
                    first_seen_time=current_time,
                    last_seen_time=current_time,
                    consecutive_hits=hits_init,
                    missed_frames=0,
                    is_latched=is_latched_init,
                )
                matched_unit_ids.add(uid)

        unmatched_unit_ids = set(self._stationary_motor_units.keys()) - matched_unit_ids
        to_delete = []
        for uid in unmatched_unit_ids:
            unit = self._stationary_motor_units[uid]
            unit.missed_frames += 1
            unit.consecutive_hits = 0

            time_since_last_seen = current_time - unit.last_seen_time

            if unit.is_latched:
                if time_since_last_seen > UNIT_TTL_SEC:
                    to_delete.append(uid)
            else:
                if time_since_last_seen > 3.0:
                    to_delete.append(uid)

        for uid in to_delete:
            del self._stationary_motor_units[uid]

        # Prune redundant/overlapping stationary units
        prune_uids = set()
        active_uids = list(self._stationary_motor_units.keys())
        for i in range(len(active_uids)):
            u1_id = active_uids[i]
            if u1_id in prune_uids or u1_id not in self._stationary_motor_units:
                continue
            u1 = self._stationary_motor_units[u1_id]
            for j in range(i + 1, len(active_uids)):
                u2_id = active_uids[j]
                if u2_id in prune_uids or u2_id not in self._stationary_motor_units:
                    continue
                u2 = self._stationary_motor_units[u2_id]
                u_dist = math.hypot(u1.centroid[0] - u2.centroid[0], u1.centroid[1] - u2.centroid[1])
                u_iou = bbox_iou(u1.bbox, u2.bbox)
                u_ios = bbox_ios(u1.bbox, u2.bbox)
                u_dx = abs(u1.centroid[0] - u2.centroid[0])
                u_dy2 = abs(u1.bbox[3] - u2.bbox[3])
                if u_iou >= 0.40 or u_ios >= 0.60 or (u_dist <= 28.0 and u_dx < (6.0 * scale_factor) and u_dy2 < 8.0):
                    if u1.consecutive_hits >= u2.consecutive_hits:
                        prune_uids.add(u2_id)
                    else:
                        prune_uids.add(u1_id)
                        break
        for uid in prune_uids:
            if uid in self._stationary_motor_units:
                del self._stationary_motor_units[uid]

        if self.stationary_dwell_sec > 0.0:
            density_count = sum(
                1
                for u in self._stationary_motor_units.values()
                if (u.is_latched or (current_time - u.first_seen_time) >= self.stationary_dwell_sec)
                and is_valid_motorcycle_anatomy(u.bbox, sx, sy)
                and (drum_excl_x_scaled is None or u.centroid[0] >= drum_excl_x_scaled)
            )
        else:
            density_count = sum(
                1
                for u in self._stationary_motor_units.values()
                if is_valid_motorcycle_anatomy(u.bbox, sx, sy)
                and (drum_excl_x_scaled is None or u.centroid[0] >= drum_excl_x_scaled)
            )

        self._last_density_time = current_time
        self._density_history.append(density_count)
        # Moving Median Window 5 detik (25 frame @ ~5fps)
        consolidated_density = int(round(float(np.median(self._density_history))))

        occupied = int(min(self.block_capacity, consolidated_density))
        self._last_stable_occupied = occupied

        return self._block_stats(occupied)
