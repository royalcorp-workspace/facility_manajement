"""
engine/smart_parking.py
=======================
Smart Parking Occupancy & Gate Traffic Tracker untuk Facility Management.
Fitur:
1. Dynamic Slot Capacity: Total Kapasitas = len(polygons) otomatis tanpa hardcode.
2. Pairing 1-ke-1 Berbasis Suffix Indeks: slot_idx = zone_id.split("_")[-1] <-> tw_{slot_idx}.
3. Triple-Check Verification State Machine:
   - State per Slot: VACANT -> ENTERING -> OCCUPIED -> LEAVING -> VACANT
   - Pilar 1: YOLO Detection & Tracking (vehicle classes: car, truck, bus).
   - Pilar 2: Wheel Contact in Polygon & Dwell Threshold (10 detik).
   - Pilar 3: Directional Tripwire Signals (A_TO_B = masuk, B_TO_A = keluar).
   - Debounce Anti-Jitter: Re-park window 5s, grace period 2s, timeout false-entry 30s.
4. Total UI/UX Decluttering (Sleek Modern IVA):
   - Outline slot VACANT 1px cyan transparan halus tanpa teks label.
   - Outline slot ENTERING 1px amber tanpa teks label.
   - Outline slot LEAVING 1px kuning tanpa teks label.
   - Outline slot OCCUPIED 2px merah + fill transparan 0.12 alpha + mini-badge S{num} #{track_id}.
   - Garis tripwire 1px amber tipis + chevron mikro (tanpa teks TW-XX).
   - Titik tumpu roda solid cyan r=2px.
   - Status bar atas ringkas: PARKING: {available}/{total} SLOTS AVAILABLE | OCCUPIED: [{list}]
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import cv2
import numpy as np

from engine.config_loader import ROIZone, TripwireRule
from engine.geometry import bbox_bottom_center, bbox_iou, point_in_polygon, scale_points
from engine.tracker_interface import TrackResult


@dataclass
class SlotState:
    """Status representasi satu slot parkir berbasis State Machine."""

    slot_id: str
    slot_num: int
    slot_idx_str: str
    label: str
    # --- State Machine Phase ---
    phase: Literal["VACANT", "ENTERING", "OCCUPIED", "LEAVING"] = "VACANT"
    track_id: Optional[int] = None
    vehicle_class: Optional[str] = None
    first_seen_time: Optional[float] = None
    dwell_duration: float = 0.0
    last_seen_time: float = 0.0
    # --- Triple-Check Fields ---
    tripwire_crossed_in_time: Optional[float] = None
    entering_timeout_sec: float = 30.0
    leaving_since: Optional[float] = None
    repark_window_sec: float = 5.0
    # --- Anti-ID Churn / Spatial Stabilization Fields ---
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    # Timestamp pertama kali poligon benar-benar terbukti kosong secara fisik.
    # Slot hanya boleh transisi OCCUPIED->LEAVING setelah area kosong >= vacant_confirm_sec (anti-flicker 5 detik).
    polygon_clear_since: Optional[float] = None
    vacant_confirm_sec: float = 5.0
    exit_grace_sec: float = 2.0
    # --- Sticky Occupied Latch Fields ---
    latch_occupied: bool = False
    latch_dwell_threshold_sec: float = 5.0

    @property
    def occupied(self) -> bool:
        """Backward-compatible property: True jika phase == OCCUPIED."""
        return self.phase == "OCCUPIED"


def bbox_polygon_overlap_ratio(
    bbox: Tuple[float, float, float, float],
    pts_scaled: np.ndarray,
) -> float:
    """
    Menghitung rasio tumpang-tindih luasan (Intersection over BBox Area):
    Intersection(BBox, Polygon) / Area(BBox).
    Menggunakan rasterisasi lokal OpenCV O(1) cepat.
    """
    bx1, by1, bx2, by2 = bbox
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    bbox_area = bw * bh

    px, py, pw, ph = cv2.boundingRect(pts_scaled)
    ix1 = max(bx1, float(px))
    iy1 = max(by1, float(py))
    ix2 = min(bx2, float(px + pw))
    iy2 = min(by2, float(py + ph))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    rx1, ry1 = int(math.floor(ix1)), int(math.floor(iy1))
    rx2, ry2 = int(math.ceil(ix2)), int(math.ceil(iy2))
    mw = rx2 - rx1
    mh = ry2 - ry1
    if mw <= 0 or mh <= 0:
        return 0.0

    mask = np.zeros((mh, mw), dtype=np.uint8)
    poly_shifted = pts_scaled - np.array([rx1, ry1], dtype=np.int32)
    cv2.fillPoly(mask, [poly_shifted], 255)

    bx_s1 = max(0, int(round(bx1 - rx1)))
    by_s1 = max(0, int(round(by1 - ry1)))
    bx_s2 = min(mw, int(round(bx2 - rx1)))
    by_s2 = min(mh, int(round(by2 - ry1)))

    if bx_s2 <= bx_s1 or by_s2 <= by_s1:
        return 0.0

    box_submask = mask[by_s1:by_s2, bx_s1:bx_s2]
    intersection_area = float(cv2.countNonZero(box_submask))
    return intersection_area / bbox_area


def bbox_ios(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Intersection over Smaller Area (IoS): intersection / min(area_a, area_b)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(intersection / min(area_a, area_b))


def deduplicate_motorcycle_tracks(
    tracks: List[TrackResult],
    iou_thresh: float = 0.30,
    ios_thresh: float = 0.50,
    max_centroid_dist_px: float = 28.0,
) -> List[TrackResult]:
    """
    Deduplikasi spasial proposal motor (Local NMS / IoU + IoS + Centroid Proximity):
    Urutkan kandidat berdasarkan confidence menurun.
    Gugurkan deteksi ber-confidence lebih rendah jika:
      - IoU >= iou_thresh (default 0.30), ATAU
      - IoS >= ios_thresh (default 0.50), ATAU
      - Jarak Euclidean centroid (cx, cy) <= max_centroid_dist_px (default 28.0 px).
    """
    sorted_tracks = sorted(tracks, key=lambda t: t.confidence, reverse=True)
    deduped: List[TrackResult] = []
    for cand in sorted_tracks:
        c_cx = float((cand.bbox[0] + cand.bbox[2]) / 2.0)
        c_cy = float((cand.bbox[1] + cand.bbox[3]) / 2.0)
        is_dup = False
        for acc in deduped:
            iou = bbox_iou(cand.bbox, acc.bbox)
            ios = bbox_ios(cand.bbox, acc.bbox)
            a_cx = float((acc.bbox[0] + acc.bbox[2]) / 2.0)
            a_cy = float((acc.bbox[1] + acc.bbox[3]) / 2.0)
            dist_px = math.hypot(c_cx - a_cx, c_cy - a_cy)
            if iou >= iou_thresh or ios >= ios_thresh or dist_px <= max_centroid_dist_px:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)
    return deduped


@dataclass
class StationaryMotorUnit:
    """Representasi memori spasial unit motor stasioner untuk temporal latching (anti-flapping)."""
    unit_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    centroid: Tuple[float, float]
    first_seen_time: float
    last_seen_time: float
    consecutive_hits: int = 1
    missed_frames: int = 0
    is_latched: bool = False


class SmartParkingTracker:
    """
    Pelacak status okupansi slot parkir dan telemetri arus gerbang tripwire
    menggunakan Triple-Check Verification State Machine.
    """

    DEFAULT_VEHICLE_CLASSES = {"car", "truck", "bus"}

    def __init__(
        self,
        total_slots: Optional[int] = None,
        dwell_threshold_sec: float = 10.0,
        vehicle_classes: Optional[Set[str]] = None,
        parking_mode: str = "slot",
        block_capacity: int = 30,
        stationary_dwell_sec: float = 0.0,
        block_exclusion_x_1080p: Optional[int] = None,
    ) -> None:
        self._total_slots_override = total_slots
        self.dwell_threshold_sec = dwell_threshold_sec
        self.vehicle_classes = vehicle_classes or self.DEFAULT_VEHICLE_CLASSES
        self.parking_mode = parking_mode
        self.block_capacity = block_capacity
        self.stationary_dwell_sec = stationary_dwell_sec
        # Exclusion zone kiri untuk block mode (koordinat 1080p).
        # Jika di-set, motor yang centroid-x-nya < nilai ini akan dieliminasi sebelum
        # masuk ke perhitungan kuota. Poligon ROI JSON pengguna tidak diubah.
        # None = tidak aktif (default, agar unit test tidak terpengaruh).
        self._block_exclusion_x_1080p: Optional[int] = block_exclusion_x_1080p
        # Map: track_id -> first_seen_in_zone_time (for block density tracking)
        self._zone_dwell_map: Dict[int, float] = {}
        # Stationary Unit Latch & Anti-Flapping Moving Window (motorcycle_block mode)
        self._stationary_motor_units: Dict[int, StationaryMotorUnit] = {}
        self._next_motor_unit_id: int = 1
        self._density_history: deque = deque(maxlen=25)
        self._last_density_time: Optional[float] = None
        self._last_stable_occupied: Optional[int] = None
        self._candidate_occupied: Optional[int] = None
        self._candidate_occupied_since: float = 0.0
        # Map: slot_id -> SlotState
        self.slot_states: Dict[str, SlotState] = {}
        # Gate traffic counters
        self.gate_in_count: int = 0
        self.gate_out_count: int = 0
        # Flash trigger tracker: tripwire_id -> flash_until timestamp
        self.tripwire_flash: Dict[str, float] = {}

    @property
    def total_slots(self) -> int:
        """Kapasitas total dinamis berdasarkan slot yang terdaftar atau block capacity."""
        if self.parking_mode == "motorcycle_block":
            return self.block_capacity
        if self._total_slots_override is not None:
            return self._total_slots_override
        return len(self.slot_states)

    @property
    def occupied_slots(self) -> int:
        """Jumlah slot terisi (kompatibel untuk slot mobil maupun motorcycle block)."""
        if self.parking_mode == "motorcycle_block":
            return self._last_stable_occupied if self._last_stable_occupied is not None else 0
        return sum(1 for s in self.slot_states.values() if s.occupied)

    @property
    def available_slots(self) -> int:
        """Jumlah slot tersedia."""
        return max(0, self.total_slots - self.occupied_slots)

    def apply_tripwire_signal(
        self,
        slot_id: str,
        direction: str,
        track_id: int,
        current_time: float,
    ) -> None:
        """
        Terima sinyal crossing tripwire pasangan (A_TO_B = masuk, B_TO_A = keluar).
        Menggerakkan transisi state machine pada slot target.
        """
        if self.parking_mode == "motorcycle_block":
            return

        if slot_id not in self.slot_states:
            slot_idx_str = slot_id.split("_")[-1]
            try:
                slot_num = int(slot_idx_str)
            except ValueError:
                slot_num = len(self.slot_states) + 1
            self.slot_states[slot_id] = SlotState(
                slot_id=slot_id,
                slot_num=slot_num,
                slot_idx_str=slot_idx_str,
                label=f"Slot {slot_num}",
            )

        state = self.slot_states[slot_id]

        if direction == "A_TO_B":
            if state.phase == "VACANT":
                state.phase = "ENTERING"
                state.tripwire_crossed_in_time = current_time
                state.track_id = track_id
        elif direction == "B_TO_A":
            if state.phase == "ENTERING":
                # False entry (kendaraan sempat mau masuk lalu batal mundur keluar)
                state.phase = "VACANT"
                state.track_id = None
                state.tripwire_crossed_in_time = None
                state.first_seen_time = None
                state.dwell_duration = 0.0
            elif state.phase == "OCCUPIED":
                state.phase = "LEAVING"
                state.leaving_since = current_time

    def record_gate_crossing(
        self,
        direction_or_note: str,
        tripwire_id: str = "",
        timestamp: float = 0.0,
    ) -> None:
        """Catat telemetri kendaraan melintasi tripwire gerbang dan trigger flash effect 1.0s."""
        text = str(direction_or_note).upper()
        # Cek arah OUT terlebih dahulu
        if (
            "B_TO_A" in text
            or "B TO A" in text
            or " OUT" in text
            or text.endswith("OUT")
            or text == "OUT"
        ):
            self.gate_out_count += 1
        elif (
            "A_TO_B" in text
            or "A TO B" in text
            or " IN" in text
            or text.endswith("IN")
            or text == "IN"
        ):
            self.gate_in_count += 1
        else:
            self.gate_in_count += 1

        # Flash trigger effect 1.0 detik
        if tripwire_id and timestamp > 0:
            self.tripwire_flash[tripwire_id] = timestamp + 1.0

    def _is_same_vehicle(
        self,
        state: "SlotState",
        new_track: TrackResult,
        pts_scaled: np.ndarray,
        current_time: float,
        grace_period_sec: float = 3.0,
        iou_threshold: float = 0.35,
    ) -> bool:
        """
        Evaluasi apakah track baru dengan ID berbeda adalah kendaraan SAMA
        yang menempati slot ini (ID Churn / Track Switching detection).

        Syarat temporal (wajib): selisih waktu hilang <= grace_period_sec (3.0s).
        Syarat spasial (salah satu):
          [A] bottom_center roda track baru ada di dalam poligon slot.
          [B] IoU antara bbox baru vs bbox lama yang tersimpan >= iou_threshold (0.35).

        Returns True jika kendaraan dianggap SAMA → lakukan ID update senyap.
        """
        # Gating temporal: jika sudah terlalu lama hilang, bukan ID churn
        if state.last_seen_time > 0:
            elapsed = current_time - state.last_seen_time
            if elapsed > grace_period_sec:
                return False

        x1, y1, x2, y2 = new_track.bbox
        cx = float((x1 + x2) / 2.0)
        wheel_pt = (cx, float(y2))
        lower_pt = (cx, float(y1 + (y2 - y1) * 0.75))
        center_pt = (cx, float((y1 + y2) / 2.0))

        # Syarat Spasial A (kuat): titik kontak roda / lower body / center di dalam poligon slot (margin toleransi ketat -3.0px)
        if (
            cv2.pointPolygonTest(pts_scaled, wheel_pt, True) >= -3.0
            or cv2.pointPolygonTest(pts_scaled, lower_pt, True) >= -3.0
            or cv2.pointPolygonTest(pts_scaled, center_pt, True) >= -3.0
        ):
            return True

        # Syarat Spasial B (toleran): IoU bbox baru vs bbox tersimpan >= threshold
        if state.last_bbox is not None:
            iou = bbox_iou(new_track.bbox, state.last_bbox)
            if iou >= iou_threshold:
                return True

        return False

    def _block_stats(self, occupied: int) -> Dict[str, Any]:
        self._last_stable_occupied = occupied
        available = max(0, self.block_capacity - occupied)
        return {
            "total_slots": self.block_capacity,
            "occupied_slots": occupied,
            "available_slots": available,
            "gate_in": self.gate_in_count,
            "gate_out": self.gate_out_count,
            "slot_states": {},
            "parking_mode": "motorcycle_block",
        }

    def _update_block_mode(
        self,
        tracks: List[TrackResult],
        polygons: List[ROIZone],
        scale_x: float,
        scale_y: float,
        current_time: float,
        is_warmup: bool = False,
    ) -> Dict[str, Any]:
        """
        Block Parking Mode untuk cam_03 (Motor 30 unit).
        Double Verification:
          Pilar 1: tw_count  = max(0, gate_in_count - gate_out_count) (net akumulasi tripwire)
          Pilar 2: density   = jumlah motor stasioner di dalam polygon (dwell >= stationary_dwell_sec)
          Consolidated:  occupied = clamp(max(tw_count, density), 0, block_capacity)
        """
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
        # 1. Filter footprint awal: Titik tumpu roda bawah (cx, y2) WAJIB berada di dalam poligon
        # ─────────────────────────────────────────────────────────────────────────────────────────
        # EXCLUSION ZONE DRUM BIRU (kode — BUKAN ubah poligon JSON pengguna):
        #   Motor di sisi kiri area drum dieliminasi sebelum masuk ke perhitungan kuota.
        #   Aktif hanya jika self._block_exclusion_x_1080p di-set (non-None) saat inisialisasi.
        #   Poligon ROI JSON pengguna TIDAK PERNAH diubah. Garis hijau tetap digambar penuh.
        # ─────────────────────────────────────────────────────────────────────────────────────────
        drum_excl_x_scaled: Optional[int] = None
        if self._block_exclusion_x_1080p is not None:
            drum_excl_x_scaled = int(self._block_exclusion_x_1080p * sx)

        candidate_tracks: List[TrackResult] = []
        for track in vehicle_tracks:
            rx1, ry1, rx2, ry2 = track.bbox
            cx = float((rx1 + rx2) / 2.0)
            wheel_y = float(ry2)
            wheel_pt = (int(cx), int(wheel_y))

            # Exclusion Zone Drum (hanya aktif jika block_exclusion_x_1080p di-set):
            # Motor dengan centroid-x di bawah ambang dikritisi sebagai area drum/non-parkir.
            if drum_excl_x_scaled is not None and int(cx) < drum_excl_x_scaled:
                continue

            # Filter footprint standar: roda bawah WAJIB di dalam poligon
            wheel_in = cv2.pointPolygonTest(pts_scaled, wheel_pt, False) >= 0
            if not wheel_in:
                continue

            overlap_ratio = bbox_polygon_overlap_ratio(track.bbox, pts_scaled)
            if overlap_ratio < 0.20:
                continue

            candidate_tracks.append(track)

        # 2. Deduplikasi spasial diperketat (Local NMS / IoU + IoS + Centroid Proximity <= 28px)
        valid_tracks = deduplicate_motorcycle_tracks(
            candidate_tracks,
            iou_thresh=0.30,
            ios_thresh=0.50,
            max_centroid_dist_px=28.0,
        )

        # 3. Time-Based Persistent Latching untuk Motor Stasioner (Anchor Matching <= 35px)
        # ─────────────────────────────────────────────────────────────────────────────────
        # LATCH RULES (berbasis waktu riil, bukan frame count):
        #   - Unit baru dianggap LATCHED setelah TERDETEKSI YOLO selama >= unit_latch_sec (3.0s).
        #   - Selama LATCHED, unit dipertahankan di memori selama unit_ttl_sec (15.0s) sejak
        #     last_seen_time, MESKIPUN tidak muncul di frame (bayangan/oklusi/RTSP packet loss).
        #   - Unit NON-LATCHED yang hilang > 1.0 detik langsung dieliminasi.
        # ─────────────────────────────────────────────────────────────────────────────────
        UNIT_LATCH_SEC = 3.0   # durasi minimal terdeteksi sebelum dianggap stasioner
        UNIT_TTL_SEC = 15.0    # durasi retensi unit latched setelah hilang dari YOLO

        matched_unit_ids: Set[int] = set()

        for cand in sorted(valid_tracks, key=lambda t: t.confidence, reverse=True):
            cand_cx = float((cand.bbox[0] + cand.bbox[2]) / 2.0)
            cand_cy = float((cand.bbox[1] + cand.bbox[3]) / 2.0)

            best_unit_id = None
            best_dist = 35.0  # ambang jarak anchor centroid (px)

            for uid, unit in self._stationary_motor_units.items():
                if uid in matched_unit_ids:
                    continue
                dist = math.hypot(cand_cx - unit.centroid[0], cand_cy - unit.centroid[1])
                if dist < best_dist:
                    best_dist = dist
                    best_unit_id = uid

            if best_unit_id is not None:
                # Unit lama ditemukan: update posisi dan waktu
                unit = self._stationary_motor_units[best_unit_id]
                unit.bbox = cand.bbox
                unit.confidence = cand.confidence
                unit.centroid = (cand_cx, cand_cy)
                unit.last_seen_time = current_time
                unit.consecutive_hits += 1
                unit.missed_frames = 0
                # TIME-BASED LATCH: latched jika durasi keberadaan >= UNIT_LATCH_SEC
                if not unit.is_latched:
                    elapsed_visible = current_time - unit.first_seen_time
                    if elapsed_visible >= UNIT_LATCH_SEC or is_warmup:
                        unit.is_latched = True
                matched_unit_ids.add(best_unit_id)
            else:
                # Unit baru: daftarkan ke memori temporal
                uid = self._next_motor_unit_id
                self._next_motor_unit_id += 1
                # Pada warmup (cold-start), langsung anggap latched
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

        # Update unit tak terdeteksi pada frame ini (toleransi missed detection)
        unmatched_unit_ids = set(self._stationary_motor_units.keys()) - matched_unit_ids
        to_delete = []
        for uid in unmatched_unit_ids:
            unit = self._stationary_motor_units[uid]
            unit.missed_frames += 1
            unit.consecutive_hits = 0

            time_since_last_seen = current_time - unit.last_seen_time

            if unit.is_latched:
                # Time-Based Persistent Latch (TTL = UNIT_TTL_SEC = 15.0 detik).
                # Sekali terkonfirmasi latched, unit tetap hidup selama 15 detik meski
                # hilang dari deteksi YOLO (toleransi miss frame RTSP / bayangan / oklusi).
                if time_since_last_seen > UNIT_TTL_SEC:
                    to_delete.append(uid)
            else:
                # Unit belum latched: eliminasi agresif jika hilang > 1.0 detik
                if time_since_last_seen > 1.0:
                    to_delete.append(uid)

        for uid in to_delete:
            del self._stationary_motor_units[uid]

        # Hitung kepadatan motor stasioner aktif
        if self.stationary_dwell_sec > 0.0:
            density_count = sum(
                1 for u in self._stationary_motor_units.values()
                if u.is_latched or (current_time - u.first_seen_time) >= self.stationary_dwell_sec
            )
        else:
            density_count = len(self._stationary_motor_units)

        # 4. Moving Median Rolling Window pada Kuota (deque maxlen=25)
        # TIDAK mereset buffer berdasarkan jeda waktu (quiescent scan berjalan tiap 3 detik
        # dan akan menyebabkan reset terus-menerus jika logika berbasis waktu dipakai).
        self._last_density_time = current_time

        self._density_history.append(density_count)
        consolidated_density = int(round(float(np.median(self._density_history))))

        # Pilar 1: Tripwire Net Count
        tw_count = max(0, self.gate_in_count - self.gate_out_count)

        # Konsolidasi Double Verification (Kunci kuota anti-flapping)
        occupied = int(min(self.block_capacity, max(tw_count, consolidated_density)))
        return self._block_stats(occupied)

    def update(
        self,
        tracks: List[TrackResult],
        polygons: List[ROIZone],
        scale_x: float,
        scale_y: float,
        current_time: float,
        is_warmup: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluasi keberadaan kendaraan pada masing-masing slot poligon berbasis State Machine:
        VACANT -> ENTERING -> OCCUPIED -> LEAVING -> VACANT.
        """
        if self.parking_mode == "motorcycle_block":
            return self._update_block_mode(
                tracks=tracks,
                polygons=polygons,
                scale_x=scale_x,
                scale_y=scale_y,
                current_time=current_time,
                is_warmup=is_warmup,
            )

        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        active_slots = [p for p in polygons if p.active]
        total_capacity = self._total_slots_override if self._total_slots_override is not None else len(polygons)

        # Filter track kendaraan yang aktif & terkonfirmasi (atau masa warmup cold-start)
        vehicle_tracks = [
            t
            for t in tracks
            if (t.is_confirmed or is_warmup) and (t.class_label in self.vehicle_classes)
        ]

        # ── Pre-inisialisasi Slot States & Pre-komputasi Geometri Poligon ──────
        slot_geometries: Dict[str, Dict[str, Any]] = {}
        for idx, slot in enumerate(active_slots):
            slot_id = slot.zone_id
            # Pairing suffix index: zone_01 -> "01", zone_07 -> "07"
            slot_idx_str = slot_id.split("_")[-1]
            try:
                slot_num = int(slot_idx_str)
            except ValueError:
                slot_num = idx + 1

            if slot_id not in self.slot_states:
                self.slot_states[slot_id] = SlotState(
                    slot_id=slot_id,
                    slot_num=slot_num,
                    slot_idx_str=slot_idx_str,
                    label=slot.label or f"Slot {slot_num}",
                )

            pts_scaled = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )
            M = cv2.moments(pts_scaled)
            area = M["m00"]
            slot_cx = M["m10"] / (area + 1e-5) if area > 0 else float(np.mean(pts_scaled[:, 0]))
            slot_cy = M["m01"] / (area + 1e-5) if area > 0 else float(np.mean(pts_scaled[:, 1]))
            rx, ry, rw, rh = cv2.boundingRect(pts_scaled)
            slot_bbox = (float(rx), float(ry), float(rx + rw), float(ry + rh))

            slot_geometries[slot_id] = {
                "pts_scaled": pts_scaled,
                "cx": slot_cx,
                "cy": slot_cy,
                "diag": max(10.0, math.hypot(rw, rh)),
                "bbox": slot_bbox,
            }

        # ── Exclusive Slot Assignment (Bipartite Matching: 1 Mobil = Max 1 Slot) ──
        # Evaluasi seluruh kandidat pasangan (slot, vehicle)
        # Menghitung skor afinitas: kedalaman penetrasi poligon, kedekatan ke centroid, IoU bbox, dan hysteresis bonus
        candidates: List[Tuple[float, str, TrackResult]] = []

        for slot in active_slots:
            s_id = slot.zone_id
            state = self.slot_states[s_id]
            geom = slot_geometries[s_id]
            pts_scaled = geom["pts_scaled"]
            slot_cx = geom["cx"]
            slot_cy = geom["cy"]
            slot_diag = geom["diag"]
            slot_bbox = geom["bbox"]

            for vt in vehicle_tracks:
                x1, y1, x2, y2 = vt.bbox
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                wheel_pt = (cx, float(y2))
                lower_pt = (cx, float(y1 + (y2 - y1) * 0.75))
                center_pt = (cx, cy)

                # Ukur penetrasi titik kontak ke poligon (margin toleransi ketat -3.0px)
                d_wheel = cv2.pointPolygonTest(pts_scaled, wheel_pt, True)
                d_lower = cv2.pointPolygonTest(pts_scaled, lower_pt, True)
                d_center = cv2.pointPolygonTest(pts_scaled, center_pt, True)
                d_max = max(d_wheel, d_lower, d_center)

                # IoU bbox kendaraan vs slot bounding box
                iou_slot = bbox_iou(vt.bbox, slot_bbox)

                # IoU anchor dengan bbox kendaraan yang tersimpan di slot OCCUPIED
                iou_anchor = 0.0
                if state.phase == "OCCUPIED" and state.last_bbox is not None:
                    iou_anchor = bbox_iou(vt.bbox, state.last_bbox)

                # Kriteria kandidat: memiliki kontak spasial nyata ATAU kecocokan anchor stabil
                # Longgarkan threshold iou_anchor menjadi 0.15 untuk slot latched agar bayangan atap tidak menggugurkan kandidat
                anchor_threshold = 0.15 if state.latch_occupied else 0.25
                if d_max >= -3.0 or iou_anchor >= anchor_threshold:
                    dist_to_center = math.hypot(cx - slot_cx, cy - slot_cy)
                    dist_norm = dist_to_center / slot_diag

                    # Bonus kontinuitas untuk slot yang sudah mantap OCCUPIED oleh ID yang sama
                    continuity_bonus = 25.0 if (state.track_id == vt.track_id and state.phase == "OCCUPIED") else 0.0
                    latch_bonus = 20.0 if state.latch_occupied else 0.0

                    # Formula Skor Afinitas Komprehensif:
                    # Semakin dalam di dalam poligon (d_max tinggi), semakin dekat ke pusat slot (dist_norm rendah),
                    # dan semakin besar IoU dengan slot, semakin tinggi skornya.
                    score = (d_max * 2.5) - (dist_norm * 30.0) + (iou_slot * 35.0) + (iou_anchor * 30.0) + continuity_bonus + latch_bonus
                    candidates.append((score, s_id, vt))

        # Urutkan kandidat dari skor afinitas tertinggi ke terendah (Greedy Optimal Assignment)
        candidates.sort(key=lambda c: c[0], reverse=True)

        assigned_slot_to_track: Dict[str, TrackResult] = {}
        assigned_track_ids: Set[int] = set()

        for score, s_id, vt in candidates:
            # 1 Slot hanya boleh diisi maksimal 1 mobil, dan 1 Mobil hanya boleh mengklaim maksimal 1 slot!
            if s_id not in assigned_slot_to_track and vt.track_id not in assigned_track_ids:
                assigned_slot_to_track[s_id] = vt
                assigned_track_ids.add(vt.track_id)

        # ── State Transitions Tiap Slot ────────────────────────────────────────
        for slot in active_slots:
            slot_id = slot.zone_id
            state = self.slot_states[slot_id]
            geom = slot_geometries[slot_id]
            pts_scaled = geom["pts_scaled"]

            matched_track = assigned_slot_to_track.get(slot_id)

            if matched_track is not None:
                # Kendaraan terdeteksi di dalam poligon — 3 cabang kepemilikan:
                if state.track_id == matched_track.track_id:
                    # Kasus 1: ID sama — kendaraan yang sama
                    if state.first_seen_time is not None:
                        # 1a: Kendaraan sudah pernah terlihat → akumulasi dwell normal
                        state.dwell_duration = current_time - state.first_seen_time
                    else:
                        # 1b: Kendaraan baru pertama kali masuk poligon (setelah tripwire A_TO_B)
                        state.first_seen_time = current_time
                        state.dwell_duration = 0.0
                    state.vehicle_class = matched_track.class_label
                    state.last_seen_time = current_time
                    state.last_bbox = matched_track.bbox
                    state.polygon_clear_since = None  # reset timer kosong

                elif state.phase in ("OCCUPIED", "ENTERING", "LEAVING") and self._is_same_vehicle(
                    state, matched_track, pts_scaled, current_time
                ):
                    # Kasus 2: ID berbeda TAPI spasial/IoU cocok → ID CHURN
                    # Update ID secara senyap TANPA mereset dwell timer atau fase
                    state.track_id = matched_track.track_id
                    state.vehicle_class = matched_track.class_label
                    state.last_seen_time = current_time
                    state.last_bbox = matched_track.bbox
                    state.polygon_clear_since = None  # reset timer kosong
                    # Pertahankan first_seen_time — hitung ulang dwell dari referensi asal
                    if state.first_seen_time is not None:
                        state.dwell_duration = current_time - state.first_seen_time

                else:
                    # Kasus 3: Kendaraan benar-benar baru (atau slot dari VACANT/kosong)
                    state.track_id = matched_track.track_id
                    state.vehicle_class = matched_track.class_label
                    state.first_seen_time = current_time
                    state.dwell_duration = 0.0
                    state.last_seen_time = current_time
                    state.last_bbox = matched_track.bbox
                    state.polygon_clear_since = None

                if is_warmup:
                    # Evaluasi baseline okupansi slot pada masa cold-start:
                    # Kendaraan yang terdeteksi di dalam poligon langsung diberi status OCCUPIED
                    state.phase = "OCCUPIED"
                    state.dwell_duration = max(state.dwell_duration, self.dwell_threshold_sec)
                    state.leaving_since = None
                else:
                    if state.phase == "VACANT":
                        # Fallback dwell (tanpa tripwire crossing terdeteksi)
                        if state.dwell_duration >= self.dwell_threshold_sec:
                            state.phase = "OCCUPIED"
                    elif state.phase == "ENTERING":
                        # Menunggu konfirmasi dwell di dalam slot
                        if state.dwell_duration >= self.dwell_threshold_sec:
                            state.phase = "OCCUPIED"
                    elif state.phase == "LEAVING":
                        # Manuver maju-mundur (re-park debounce dalam window 5 detik)
                        state.phase = "OCCUPIED"
                        state.leaving_since = None
                        state.polygon_clear_since = None
                    elif state.phase == "OCCUPIED":
                        state.leaving_since = None

                # Latch occupancy jika kendaraan telah terparkir stabil melampaui ambang batas dwell
                if state.phase == "OCCUPIED" and state.dwell_duration >= state.latch_dwell_threshold_sec:
                    state.latch_occupied = True

            else:
                # Tidak ada kendaraan terdeteksi di dalam poligon
                if state.track_id is not None and state.track_id in assigned_track_ids and not state.latch_occupied:
                    # Mobil ini telah terbukti terparkir di slot lain secara eksklusif (Exclusive Assignment)
                    # Segera bebaskan slot ini kembali ke status VACANT hanya jika slot BELUM latched (belum stabil)
                    state.phase = "VACANT"
                    state.track_id = None
                    state.vehicle_class = None
                    state.first_seen_time = None
                    state.dwell_duration = 0.0
                    state.leaving_since = None
                    state.polygon_clear_since = None
                    state.tripwire_crossed_in_time = None
                elif state.phase == "ENTERING":
                    # Timeout jika sinyal masuk sudah lewat tanpa kendaraan masuk poligon
                    if (
                        state.tripwire_crossed_in_time is not None
                        and (current_time - state.tripwire_crossed_in_time) > state.entering_timeout_sec
                    ):
                        state.phase = "VACANT"
                        state.track_id = None
                        state.first_seen_time = None
                        state.dwell_duration = 0.0
                        state.tripwire_crossed_in_time = None
                elif state.phase == "OCCUPIED":
                    # Penguatan hysteresis fisik ganda:
                    # Slot hanya boleh LEAVING setelah poligon terbukti kosong >= confirm_threshold.
                    # Gunakan threshold adaptif: 10.0 detik untuk slot latched, atau vacant_confirm_sec (5.0s) normal.
                    confirm_threshold = 10.0 if state.latch_occupied else state.vacant_confirm_sec
                    if state.last_seen_time > 0 and (current_time - state.last_seen_time) > 0.5:
                        # Mulai atau lanjutkan timer poligon kosong
                        if state.polygon_clear_since is None:
                            state.polygon_clear_since = current_time
                        elif (current_time - state.polygon_clear_since) >= confirm_threshold:
                            state.phase = "LEAVING"
                            state.leaving_since = current_time
                            state.polygon_clear_since = None
                    else:
                        # Kendaraan terlihat di frame sebelumnya (baru hilang < 0.5s), reset timer
                        state.polygon_clear_since = None
                elif state.phase == "LEAVING":
                    # Konfirmasi poligon kosong setelah masa leave / grace period
                    leave_time = state.leaving_since if state.leaving_since is not None else state.last_seen_time
                    if leave_time > 0 and (current_time - leave_time) >= state.exit_grace_sec:
                        state.phase = "VACANT"
                        # Reset latch_occupied secara legal HANYA pada transisi final LEAVING -> VACANT
                        state.latch_occupied = False
                        state.track_id = None
                        state.vehicle_class = None
                        state.first_seen_time = None
                        state.dwell_duration = 0.0
                        state.leaving_since = None
                        state.tripwire_crossed_in_time = None
                elif state.phase == "VACANT":
                    if state.last_seen_time > 0 and (current_time - state.last_seen_time) > 2.0:
                        state.track_id = None
                        state.vehicle_class = None
                        state.first_seen_time = None
                        state.dwell_duration = 0.0

        occupied_count = sum(1 for s in self.slot_states.values() if s.phase == "OCCUPIED")
        available_count = max(0, total_capacity - occupied_count)
        return {
            "total_slots": total_capacity,
            "occupied_slots": occupied_count,
            "available_slots": available_count,
            "gate_in": self.gate_in_count,
            "gate_out": self.gate_out_count,
            "slot_states": self.slot_states,
        }

    def render_overlay(
        self,
        canvas: np.ndarray,
        polygons: List[ROIZone],
        tripwires: List[TripwireRule],
        tracks: List[TrackResult],
        scale_x: float,
        scale_y: float,
        parking_stats: Dict[str, Any],
        current_time: float = 0.0,
        fps: float = 20.0,
        camera_id: str = "cam_01",
        camera_name: str = "Koridor Utama",
    ) -> None:
        """
        Render visual overlay CCTV Minimalist (Capacity & Slot Status Only):
        1. Bounding Box & Vehicle Labels DIHAPUS penuh (declutter visual lantai).
        2. Titik roda, panah arah, titik P1/P2, dan teks label tripwire DIHAPUS.
        3. Tripwire: Garis hairline 1px tipis oranye/amber tanpa embel-embel teks.
        4. Poligon Petak Parkir:
           - VACANT (Kosong): Outline warna Cyan/Emerald transparan halus tanpa teks di aspal.
           - OCCUPIED (Terisi): Outline Merah tipis + fill transparan lembut (alpha=0.10)
             + mini label di centroid: [S{num}] (fontScale=0.35).
        5. Capacity HUD: 1 baris ringkas di pojok kanan atas:
           PARKING: {occupied}/{total} OCCUPIED | TERISI: [{occupied_slots_str}]
        """
        if parking_stats.get("parking_mode") == "motorcycle_block" or self.parking_mode == "motorcycle_block":
            self._render_block_overlay(
                canvas=canvas,
                polygons=polygons,
                tripwires=tripwires,
                tracks=tracks,
                scale_x=scale_x,
                scale_y=scale_y,
                parking_stats=parking_stats,
                current_time=current_time,
            )
            return

        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        total_slots = parking_stats.get("total_slots", len(polygons))
        slot_states = parking_stats.get("slot_states", self.slot_states)
        occupied_slots = parking_stats.get(
            "occupied_slots",
            sum(1 for s in slot_states.values() if s.phase == "OCCUPIED")
        )
        available_slots = parking_stats.get("available_slots", max(0, total_slots - occupied_slots))

        active_slots = [p for p in polygons if p.active]

        # ── 1. Minimalist Slot Polygons (1px Outline & Centroid Label) ──
        occupied_polys = []
        occupied_badges = []

        for slot in active_slots:
            pts_scaled = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )

            state = slot_states.get(slot.zone_id)
            phase = state.phase if state else "VACANT"

            if phase == "OCCUPIED":
                # Outline 1px Merah tipis + Fill transparan 0.10 alpha
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                occupied_polys.append(pts_scaled)
                # Mini label kecil di centroid: [S{num}]
                cx = int(np.mean([pt[0] for pt in pts_scaled]))
                cy = int(np.mean([pt[1] for pt in pts_scaled]))
                occupied_badges.append((cx, cy, f"[S{state.slot_num}]"))
            elif phase == "ENTERING":
                # Outline 1px Amber halus
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 215, 255), thickness=1, lineType=cv2.LINE_AA)
            elif phase == "LEAVING":
                # Outline 1px Kuning tipis
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                # Slot VACANT: outline 1px Cyan/Emerald transparan halus tanpa teks apapun di aspal
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(220, 220, 0), thickness=1, lineType=cv2.LINE_AA)

        # Arsiran fill transparan lembut pada slot OCCUPIED (alpha = 0.10)
        if occupied_polys:
            overlay = canvas.copy()
            for opp in occupied_polys:
                cv2.fillPoly(overlay, [opp], color=(0, 0, 220))
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # Mini label kecil di centroid slot OCCUPIED: [S{num}]
        for cx, cy, text in occupied_badges:
            self._draw_slot_badge(canvas, text, cx, cy)

        # ── 2. Minimalist Tripwire (Hairline 1px Tipis Oranye/Amber) ──
        active_tripwires = [tw for tw in tripwires if tw.active]
        for tw in active_tripwires:
            p1_scaled = (int(tw.p1.x * sx), int(tw.p1.y * sy))
            p2_scaled = (int(tw.p2.x * sx), int(tw.p2.y * sy))

            flash_until = self.tripwire_flash.get(tw.tripwire_id, 0.0)
            is_flashing = (current_time > 0 and current_time < flash_until)
            tw_color = (255, 255, 255) if is_flashing else (0, 165, 255)

            # Cukup garis hairline 1px tipis oranye/amber tanpa embel-embel teks/titik
            cv2.line(canvas, p1_scaled, p2_scaled, tw_color, 1, cv2.LINE_AA)

        # ── 3. Minimalist Capacity HUD (1 Baris Ringkas Pojok Kanan Atas) ──
        occ_slots = [
            f"S{s.slot_num}"
            for s in sorted(slot_states.values(), key=lambda x: x.slot_num)
            if s.phase == "OCCUPIED"
        ]
        occ_text = ", ".join(occ_slots) if occ_slots else "-"

        hud_line = f"PARKING: {occupied_slots}/{total_slots} OCCUPIED | TERISI: [{occ_text}]"
        self._draw_hud_pill_top_right(canvas, hud_line, available_slots)

    def _draw_hud_pill_top_right(
        self,
        canvas: np.ndarray,
        line: str,
        available: int,
    ) -> None:
        """
        Render status HUD di POJOK KANAN ATAS frame dengan Dark Semi-Transparent Pill
        (1 baris ringkas).
        Pojok kiri atas sengaja dikosongkan agar OSD timestamp asli kamera terlihat jelas.
        """
        font_sc = 0.35
        font_thick = 1
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), _ = cv2.getTextSize(line, font_face, font_sc, font_thick)

        pad_h, pad_v = 10, 6
        pill_w = tw + pad_h * 2
        pill_h = th + pad_v * 2

        margin_right = 8
        margin_top = 8

        x2 = canvas.shape[1] - margin_right
        x1 = max(0, x2 - pill_w)
        y1 = margin_top
        y2 = min(canvas.shape[0] - 1, y1 + pill_h)

        # Background semi-transparan gelap (alpha 0.75)
        sub = canvas[y1:y2, x1:x2]
        bg_color = np.full_like(sub, (16, 20, 28), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.25, bg_color, 0.75, 0)

        # Border tipis 1px
        border_col = (0, 220, 100) if available > 0 else (0, 60, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1, cv2.LINE_AA)

        # Teks 1 baris
        tx = x1 + pad_h
        ty = y1 + pad_v + th
        text_col = (0, 235, 120) if available > 0 else (0, 80, 255)
        cv2.putText(canvas, line, (tx, ty), font_face, font_sc, text_col, font_thick, cv2.LINE_AA)

    def _render_block_overlay(
        self,
        canvas: np.ndarray,
        polygons: List[ROIZone],
        tripwires: List[TripwireRule],
        tracks: List[TrackResult],
        scale_x: float,
        scale_y: float,
        parking_stats: Dict[str, Any],
        current_time: float = 0.0,
    ) -> None:
        """
        Render visual overlay CCTV Minimalist untuk Block Motor Parking (cam_03):
        1. Poligon area: outline tebal 2px hijau jika ada slot, merah jika penuh + fill 0.10 alpha.
        2. Centroid label: "{occupied}/{total} MOTOR".
        3. Tripwire: garis hairline 1px tipis oranye/amber (flash 1.0s putih).
        4. HUD 2-baris di pojok kanan atas:
           Baris 1: PARKING: {occupied}/{total} TERISI | KOSONG: {available} UNIT
           Baris 2: IN: {gate_in} | OUT: {gate_out}
        """
        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        occupied = parking_stats.get("occupied_slots", 0)
        total = parking_stats.get("total_slots", self.block_capacity)
        available = parking_stats.get("available_slots", max(0, total - occupied))
        gate_in = parking_stats.get("gate_in", self.gate_in_count)
        gate_out = parking_stats.get("gate_out", self.gate_out_count)

        zone_color = (0, 220, 100) if available > 0 else (0, 60, 255)

        active_slots = [p for p in polygons if p.active]
        for slot in active_slots:
            pts = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )
            # Outline tebal 2px
            cv2.polylines(canvas, [pts], isClosed=True, color=zone_color, thickness=2, lineType=cv2.LINE_AA)

            # Fill transparan 0.10 alpha
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color=zone_color)
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # ── Visualisasi Bounding Box & Badge Motor Terdeteksi di Dalam Poligon ──
        if active_slots:
            pts_zone = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in active_slots[0].points],
                dtype=np.int32,
            )
            # Visualisasi kotak motor: Utamakan unit ter-latch dari memori temporal agar stabil tanpa flicker
            # Filter render: tampilkan unit yang masih "hidup" (belum melebihi TTL 15s)
            if self._stationary_motor_units:
                render_units = sorted(
                    self._stationary_motor_units.values(),
                    key=lambda u: u.centroid[0],
                )
                render_items = [
                    (u.bbox, u.confidence)
                    for u in render_units
                    if (current_time - u.last_seen_time) <= 15.0
                ]
            else:
                candidate_render: List[TrackResult] = []
                for track in tracks:
                    if track.class_label not in self.vehicle_classes:
                        continue

                    rx1, ry1, rx2, ry2 = track.bbox
                    cx = float((rx1 + rx2) / 2.0)
                    wheel_y = float(ry2)
                    wheel_pt = (int(cx), int(wheel_y))

                    wheel_in = cv2.pointPolygonTest(pts_zone, wheel_pt, False) >= 0
                    if not wheel_in:
                        continue

                    overlap_ratio = bbox_polygon_overlap_ratio(track.bbox, pts_zone)
                    if overlap_ratio < 0.20:
                        continue

                    candidate_render.append(track)

                valid_render = deduplicate_motorcycle_tracks(
                    candidate_render,
                    iou_thresh=0.30,
                    ios_thresh=0.50,
                    max_centroid_dist_px=28.0,
                )
                render_items = [(t.bbox, t.confidence) for t in valid_render]

            for motor_idx, (r_bbox, r_conf) in enumerate(render_items, 1):
                x1, y1, x2, y2 = [int(round(v)) for v in r_bbox]
                cx = float((r_bbox[0] + r_bbox[2]) / 2.0)
                wheel_y = float(r_bbox[3])

                # Kotak tipis 1px warna Kuning (Yellow)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

                # Titik tumpu roda bawah (lingkaran kecil hijau solid radius 2px)
                cv2.circle(canvas, (int(cx), int(wheel_y)), 2, (0, 255, 0), -1, cv2.LINE_AA)

                # Mini badge di atas box motor: #{idx} ({conf:.2f})
                badge_text = f"#{motor_idx} ({r_conf:.2f})"

                font_scale = 0.32
                font_thick = 1
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(badge_text, font_face, font_scale, font_thick)

                bx1 = max(0, x1)
                by1 = max(0, y1 - th - 3)
                bx2 = min(canvas.shape[1] - 1, bx1 + tw + 4)
                by2 = min(canvas.shape[0] - 1, by1 + th + 3)

                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 24, 32), -1)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    badge_text,
                    (bx1 + 2, by1 + th),
                    font_face,
                    font_scale,
                    (0, 255, 255),
                    font_thick,
                    cv2.LINE_AA,
                )

        # Label kapasitas di centroid area poligon (digambar setelah box agar selalu di atas)
        for slot in active_slots:
            pts = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )
            cx = int(np.mean([pt[0] for pt in pts]))
            cy = int(np.mean([pt[1] for pt in pts]))
            label = f"{occupied}/{total} MOTOR"
            self._draw_slot_badge(canvas, label, cx, cy)

        # Tripwire render (dengan flash)
        active_tripwires = [tw for tw in tripwires if tw.active]
        for tw in active_tripwires:
            p1_scaled = (int(tw.p1.x * sx), int(tw.p1.y * sy))
            p2_scaled = (int(tw.p2.x * sx), int(tw.p2.y * sy))
            flash_until = self.tripwire_flash.get(tw.tripwire_id, 0.0)
            is_flashing = (current_time > 0 and current_time < flash_until)
            tw_color = (255, 255, 255) if is_flashing else (0, 165, 255)
            cv2.line(canvas, p1_scaled, p2_scaled, tw_color, 1, cv2.LINE_AA)

        # HUD 1-baris di pojok kanan atas
        hud_line = f"PARKING: {occupied}/{total} TERISI | KOSONG: {available} UNIT"
        self._draw_hud_pill_top_right(canvas, hud_line, available)

    def _draw_hud_pill_top_right_dual(
        self,
        canvas: np.ndarray,
        line1: str,
        line2: str,
        available: int,
    ) -> None:
        """Render HUD 2-baris di pojok kanan atas dengan Dark Semi-Transparent Pill."""
        font_sc = 0.35
        font_thick = 1
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        (tw1, th1), _ = cv2.getTextSize(line1, font_face, font_sc, font_thick)
        (tw2, th2), _ = cv2.getTextSize(line2, font_face, font_sc, font_thick)

        pad_h, pad_v = 10, 6
        line_gap = 4
        pill_w = max(tw1, tw2) + pad_h * 2
        pill_h = th1 + th2 + pad_v * 2 + line_gap

        margin_right = 8
        margin_top = 8

        x2 = canvas.shape[1] - margin_right
        x1 = max(0, x2 - pill_w)
        y1 = margin_top
        y2 = min(canvas.shape[0] - 1, y1 + pill_h)

        sub = canvas[y1:y2, x1:x2]
        bg_color = np.full_like(sub, (16, 20, 28), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.25, bg_color, 0.75, 0)

        border_col = (0, 220, 100) if available > 0 else (0, 60, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1, cv2.LINE_AA)

        tx = x1 + pad_h
        ty1 = y1 + pad_v + th1
        ty2 = ty1 + th2 + line_gap
        text_col1 = (0, 235, 120) if available > 0 else (0, 80, 255)
        text_col2 = (180, 180, 180)
        cv2.putText(canvas, line1, (tx, ty1), font_face, font_sc, text_col1, font_thick, cv2.LINE_AA)
        cv2.putText(canvas, line2, (tx, ty2), font_face, font_sc, text_col2, font_thick, cv2.LINE_AA)


    def _draw_slot_badge(
        self,
        canvas: np.ndarray,
        text: str,
        cx: int,
        cy: int,
    ) -> None:
        """Render mini label kecil di centroid petak parkir OCCUPIED: [S{num}] (fontScale=0.35)."""
        font_scale = 0.35
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pad_h, pad_v = 4, 3
        bx1 = max(2, cx - tw // 2 - pad_h)
        by1 = max(2, cy - th // 2 - pad_v)
        bx2 = min(canvas.shape[1] - 3, bx1 + tw + pad_h * 2)
        by2 = min(canvas.shape[0] - 3, by1 + th + pad_v * 2)

        # Semi-transparan dark background dengan border merah 1px
        sub = canvas[by1:by2, bx1:bx2]
        dark_bg = np.full_like(sub, (15, 15, 30), dtype=np.uint8)
        canvas[by1:by2, bx1:bx2] = cv2.addWeighted(sub, 0.25, dark_bg, 0.75, 0)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            text,
            (bx1 + pad_h, by1 + pad_v + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    def _draw_micro_chevron(
        self,
        canvas: np.ndarray,
        p1_x: int,
        p1_y: int,
        p2_x: int,
        p2_y: int,
        direction: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Gambarkan chevron mikro ramping (ukuran ~5px) di tengah garis tripwire."""
        dx = float(p2_x - p1_x)
        dy = float(p2_y - p1_y)
        length = max(1.0, np.hypot(dx, dy))
        nx = -dy / length
        ny = dx / length
        ux = dx / length
        uy = dy / length

        mid_x = (p1_x + p2_x) / 2.0
        mid_y = (p1_y + p2_y) / 2.0
        chev_depth = 5.0
        chev_wing = 4.0

        if direction in ("A_TO_B", "BOTH"):
            tip_x = int(mid_x + nx * chev_depth)
            tip_y = int(mid_y + ny * chev_depth)
            w1_x = int(mid_x - ux * chev_wing)
            w1_y = int(mid_y - uy * chev_wing)
            w2_x = int(mid_x + ux * chev_wing)
            w2_y = int(mid_y + uy * chev_wing)
            cv2.line(canvas, (w1_x, w1_y), (tip_x, tip_y), color, 1, cv2.LINE_AA)
            cv2.line(canvas, (w2_x, w2_y), (tip_x, tip_y), color, 1, cv2.LINE_AA)

        if direction in ("B_TO_A", "BOTH"):
            tip_x = int(mid_x - nx * chev_depth)
            tip_y = int(mid_y - ny * chev_depth)
            w1_x = int(mid_x - ux * chev_wing)
            w1_y = int(mid_y - uy * chev_wing)
            w2_x = int(mid_x + ux * chev_wing)
            w2_y = int(mid_y + uy * chev_wing)
            cv2.line(canvas, (w1_x, w1_y), (tip_x, tip_y), color, 1, cv2.LINE_AA)
            cv2.line(canvas, (w2_x, w2_y), (tip_x, tip_y), color, 1, cv2.LINE_AA)

    def _draw_mini_badge(
        self,
        canvas: np.ndarray,
        text: str,
        cx: int,
        cy: int,
        outline_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int] = (15, 15, 45),
    ) -> None:
        """Helper backward-compatible render mini-badge."""
        self._draw_slot_badge(canvas, text, cx, cy)

    def _draw_telemetry_banner(
        self, canvas: np.ndarray, text: str, available: int, total: int
    ) -> None:
        """Helper backward-compatible telemetry banner."""
        pass
