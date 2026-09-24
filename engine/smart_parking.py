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

import time
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

    @property
    def occupied(self) -> bool:
        """Backward-compatible property: True jika phase == OCCUPIED."""
        return self.phase == "OCCUPIED"


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
    ) -> None:
        self._total_slots_override = total_slots
        self.dwell_threshold_sec = dwell_threshold_sec
        self.vehicle_classes = vehicle_classes or self.DEFAULT_VEHICLE_CLASSES
        # Map: slot_id -> SlotState
        self.slot_states: Dict[str, SlotState] = {}
        # Gate traffic counters
        self.gate_in_count: int = 0
        self.gate_out_count: int = 0
        # Flash trigger tracker: tripwire_id -> flash_until timestamp
        self.tripwire_flash: Dict[str, float] = {}

    @property
    def total_slots(self) -> int:
        """Kapasitas total dinamis berdasarkan slot yang terdaftar."""
        if self._total_slots_override is not None:
            return self._total_slots_override
        return len(self.slot_states)

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

        # Syarat Spasial A (kuat): titik kontak roda / lower body / center di dalam poligon slot (margin toleransi -10px)
        if (
            cv2.pointPolygonTest(pts_scaled, wheel_pt, True) >= -10.0
            or cv2.pointPolygonTest(pts_scaled, lower_pt, True) >= -10.0
            or cv2.pointPolygonTest(pts_scaled, center_pt, True) >= -10.0
        ):
            return True

        # Syarat Spasial B (toleran): IoU bbox baru vs bbox tersimpan >= threshold
        if state.last_bbox is not None:
            iou = bbox_iou(new_track.bbox, state.last_bbox)
            if iou >= iou_threshold:
                return True

        return False

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

            state = self.slot_states[slot_id]
            pts_scaled = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )

            # Cari apakah ada kontak kendaraan dalam slot ini (wheel contact, lower body, centroid, atau IoU anchor)
            matched_track: Optional[TrackResult] = None
            for vt in vehicle_tracks:
                x1, y1, x2, y2 = vt.bbox
                cx = float((x1 + x2) / 2.0)
                wheel_pt = (cx, float(y2))
                lower_pt = (cx, float(y1 + (y2 - y1) * 0.75))
                center_pt = (cx, float((y1 + y2) / 2.0))

                # Toleransi spasial margin 10px (anti-flickering roda mepet bibir slot)
                is_contact = (
                    cv2.pointPolygonTest(pts_scaled, wheel_pt, True) >= -10.0
                    or cv2.pointPolygonTest(pts_scaled, lower_pt, True) >= -10.0
                    or cv2.pointPolygonTest(pts_scaled, center_pt, True) >= -10.0
                )

                # Toleransi IoU anchor: jika slot sudah OCCUPIED dan posisi mobil stabil dengan last_bbox
                if not is_contact and state.phase == "OCCUPIED" and state.last_bbox is not None:
                    if bbox_iou(vt.bbox, state.last_bbox) >= 0.30:
                        is_contact = True

                if is_contact:
                    matched_track = vt
                    break

            # ── State Transitions ─────────────────────────────────────────
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

            else:
                # Tidak ada kendaraan terdeteksi di dalam poligon
                if state.phase == "ENTERING":
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
                    # Slot hanya boleh LEAVING setelah poligon terbukti kosong >= vacant_confirm_sec.
                    if state.last_seen_time > 0 and (current_time - state.last_seen_time) > 0.5:
                        # Mulai atau lanjutkan timer poligon kosong
                        if state.polygon_clear_since is None:
                            state.polygon_clear_since = current_time
                        elif (current_time - state.polygon_clear_since) >= state.vacant_confirm_sec:
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
           PARKING: {available}/{total} AVAILABLE | TERISI: [{occupied_slots_str}]
        """
        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        available_slots = parking_stats.get("available_slots", len(polygons))
        total_slots = parking_stats.get("total_slots", len(polygons))
        slot_states = parking_stats.get("slot_states", self.slot_states)

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

        hud_line = f"PARKING: {available_slots}/{total_slots} AVAILABLE | TERISI: [{occ_text}]"
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
