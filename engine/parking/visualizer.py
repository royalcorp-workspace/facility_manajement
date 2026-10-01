"""
engine.parking.visualizer
=========================
OpenCV rendering and visual overlay components for Smart Parking.
Handles:
- Car slot polygon outlines and occupied centroid badges (cam_01).
- Minimalist tripwire rendering with event flash.
- Top-right corner HUD pills (single/dual line semi-transparent).
- Motorcycle block zone outline and capacity badge (cam_03).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from engine.config_loader import ROIZone, TripwireRule
from engine.parking.base import SlotState, StationaryMotorUnit
from engine.parking.motorcycle_block import deduplicate_motorcycle_tracks, is_valid_motorcycle_anatomy
from engine.parking.spatial import bbox_polygon_overlap_ratio
from engine.tracker_interface import TrackResult


class ParkingVisualizer:
    """
    OpenCV visual overlay renderer for smart parking streams.
    """

    def __init__(
        self,
        parking_mode: str = "slot",
        block_capacity: int = 30,
        block_exclusion_x_1080p: Optional[int] = None,
        stationary_dwell_sec: float = 0.0,
        vehicle_classes: Optional[Set[str]] = None,
    ) -> None:
        self.parking_mode = parking_mode
        self.block_capacity = block_capacity
        self._block_exclusion_x_1080p = block_exclusion_x_1080p
        self.stationary_dwell_sec = stationary_dwell_sec
        self.vehicle_classes = set(vehicle_classes) if vehicle_classes is not None else {"car", "truck", "bus"}

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
        tripwire_flash: Optional[Dict[str, float]] = None,
        stationary_motor_units: Optional[Dict[int, StationaryMotorUnit]] = None,
        slot_states: Optional[Dict[str, SlotState]] = None,
        gate_in_count: int = 0,
        gate_out_count: int = 0,
    ) -> None:
        """
        Render visual overlay CCTV Minimalist:
        - Car slot mode (cam_01)
        - Motorcycle block mode (cam_03)
        """
        mode = parking_stats.get("parking_mode", self.parking_mode)
        if mode == "motorcycle_block":
            self._render_block_overlay(
                canvas=canvas,
                polygons=polygons,
                tripwires=tripwires,
                tracks=tracks,
                scale_x=scale_x,
                scale_y=scale_y,
                parking_stats=parking_stats,
                current_time=current_time,
                stationary_motor_units=stationary_motor_units,
                gate_in_count=gate_in_count,
                gate_out_count=gate_out_count,
            )
            return

        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        total_slots = parking_stats.get("total_slots", len(polygons))
        active_slot_states = parking_stats.get("slot_states", slot_states if slot_states is not None else {})
        occupied_slots = parking_stats.get(
            "occupied_slots",
            sum(1 for s in active_slot_states.values() if getattr(s, "phase", "") == "OCCUPIED"),
        )
        available_slots = parking_stats.get("available_slots", max(0, total_slots - occupied_slots))

        active_slots = [p for p in polygons if p.active]

        # 1. Minimalist Slot Polygons (1px Outline & Centroid Label)
        occupied_polys = []
        occupied_badges = []

        for slot in active_slots:
            pts_scaled = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )

            state = active_slot_states.get(slot.zone_id)
            phase = state.phase if state else "VACANT"

            if phase == "OCCUPIED":
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                occupied_polys.append(pts_scaled)
                cx = int(np.mean([pt[0] for pt in pts_scaled]))
                cy = int(np.mean([pt[1] for pt in pts_scaled]))
                slot_num = state.slot_num if state else "?"
                occupied_badges.append((cx, cy, f"[S{slot_num}]"))
            elif phase == "ENTERING":
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 215, 255), thickness=1, lineType=cv2.LINE_AA)
            elif phase == "LEAVING":
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                cv2.polylines(canvas, [pts_scaled], isClosed=True, color=(220, 220, 0), thickness=1, lineType=cv2.LINE_AA)

        # Fill transparan lembut OCCUPIED (alpha = 0.10)
        if occupied_polys:
            overlay = canvas.copy()
            for opp in occupied_polys:
                cv2.fillPoly(overlay, [opp], color=(0, 0, 220))
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # Mini label kecil di centroid slot OCCUPIED: [S{num}]
        for cx, cy, text in occupied_badges:
            self._draw_slot_badge(canvas, text, cx, cy)

        # 2. Minimalist Tripwire (Hairline 1px Tipis Oranye/Amber)
        active_tripwires = [tw for tw in tripwires if tw.active]
        tw_flash_map = tripwire_flash or {}
        for tw in active_tripwires:
            p1_scaled = (int(tw.p1.x * sx), int(tw.p1.y * sy))
            p2_scaled = (int(tw.p2.x * sx), int(tw.p2.y * sy))

            flash_until = tw_flash_map.get(tw.tripwire_id, 0.0)
            is_flashing = current_time > 0 and current_time < flash_until
            tw_color = (255, 255, 255) if is_flashing else (0, 165, 255)

            cv2.line(canvas, p1_scaled, p2_scaled, tw_color, 1, cv2.LINE_AA)

        # 3. Minimalist Capacity HUD (1 Baris Ringkas Pojok Kanan Atas)
        occ_slots = [
            f"S{s.slot_num}"
            for s in sorted(active_slot_states.values(), key=lambda x: getattr(x, "slot_num", 0))
            if getattr(s, "phase", "") == "OCCUPIED"
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
        Render status HUD di POJOK KANAN ATAS frame dengan Dark Semi-Transparent Pill (1 baris ringkas).
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

        sub = canvas[y1:y2, x1:x2]
        bg_color = np.full_like(sub, (16, 20, 28), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.25, bg_color, 0.75, 0)

        border_col = (0, 220, 100) if available > 0 else (0, 60, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1, cv2.LINE_AA)

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
        stationary_motor_units: Optional[Dict[int, StationaryMotorUnit]] = None,
        gate_in_count: int = 0,
        gate_out_count: int = 0,
    ) -> None:
        """
        Render visual overlay CCTV Minimalist untuk Block Motor Parking (cam_03).
        """
        sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sy = (1.0 / scale_y) if scale_y > 1.0 else scale_y

        occupied = parking_stats.get("occupied_slots", 0)
        total = parking_stats.get("total_slots", self.block_capacity)
        available = parking_stats.get("available_slots", max(0, total - occupied))

        zone_color = (0, 220, 100) if available > 0 else (0, 60, 255)

        active_slots = [p for p in polygons if p.active]
        for slot in active_slots:
            pts = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], isClosed=True, color=zone_color, thickness=2, lineType=cv2.LINE_AA)

            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color=zone_color)
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # 2. Render unit motor terparkir (bounding box kuning tipis & badge ID/confidence)
        if active_slots:
            pts_zone = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in active_slots[0].points],
                dtype=np.int32,
            )
            drum_excl_x_render = int(self._block_exclusion_x_1080p * sx) if self._block_exclusion_x_1080p is not None else None

            render_items: List[Tuple[Tuple[float, float, float, float], float]] = []
            if stationary_motor_units:
                render_units = sorted(
                    stationary_motor_units.values(),
                    key=lambda u: u.first_seen_time,
                )
                render_items = [
                    (u.bbox, u.confidence)
                    for u in render_units
                    if (u.is_latched or (self.stationary_dwell_sec == 0.0) or (current_time > 0 and (current_time - u.first_seen_time) >= 1.2))
                    and (current_time <= 0 or (current_time - u.last_seen_time) <= 12.0)
                    and is_valid_motorcycle_anatomy(u.bbox, sx, sy)
                    and (drum_excl_x_render is None or u.centroid[0] >= drum_excl_x_render)
                ]
            else:
                candidate_render: List[TrackResult] = []
                for track in tracks:
                    if getattr(track, "class_label", None) in ("motorcycle", "bicycle") and (track.is_confirmed or current_time == 0.0):
                        rx1, ry1, rx2, ry2 = track.bbox
                        cx = float((rx1 + rx2) / 2.0)
                        wheel_y = float(ry2)
                        wheel_pt = (int(cx), int(wheel_y))

                        if drum_excl_x_render is not None and cx < drum_excl_x_render:
                            continue

                        if not is_valid_motorcycle_anatomy(track.bbox, sx, sy):
                            continue

                        wheel_in = cv2.pointPolygonTest(pts_zone, wheel_pt, True) >= -12.0
                        if not wheel_in:
                            continue
                        candidate_render.append(track)

                scale_factor = max(1.0, sx / (1.0 / 3.0))
                valid_render = deduplicate_motorcycle_tracks(
                    candidate_render,
                    iou_thresh=0.55,
                    ios_thresh=0.85,
                    min_dx_px=6.0 * scale_factor,
                    cumulative_overlap_thresh=0.70,
                )
                render_items = [(t.bbox, t.confidence) for t in valid_render]

            for idx, (r_bbox, r_conf) in enumerate(render_items, start=1):
                x1, y1, x2, y2 = [int(round(coord)) for coord in r_bbox]
                cx = int(round((r_bbox[0] + r_bbox[2]) / 2.0))
                wheel_y = int(round(r_bbox[3]))

                # Kotak kuning tipis (thickness = 1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

                # Titik kontak roda bawah (cx, y2) lingkaran hijau radius 2 px
                cv2.circle(canvas, (cx, wheel_y), 2, (0, 255, 120), -1, cv2.LINE_AA)

                # Mini-badge / label ID dan confidence
                cv2.putText(
                    canvas,
                    f"#{idx} ({r_conf:.2f})",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Label kapasitas di centroid area poligon
        for slot in active_slots:
            pts = np.array(
                [[int(pt.x * sx), int(pt.y * sy)] for pt in slot.points],
                dtype=np.int32,
            )
            cx = int(np.mean([pt[0] for pt in pts]))
            cy = int(np.mean([pt[1] for pt in pts]))
            label = f"{occupied}/{total} MOTOR"
            self._draw_slot_badge(canvas, label, cx, cy)

        # 4. HUD 1-baris bersih di pojok kanan atas
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
        length = max(1.0, float(np.hypot(dx, dy)))
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
