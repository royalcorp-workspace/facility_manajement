"""
Modul Event Dispatcher untuk Facility Management Vision Engine.
Mengatur logika siklus hidup event (ENTER -> LINGER -> EXIT), penyimpanan bukti snapshot JPEG,
dan persistensi ke SQLite database.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from engine.config_loader import ROIZonesConfig, ROIZone
from engine.geometry import point_in_polygon, scale_points, bbox_bottom_center
from engine.preprocessor import ProcessedFrame
from engine.spatial_rules import SpatialRuleEngine
from engine.tracker_interface import TrackResult
from storage.models import (
    EVENT_CONGESTION_ALERT,
    EVENT_ENTER,
    EVENT_EXIT,
    EVENT_LINE_CROSSING,
    EVENT_LINGER,
    EVENT_WALKWAY_VIOLATION,
    LifecycleEvent,
)


@dataclass
class OpenEventState:
    """State internal event ENTER yang sedang berlangsung (unresolved)."""

    db_event_id: int
    enter_time: float
    last_seen_time: float
    camera_id: str
    zone_id: str
    track_id: int
    class_label: str
    confidence: float
    linger_triggered: bool = False


class EventDispatcher:
    """
    Event Dispatcher & Snapshot Manager dengan integrasi Aturan Spasial Lanjutan.
    """

    def __init__(self, snapshot_quality: int = 85, exit_timeout_sec: float = 3.0) -> None:
        self.snapshot_quality = snapshot_quality
        self.exit_timeout_sec = exit_timeout_sec
        # Map (camera_id, zone_id, track_id) -> OpenEventState
        self.open_events: Dict[Tuple[str, str, int], OpenEventState] = {}
        # Mesin analitik spasial tingkat lanjut
        self.spatial_engine = SpatialRuleEngine()

    def process_tracks(
        self,
        processed_frame: ProcessedFrame,
        tracks: List[TrackResult],
        roi_config: ROIZonesConfig,
        db_conn: sqlite3.Connection,
    ) -> List[LifecycleEvent]:
        """
        Evaluasi status posisi semua track terhadap ROI zones & Aturan Spasial Lanjutan,
        serta catat event ke DB.
        """
        generated_events: List[LifecycleEvent] = []
        now = processed_frame.timestamp
        iso_ts = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        cam_id = processed_frame.camera_id

        # Track IDs yang aktif saat ini
        current_track_map = {t.track_id: t for t in tracks}
        active_zones = roi_config.active_zones

        # ── 1. EVALUASI ZONA POLIGON (ENTER -> LINGER -> EXIT) DENGAN EXCLUSION MASK ──
        for zone in active_zones:
            raw_poly = [(p.x, p.y) for p in zone.points]
            # Convert polygon ke AI canvas scale
            ai_poly = scale_points(raw_poly, processed_frame.scale_x, processed_frame.scale_y)

            for track in tracks:
                if not track.is_confirmed:
                    continue

                bottom_center = bbox_bottom_center(track.bbox)
                is_inside = point_in_polygon(bottom_center, ai_poly)
                key = (cam_id, zone.zone_id, track.track_id)

                # Filter exclusion mask untuk dwell
                in_exclusion = False
                if is_inside and roi_config.active_exclusion_masks:
                    in_exclusion = self.spatial_engine.filter_exclusion(
                        track,
                        roi_config.active_exclusion_masks,
                        processed_frame.scale_x,
                        processed_frame.scale_y,
                        "dwell",
                    )

                if is_inside and not in_exclusion:
                    if key not in self.open_events:
                        # ── EVENT: ENTER ──────────────────────────────────────────
                        snapshot_path = self._generate_and_save_snapshot(
                            processed_frame=processed_frame,
                            track=track,
                            zone=zone,
                            event_type=EVENT_ENTER,
                            dwell_sec=0.0,
                        )
                        event = LifecycleEvent(
                            camera_id=cam_id,
                            zone_id=zone.zone_id,
                            event_type=EVENT_ENTER,
                            timestamp=iso_ts,
                            track_id=track.track_id,
                            class_label=track.class_label,
                            confidence=track.confidence,
                            snapshot_path=snapshot_path,
                            is_resolved=0,
                        )
                        event.insert(db_conn)

                        self.open_events[key] = OpenEventState(
                            db_event_id=event.id,
                            enter_time=now,
                            last_seen_time=now,
                            camera_id=cam_id,
                            zone_id=zone.zone_id,
                            track_id=track.track_id,
                            class_label=track.class_label,
                            confidence=track.confidence,
                            linger_triggered=False,
                        )
                        generated_events.append(event)
                    else:
                        # ── Update last seen & Cek LINGER ─────────────────────────
                        state = self.open_events[key]
                        state.last_seen_time = now
                        dwell_sec = now - state.enter_time

                        if (
                            "linger" in zone.trigger_on
                            and zone.linger_threshold_sec is not None
                            and dwell_sec >= zone.linger_threshold_sec
                            and not state.linger_triggered
                        ):
                            state.linger_triggered = True
                            snapshot_path = self._generate_and_save_snapshot(
                                processed_frame=processed_frame,
                                track=track,
                                zone=zone,
                                event_type=EVENT_LINGER,
                                dwell_sec=dwell_sec,
                            )
                            event = LifecycleEvent(
                                camera_id=cam_id,
                                zone_id=zone.zone_id,
                                event_type=EVENT_LINGER,
                                timestamp=iso_ts,
                                track_id=track.track_id,
                                class_label=track.class_label,
                                confidence=track.confidence,
                                snapshot_path=snapshot_path,
                                is_resolved=0,
                                notes=f"Dwell duration: {dwell_sec:.1f}s",
                            )
                            event.insert(db_conn)
                            generated_events.append(event)

                else:
                    # Tidak lagi di dalam zone, jika sebelumnya ada -> EXIT
                    if key in self.open_events:
                        state = self.open_events.pop(key)
                        dwell_duration_sec = now - state.enter_time

                        # Resolve all unresolved events (enter/linger) for this track in DB
                        db_conn.execute(
                            "UPDATE lifecycle_events SET is_resolved = 1, resolved_time = :resolved_time, "
                            "notes = COALESCE(:notes, notes) "
                            "WHERE camera_id = :camera_id AND zone_id = :zone_id AND track_id = :track_id AND is_resolved = 0",
                            {
                                "resolved_time": iso_ts,
                                "notes": f"Dwell duration: {dwell_duration_sec:.1f}s",
                                "camera_id": state.camera_id,
                                "zone_id": state.zone_id,
                                "track_id": state.track_id,
                            },
                        )

                        # Catat EXIT event
                        snapshot_path = self._generate_and_save_snapshot(
                            processed_frame=processed_frame,
                            track=track,
                            zone=zone,
                            event_type=EVENT_EXIT,
                            dwell_sec=dwell_duration_sec,
                        )
                        exit_event = LifecycleEvent(
                            camera_id=cam_id,
                            zone_id=zone.zone_id,
                            event_type=EVENT_EXIT,
                            timestamp=iso_ts,
                            track_id=track.track_id,
                            class_label=track.class_label,
                            confidence=track.confidence,
                            snapshot_path=snapshot_path,
                            is_resolved=1,
                            resolved_time=iso_ts,
                            notes=f"Dwell duration: {dwell_duration_sec:.1f}s",
                        )
                        exit_event.insert(db_conn)
                        generated_events.append(exit_event)

        # ── 2. EVALUASI ATURAN SPASIAL LANJUTAN (Tripwire, Barrier, Walkway, Density) ──
        # A. Directional Tripwires (Zero-Dwell Line Crossing)
        if roi_config.active_tripwires:
            tripwire_events = self.spatial_engine.evaluate_tripwires(
                cam_id, tracks, roi_config.active_tripwires, processed_frame.scale_x, processed_frame.scale_y, now
            )
            for sp_ev in tripwire_events:
                track = current_track_map.get(sp_ev.track_id)
                snapshot_path = None
                if track:
                    snapshot_path = self._generate_and_save_snapshot(
                        processed_frame=processed_frame,
                        track=track,
                        zone=sp_ev.rule,
                        event_type=EVENT_LINE_CROSSING,
                        custom_note=sp_ev.notes,
                    )
                ev = LifecycleEvent(
                    camera_id=cam_id,
                    zone_id=sp_ev.rule_id,
                    event_type=EVENT_LINE_CROSSING,
                    timestamp=iso_ts,
                    track_id=sp_ev.track_id,
                    class_label=sp_ev.class_label,
                    confidence=sp_ev.confidence,
                    snapshot_path=snapshot_path,
                    is_resolved=1,
                    resolved_time=iso_ts,
                    notes=sp_ev.notes,
                )
                ev.insert(db_conn)
                generated_events.append(ev)

        # B. Polyline Barriers (Virtual Perimeter Fence)
        if roi_config.active_barriers:
            barrier_events = self.spatial_engine.evaluate_barriers(
                cam_id, tracks, roi_config.active_barriers, processed_frame.scale_x, processed_frame.scale_y, now
            )
            for sp_ev in barrier_events:
                track = current_track_map.get(sp_ev.track_id)
                snapshot_path = None
                if track:
                    snapshot_path = self._generate_and_save_snapshot(
                        processed_frame=processed_frame,
                        track=track,
                        zone=sp_ev.rule,
                        event_type=EVENT_LINE_CROSSING,
                        custom_note=sp_ev.notes,
                    )
                ev = LifecycleEvent(
                    camera_id=cam_id,
                    zone_id=sp_ev.rule_id,
                    event_type=EVENT_LINE_CROSSING,
                    timestamp=iso_ts,
                    track_id=sp_ev.track_id,
                    class_label=sp_ev.class_label,
                    confidence=sp_ev.confidence,
                    snapshot_path=snapshot_path,
                    is_resolved=1,
                    resolved_time=iso_ts,
                    notes=sp_ev.notes,
                )
                ev.insert(db_conn)
                generated_events.append(ev)

        # C. Safe Walkways (K3 Pedestrian Corridor Compliance)
        if roi_config.active_safe_walkways:
            walkway_events = self.spatial_engine.evaluate_walkways(
                cam_id, tracks, roi_config.active_safe_walkways, processed_frame.scale_x, processed_frame.scale_y, now
            )
            for sp_ev in walkway_events:
                track = current_track_map.get(sp_ev.track_id)
                snapshot_path = None
                if track:
                    snapshot_path = self._generate_and_save_snapshot(
                        processed_frame=processed_frame,
                        track=track,
                        zone=sp_ev.rule,
                        event_type=EVENT_WALKWAY_VIOLATION,
                        custom_note=sp_ev.notes,
                    )
                ev = LifecycleEvent(
                    camera_id=cam_id,
                    zone_id=sp_ev.rule_id,
                    event_type=EVENT_WALKWAY_VIOLATION,
                    timestamp=iso_ts,
                    track_id=sp_ev.track_id,
                    class_label=sp_ev.class_label,
                    confidence=sp_ev.confidence,
                    snapshot_path=snapshot_path,
                    is_resolved=1,
                    resolved_time=iso_ts,
                    notes=sp_ev.notes,
                )
                ev.insert(db_conn)
                generated_events.append(ev)

        # D. Density & Crowd Monitors (Overcrowding / Congestion Alert)
        if roi_config.active_density_rules:
            density_events = self.spatial_engine.evaluate_density(
                cam_id,
                tracks,
                roi_config.active_density_rules,
                roi_config.polygons,
                processed_frame.scale_x,
                processed_frame.scale_y,
                now,
            )
            for sp_ev in density_events:
                track = current_track_map.get(sp_ev.track_id) if sp_ev.track_id else None
                snapshot_path = self._generate_and_save_snapshot(
                    processed_frame=processed_frame,
                    track=track,
                    zone=sp_ev.rule,
                    event_type=EVENT_CONGESTION_ALERT,
                    custom_note=sp_ev.notes,
                )
                ev = LifecycleEvent(
                    camera_id=cam_id,
                    zone_id=sp_ev.rule_id,
                    event_type=EVENT_CONGESTION_ALERT,
                    timestamp=iso_ts,
                    track_id=sp_ev.track_id if sp_ev.track_id else 0,
                    class_label=sp_ev.class_label,
                    confidence=sp_ev.confidence,
                    snapshot_path=snapshot_path,
                    is_resolved=1,
                    resolved_time=iso_ts,
                    notes=sp_ev.notes,
                )
                ev.insert(db_conn)
                generated_events.append(ev)

        # ── 3. CLEANUP TRACK MEMORY DI SPATIAL RULE ENGINE ────────────────────
        self.spatial_engine.cleanup_tracks(set(current_track_map.keys()))

        # ── 4. CLEANUP ORPHANED OPEN EVENTS (Track hilang saat di dalam zone) ──
        keys_to_remove = []
        for key, state in self.open_events.items():
            if state.track_id not in current_track_map:
                if (now - state.last_seen_time) >= self.exit_timeout_sec:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            state = self.open_events.pop(key)
            dwell_duration_sec = now - state.enter_time

            db_conn.execute(
                "UPDATE lifecycle_events SET is_resolved = 1, resolved_time = :resolved_time, "
                "notes = COALESCE(:notes, notes) "
                "WHERE camera_id = :camera_id AND zone_id = :zone_id AND track_id = :track_id AND is_resolved = 0",
                {
                    "resolved_time": iso_ts,
                    "notes": f"Dwell duration: {dwell_duration_sec:.1f}s (Track timeout)",
                    "camera_id": state.camera_id,
                    "zone_id": state.zone_id,
                    "track_id": state.track_id,
                },
            )

            exit_event = LifecycleEvent(
                camera_id=state.camera_id,
                zone_id=state.zone_id,
                event_type="exit",
                timestamp=iso_ts,
                track_id=state.track_id,
                class_label=state.class_label,
                confidence=state.confidence,
                snapshot_path=None,
                is_resolved=1,
                resolved_time=iso_ts,
                notes=f"Dwell duration: {dwell_duration_sec:.1f}s (Track timeout)",
            )
            exit_event.insert(db_conn)
            generated_events.append(exit_event)

        return generated_events

    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Konversi '#RRGGBB' ke (B, G, R)."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r)
        return (0, 255, 0)

    def _generate_and_save_snapshot(
        self,
        processed_frame: ProcessedFrame,
        track: Optional[TrackResult],
        zone: Optional[Any],
        event_type: str,
        dwell_sec: float = 0.0,
        custom_note: Optional[str] = None,
    ) -> str:
        """
        Anotasi raw frame (1080p) dan simpan sebagai JPEG kualitas 85.
        Mendukung ROIZone (poligon), TripwireRule (garis 2 titik), BarrierRule (polyline),
        SafeWalkwayRule (koridor), dan DensityRule.
        """
        annotated = processed_frame.raw_frame.copy()

        # 1. Gambar Geometri ROI / Aturan (koordinat raw 1080p)
        color_bgr = (0, 255, 0)
        label_text = "ROI"

        if zone is not None:
            color_hex = getattr(zone, "color_hex", "#00FF00")
            color_bgr = self._hex_to_bgr(color_hex)
            label_text = getattr(zone, "label", getattr(zone, "zone_id", getattr(zone, "rule_id", "Zone")))

            # A. Jika Tripwire (Line)
            if hasattr(zone, "p1") and hasattr(zone, "p2"):
                pt1 = (int(zone.p1.x), int(zone.p1.y))
                pt2 = (int(zone.p2.x), int(zone.p2.y))
                cv2.line(annotated, pt1, pt2, color_bgr, thickness=3)
                # Panah arah jika unidirectional
                direction = getattr(zone, "direction", "BIDIRECTIONAL")
                if direction in ("A_TO_B", "B_TO_A"):
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2
                    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                    norm = np.hypot(dx, dy) or 1.0
                    nx, ny = -dy / norm, dx / norm
                    if direction == "B_TO_A":
                        nx, ny = -nx, -ny
                    tip_x = int(mid_x + nx * 30)
                    tip_y = int(mid_y + ny * 30)
                    cv2.arrowedLine(annotated, (mid_x, mid_y), (tip_x, tip_y), color_bgr, 3, tipLength=0.4)

            # B. Jika Polyline Barrier
            elif hasattr(zone, "points") and getattr(zone, "rule_id", "").startswith("bar_"):
                pts = np.array([[p.x, p.y] for p in zone.points], dtype=np.int32)
                cv2.polylines(annotated, [pts], isClosed=False, color=color_bgr, thickness=3)

            # C. Jika Poligon tertutup (ROIZone, SafeWalkway, Density, Exclusion)
            elif hasattr(zone, "points"):
                pts = np.array([[p.x, p.y] for p in zone.points], dtype=np.int32)
                cv2.polylines(annotated, [pts], isClosed=True, color=color_bgr, thickness=3)

        # 2. Gambar Bounding Box (jika ada track)
        track_id_str = "N/A"
        class_str = "Object"
        rx1, ry1 = 30, 60

        if track is not None:
            track_id_str = str(track.track_id)
            class_str = track.class_label
            rx1, ry1, rx2, ry2 = processed_frame.map_bbox_to_raw(*track.bbox)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 0), thickness=3)

        # 3. Text Badge dengan background solid
        if custom_note:
            text = f"ID:{track_id_str} | {class_str} | {label_text} ({event_type.upper()}) | {custom_note}"
        else:
            text = f"ID:{track_id_str} | {class_str} | {dwell_sec:.0f}s | {label_text} ({event_type.upper()})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        bg_y1 = max(0, ry1 - text_h - 10)
        bg_y2 = max(bg_y1 + text_h + 10, ry1)
        cv2.rectangle(annotated, (rx1, bg_y1), (rx1 + text_w + 10, bg_y2), (0, 0, 0), cv2.FILLED)
        cv2.putText(annotated, text, (rx1 + 5, bg_y2 - 5), font, font_scale, (255, 255, 255), thickness)

        # 4. Tentukan Path File
        dt_str = datetime.fromtimestamp(processed_frame.timestamp, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        dir_path = Path(f"cameras/{processed_frame.camera_id}/snapshots")
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"{event_type}_{track_id_str}_{dt_str}.jpg"
        file_path = dir_path / filename

        # 5. Simpan JPEG Kualitas 85
        cv2.imwrite(str(file_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, self.snapshot_quality])
        return str(file_path)
