"""
Visual ROI Calibrator Tool untuk Facility Management.
Menyediakan GUI interaktif (OpenCV) pada resolusi native 1080p untuk menggambar dan mengkalibrasi
zona ROI poligon dengan auto-backup (.bak) dan validasi skema Pydantic.
Desain antarmuka modern: Floating HUD Card, Live Rubber-Band Line, Crosshair Guide, dan Transparent Fill.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Tambahkan root workspace ke python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.config_loader import (
    CameraConfig,
    ROIPoint,
    ROIZone,
    ROIZonesConfig,
    load_camera_config,
    load_roi_zones,
)
from engine.logger import get_logger

logger = get_logger("ROICalibrator")

# Palet warna zona dinamis (format BGR OpenCV) - Terang, jernih & kontras tinggi
PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 120),   # Zona 1: Hijau Zamrud Terang (Bright Emerald Green)
    (0, 215, 255),   # Zona 2: Kuning / Amber Terang
    (255, 255, 0),   # Zona 3: Cyan Terang
    (255, 0, 255),   # Zona 4: Magenta Terang
    (0, 165, 255),   # Zona 5: Oranye Terang
    (255, 190, 50),  # Zona 6: Sky Blue Terang
]


class ROICalibrator:
    """Aplikasi GUI Kalibrator Zona ROI & Aturan Spasial modern berbasis OpenCV HighGUI."""

    def __init__(
        self,
        camera_id: str = "cam_01",
        source_override: Optional[str] = None,
        skip_frame_load: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.source_override = source_override
        self.cam_dir = PROJECT_ROOT / "cameras" / camera_id
        self.roi_path = self.cam_dir / "roi_zones.json"
        self.config_path = self.cam_dir / "config.json"

        if not self.cam_dir.exists():
            raise FileNotFoundError(f"Direktori kamera tidak ditemukan: {self.cam_dir}")

        self.roi_config: ROIZonesConfig = load_roi_zones(self.roi_path)

        # Mode geometri aktif: 'polygon', 'tripwire', 'barrier', 'exclusion'
        self.geom_mode: str = "polygon"

        # Koleksi master seluruh entitas kalibrasi dalam 1 sesi terpadu
        self.entities: List[dict] = []
        for z in self.roi_config.polygons:
            d = z.model_dump()
            d["type"] = "polygon"
            self.entities.append(d)
        for t in self.roi_config.tripwires:
            d = t.model_dump()
            d["type"] = "tripwire"
            self.entities.append(d)
        for b in self.roi_config.barriers:
            d = b.model_dump()
            d["type"] = "barrier"
            self.entities.append(d)
        for e in self.roi_config.exclusion_masks:
            d = e.model_dump()
            d["type"] = "exclusion"
            self.entities.append(d)

        # Inisialisasi draft awal jika seluruh entitas kosong
        if not self.entities:
            self.entities.append(self._create_empty_element("polygon", 0))

        self.active_idx: int = 0
        if self.entities:
            self.geom_mode = self.entities[0].get("type", "polygon")

        self.mouse_pos: Optional[Tuple[int, int]] = None
        self.status_message: str = "Siap"
        self.status_color: Tuple[int, int, int] = (220, 220, 220)

        # Muat base canvas 1080p
        self.base_frame = (
            np.zeros((1080, 1920, 3), dtype=np.uint8)
            if skip_frame_load
            else self._load_base_frame()
        )

    @property
    def active_entity(self) -> dict:
        """Ambil dict entitas yang sedang aktif diedit saat ini."""
        if not self.entities:
            self.entities.append(self._create_empty_element(self.geom_mode, 0))
            self.active_idx = 0
        self.active_idx = max(0, min(self.active_idx, len(self.entities) - 1))
        return self.entities[self.active_idx]

    @property
    def current_collection(self) -> List[dict]:
        """Ambil list elemen yang sesuai dengan mode geometri yang sedang aktif."""
        return [e for e in self.entities if e.get("type") == self.geom_mode]

    @property
    def polygons(self) -> List[dict]:
        return [e for e in self.entities if e.get("type") == "polygon"]

    @polygons.setter
    def polygons(self, value: List[dict]) -> None:
        self.entities = [e for e in self.entities if e.get("type") != "polygon"]
        for item in value:
            d = dict(item)
            d["type"] = "polygon"
            self.entities.append(d)

    @property
    def tripwires(self) -> List[dict]:
        return [e for e in self.entities if e.get("type") == "tripwire"]

    @tripwires.setter
    def tripwires(self, value: List[dict]) -> None:
        self.entities = [e for e in self.entities if e.get("type") != "tripwire"]
        for item in value:
            d = dict(item)
            d["type"] = "tripwire"
            self.entities.append(d)

    @property
    def barriers(self) -> List[dict]:
        return [e for e in self.entities if e.get("type") == "barrier"]

    @barriers.setter
    def barriers(self, value: List[dict]) -> None:
        self.entities = [e for e in self.entities if e.get("type") != "barrier"]
        for item in value:
            d = dict(item)
            d["type"] = "barrier"
            self.entities.append(d)

    @property
    def exclusion_masks(self) -> List[dict]:
        return [e for e in self.entities if e.get("type") == "exclusion"]

    @exclusion_masks.setter
    def exclusion_masks(self, value: List[dict]) -> None:
        self.entities = [e for e in self.entities if e.get("type") != "exclusion"]
        for item in value:
            d = dict(item)
            d["type"] = "exclusion"
            self.entities.append(d)

    @property
    def zones(self) -> List[dict]:
        """Alias kompatibilitas ke polygons."""
        return self.polygons

    @zones.setter
    def zones(self, value: List[dict]) -> None:
        self.polygons = value

    def _entity_has_points(self, elem: dict) -> bool:
        """Cek apakah suatu entitas sudah memiliki setidaknya 1 titik."""
        if not elem:
            return False
        etype = elem.get("type", "polygon")
        if etype == "tripwire":
            return elem.get("p1") is not None or elem.get("p2") is not None
        return len(elem.get("points", [])) > 0

    def _is_entity_complete(self, elem: dict) -> bool:
        """Cek apakah entitas telah memenuhi kuota titik minimal sesuai tipenya."""
        if not elem:
            return False
        etype = elem.get("type", "polygon")
        if etype == "tripwire":
            return elem.get("p1") is not None and elem.get("p2") is not None
        elif etype == "polygon":
            return len(elem.get("points", [])) >= 3
        elif etype == "barrier":
            return len(elem.get("points", [])) >= 2
        elif etype == "exclusion":
            return len(elem.get("points", [])) >= 3
        return False

    def _reconfigure_entity(self, elem: dict, target_mode: str, index: int) -> None:
        """Ubah struktur elemen kosong menjadi tipe geometri target."""
        new_data = self._create_empty_element(target_mode, index)
        elem.clear()
        elem.update(new_data)

    def _switch_mode(self, target_mode: str) -> None:
        """Transisi mode geometri ([P], [T], [B], [E]) secara konsisten."""
        target_mode = target_mode.lower()
        self.geom_mode = target_mode
        if not self.entities:
            self.entities.append(self._create_empty_element(target_mode, 0))
            self.active_idx = 0
            self.status_message = f"Mode: {target_mode.upper()} (Siap gambar)"
            self.status_color = (100, 220, 80)
            return

        self.active_idx = max(0, min(self.active_idx, len(self.entities) - 1))
        curr = self.entities[self.active_idx]

        # 1. Jika entitas aktif saat ini belum memiliki titik: rekonfigurasi tipenya
        if not self._entity_has_points(curr):
            self._reconfigure_entity(curr, target_mode, self.active_idx)
            if target_mode == "tripwire":
                self.status_message = "Mode: TRIPWIRE (Kuota: 2 titik. Klik P1)"
            elif target_mode == "polygon":
                self.status_message = "Mode: POLYGON (Minimal 3 titik)"
            elif target_mode == "barrier":
                self.status_message = "Mode: BARRIER (Minimal 2 titik)"
            elif target_mode == "exclusion":
                self.status_message = "Mode: EXCLUSION (Minimal 3 titik)"
            self.status_color = (100, 220, 80)
            return

        # 2. Jika entitas aktif saat ini sudah bertipe target_mode
        if curr.get("type") == target_mode:
            self.status_message = f"Mode: {target_mode.upper()} aktif pada #{self.active_idx + 1}"
            self.status_color = (100, 220, 80)
            return

        # 3. Entitas aktif bertipe lain dan sudah ada titik:
        # Cek apakah ada entitas target_mode yang sudah ada sebelumnya
        existing_idx = next((i for i, e in enumerate(self.entities) if e.get("type") == target_mode), None)
        if existing_idx is not None:
            self.active_idx = existing_idx
            target_elem = self.entities[self.active_idx]
            self.geom_mode = target_mode
            self._update_nav_status(target_elem)
            return

        # 4. Belum ada entitas bertipe target_mode:
        if self._is_entity_complete(curr):
            new_idx = len(self.entities)
            new_elem = self._create_empty_element(target_mode, new_idx)
            self.entities.append(new_elem)
            self.active_idx = new_idx
            if target_mode == "tripwire":
                self.status_message = f"Tripwire Baru #{new_idx + 1}. Titik: 0/2 (Klik P1)"
            elif target_mode == "polygon":
                self.status_message = f"Polygon Baru #{new_idx + 1}. Klik minimal 3 titik"
            elif target_mode == "barrier":
                self.status_message = f"Barrier Baru #{new_idx + 1}. Klik minimal 2 titik"
            elif target_mode == "exclusion":
                self.status_message = f"Exclusion Baru #{new_idx + 1}. Klik minimal 3 titik"
            self.status_color = (100, 220, 80)
        else:
            self.status_message = f"Entitas #{self.active_idx + 1} ({curr.get('type')}) belum selesai. Tekan [N] untuk baru atau [C] reset."
            self.status_color = (0, 165, 255)

    def _undo_active_entity(self) -> None:
        """Universal Undo: hapus titik terakhir sesuai jenis geometri aktif."""
        curr = self.active_entity
        curr_type = curr.get("type", self.geom_mode)

        if curr_type == "tripwire":
            if curr.get("p2") is not None:
                curr["p2"] = None
                self.status_message = "Undo P2: Titik: 1/2 (Klik titik akhir P2)"
                self.status_color = (0, 215, 255)
            elif curr.get("p1") is not None:
                curr["p1"] = None
                self.status_message = "Undo P1: Titik: 0/2 (Klik titik awal P1)"
                self.status_color = (0, 215, 255)
            else:
                self.status_message = "Tripwire kosong, tidak ada titik untuk di-undo"
                self.status_color = (180, 180, 180)
        else:
            pts = curr.get("points", [])
            if pts:
                popped = pts.pop()
                self.status_message = f"Undo ({popped['x']}, {popped['y']}), sisa {len(pts)} titik"
                self.status_color = (0, 215, 255)
            else:
                self.status_message = f"{curr_type.upper()} kosong, tidak ada titik untuk di-undo"
                self.status_color = (180, 180, 180)

    def _clear_active_entity(self) -> None:
        """Universal Clear / Redraw: reset seluruh titik tanpa menghapus metadata/id."""
        curr = self.active_entity
        curr_type = curr.get("type", self.geom_mode)
        lbl = curr.get("label", curr.get("zone_id", f"#{self.active_idx + 1}"))

        if curr_type == "tripwire":
            curr["p1"] = None
            curr["p2"] = None
            self.status_message = f"{lbl} di-reset. Siap gambar ulang: [Klik Kiri] Tentukan P1"
        else:
            curr.setdefault("points", []).clear()
            min_req = 3 if curr_type in ("polygon", "exclusion") else 2
            self.status_message = f"{lbl} di-reset. Siap gambar ulang: [Klik Kiri] Titik 1 (min {min_req})"
        self.status_color = (0, 215, 255)

    def _select_entity(self, index: int) -> None:
        """Pilih entitas secara langsung berdasarkan indeks 0-based."""
        if 0 <= index < len(self.entities):
            self.active_idx = index
            curr = self.entities[self.active_idx]
            self.geom_mode = curr.get("type", "polygon")
            self._update_nav_status(curr)
        else:
            self.status_message = f"Zona #{index + 1} belum ada (Total: {len(self.entities)})"
            self.status_color = (180, 180, 180)

    def _cycle_active_entity(self, direction: int = 1) -> None:
        """Pindah ke entitas berikutnya (+1) atau sebelumnya (-1) secara siklis."""
        if not self.entities:
            return
        if len(self.entities) == 1:
            curr = self.entities[0]
            self.geom_mode = curr.get("type", "polygon")
            self._update_nav_status(curr)
            return
        self.active_idx = (self.active_idx + direction) % len(self.entities)
        curr = self.entities[self.active_idx]
        self.geom_mode = curr.get("type", "polygon")
        self._update_nav_status(curr)

    def _toggle_direction(self) -> None:
        """Ganti arah vektor Tripwire (BOTH -> A_TO_B -> B_TO_A)."""
        curr = self.active_entity
        if curr.get("type") == "tripwire":
            seq = ["BOTH", "A_TO_B", "B_TO_A"]
            curr_dir = curr.get("direction", "BOTH")
            next_dir = seq[(seq.index(curr_dir) + 1) % len(seq)] if curr_dir in seq else "BOTH"
            curr["direction"] = next_dir
            self.status_message = f"Arah Tripwire: {next_dir}"
            self.status_color = (100, 220, 80)
        else:
            self.status_message = "Tombol [D] hanya untuk Tripwire"
            self.status_color = (180, 180, 180)

    def _create_new_entity(self) -> None:
        """Finalisasi entitas saat ini dan buat entitas baru kosong."""
        curr = self.active_entity
        if not self._entity_has_points(curr):
            self.status_message = f"Entitas #{self.active_idx + 1} ({curr.get('type')}) masih kosong, siap digambar"
            self.status_color = (220, 220, 100)
        else:
            new_idx = len(self.entities)
            new_elem = self._create_empty_element(self.geom_mode, new_idx)
            self.entities.append(new_elem)
            self.active_idx = new_idx
            if self.geom_mode == "tripwire":
                self.status_message = f"Tripwire Baru #{new_idx + 1}. Titik: 0/2 (Klik P1)"
            elif self.geom_mode == "polygon":
                self.status_message = f"Polygon Baru #{new_idx + 1}. Klik minimal 3 titik"
            elif self.geom_mode == "barrier":
                self.status_message = f"Barrier Baru #{new_idx + 1}. Klik minimal 2 titik"
            elif self.geom_mode == "exclusion":
                self.status_message = f"Exclusion Baru #{new_idx + 1}. Klik minimal 3 titik"
            self.status_color = (100, 220, 80)

    def _update_nav_status(self, curr: dict) -> None:
        """Perbarui status pesan saat berpindah ke entitas tertentu."""
        curr_type = curr.get("type", "polygon")
        lbl = curr.get("label", curr.get("zone_id", f"#{self.active_idx + 1}"))
        if curr_type == "tripwire":
            has_p1 = curr.get("p1") is not None
            has_p2 = curr.get("p2") is not None
            if has_p1 and has_p2:
                self.status_message = f"Pilih #{self.active_idx + 1}: {lbl} (Terkunci) | [U/R-Click] Edit P2 | [C] Gambar Ulang"
            elif has_p1:
                self.status_message = f"Pilih #{self.active_idx + 1}: {lbl} (1/2) | [Klik] Set P2 | [U] Undo P1"
            else:
                self.status_message = f"Pilih #{self.active_idx + 1}: {lbl} (Kosong) | [Klik] Set P1"
        else:
            pts_cnt = len(curr.get("points", []))
            min_req = 3 if curr_type in ("polygon", "exclusion") else 2
            if pts_cnt >= min_req:
                self.status_message = f"Pilih #{self.active_idx + 1}: {lbl} ({pts_cnt}pts) | [U] Undo | [C] Gambar Ulang"
            elif pts_cnt > 0:
                self.status_message = f"Pilih #{self.active_idx + 1}: {lbl} ({pts_cnt}/{min_req}pts) | [Klik] Tambah titik"
            else:
                self.status_message = f"Pilih #{self.active_idx + 1}: {lbl} (Kosong) | [Klik] Gambar titik awal"
        self.status_color = (220, 220, 220)

    def _create_empty_element(self, mode: str, index: int) -> dict:
        """Buat struktur data draft elemen spasial baru sesuai mode."""
        mode = mode.lower()
        num = index + 1
        color_bgr = PALETTE[index % len(PALETTE)]
        color_hex = f"#{color_bgr[2]:02X}{color_bgr[1]:02X}{color_bgr[0]:02X}"

        if mode == "polygon":
            return {
                "zone_id": f"zone_{num:02d}",
                "label": f"Zone {num}",
                "type": "polygon",
                "active": True,
                "color_hex": color_hex,
                "points": [],
                "trigger_on": ["linger"],
                "linger_threshold_sec": 3,
                "target_classes": None,
            }
        elif mode == "tripwire":
            return {
                "tripwire_id": f"tw_{num:02d}",
                "label": f"Tripwire {num}",
                "type": "tripwire",
                "active": True,
                "color_hex": color_hex,
                "p1": None,
                "p2": None,
                "direction": "BOTH",
                "target_classes": ["person", "car", "motorcycle", "bus", "truck"],
                "debounce_sec": 3.0,
            }
        elif mode == "barrier":
            return {
                "barrier_id": f"bar_{num:02d}",
                "label": f"Barrier {num}",
                "type": "barrier",
                "active": True,
                "color_hex": color_hex,
                "points": [],
                "direction": "BOTH",
                "target_classes": ["person", "car", "motorcycle", "bus", "truck"],
                "debounce_sec": 3.0,
            }
        elif mode == "exclusion":
            return {
                "mask_id": f"ex_{num:02d}",
                "label": f"Exclusion {num}",
                "type": "exclusion",
                "active": True,
                "color_hex": "#808080",
                "points": [],
                "ignore_types": ["dwell", "motion"],
            }
        return {}

    def _create_empty_zone(self, index: int) -> dict:
        """Alias kompatibilitas."""
        return self._create_empty_element("polygon", index)

    def _load_base_frame(self) -> np.ndarray:
        """Ambil frame 1080p dari kamera fisik/stream atau buat kanvas grid sintetis."""
        source = self.source_override
        if not source and self.config_path.exists():
            try:
                cfg = load_camera_config(self.config_path)
                source = cfg.rtsp_url
            except Exception as e:
                logger.warning(f"Gagal membaca config kamera: {e}")

        if source:
            logger.info(f"Mencoba mengambil snapshot frame dari RTSP: {self.camera_id}")
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    if frame.shape[:2] != (1080, 1920):
                        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                    logger.info("Snapshot live RTSP 1080p berhasil dimuat ✓")
                    return frame

        logger.info("Menggunakan kanvas grid sintetis 1080p (RTSP offline atau tidak tersedia).")
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas[:] = (20, 24, 32)  # Dark slate blue background

        # Gambar grid garis referensi tiap 100px
        for x in range(0, 1920, 100):
            cv2.line(canvas, (x, 0), (x, 1080), (35, 42, 54), 1)
            cv2.putText(canvas, str(x), (x + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 80, 100), 1, cv2.LINE_AA)

        for y in range(0, 1080, 100):
            cv2.line(canvas, (0, y), (1920, y), (35, 42, 54), 1)
            cv2.putText(canvas, str(y), (6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 80, 100), 1, cv2.LINE_AA)

        return canvas

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Mouse callback: track kursor, klik kiri tambah titik, klik kanan undo."""
        x = max(0, min(1919, int(x)))
        y = max(0, min(1079, int(y)))

        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
            return

        if not self.entities:
            self.entities.append(self._create_empty_element(self.geom_mode, 0))
            self.active_idx = 0

        self.active_idx = max(0, min(self.active_idx, len(self.entities) - 1))
        curr = self.entities[self.active_idx]
        curr_type = curr.get("type", self.geom_mode)

        if event == cv2.EVENT_LBUTTONDOWN:
            if curr_type == "tripwire":
                if curr.get("p1") is None:
                    curr["p1"] = {"x": x, "y": y}
                    self.status_message = "Titik: 1/2 (Klik titik akhir P2)"
                    self.status_color = (100, 220, 80)
                elif curr.get("p2") is None:
                    curr["p2"] = {"x": x, "y": y}
                    self.status_message = "Tripwire Selesai. Tekan [N] entitas baru, [U] edit P2, atau [D] ganti arah"
                    self.status_color = (100, 220, 80)
                else:
                    # Klik ke-3: JANGAN UBAH KOORDINAT! Tampilkan peringatan visual di HUD
                    self.status_message = "Tripwire sudah terkunci (2/2)! Tekan [U/Klik Kanan] edit P2 atau [C] gambar ulang"
                    self.status_color = (0, 165, 255)
            else:
                # Mode polygon, barrier, exclusion
                pts = curr.setdefault("points", [])
                pts.append({"x": x, "y": y})
                total = len(pts)
                min_req = 3 if curr_type in ("polygon", "exclusion") else 2
                if total >= min_req:
                    self.status_message = f"Titik #{total} ({x}, {y}) [Cukup: min {min_req}]. Tekan [N] baru, [U] undo"
                else:
                    self.status_message = f"Titik #{total} ({x}, {y}) [Butuh min {min_req}]"
                self.status_color = (100, 220, 80)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._undo_active_entity()

    def save_zones(self) -> bool:
        """Simpan zona dan aturan spasial ke berkas dengan auto-backup .bak dan validasi Pydantic."""
        try:
            # Filter dan kelompokkan elemen yang valid sesuai skema Pydantic v2
            valid_polys = []
            valid_tripwires = []
            valid_barriers = []
            valid_exclusions = []

            for elem in self.entities:
                etype = elem.get("type", "polygon")
                if etype == "polygon" and len(elem.get("points", [])) >= 3:
                    p_data = {
                        "zone_id": elem.get("zone_id", f"zone_{len(valid_polys)+1:02d}"),
                        "label": elem.get("label", f"Zone {len(valid_polys)+1}"),
                        "type": "polygon",
                        "active": elem.get("active", True),
                        "color_hex": elem.get("color_hex", "#00FF00"),
                        "points": elem.get("points", []),
                        "trigger_on": elem.get("trigger_on", ["linger"]),
                        "linger_threshold_sec": elem.get("linger_threshold_sec", 3),
                        "target_classes": elem.get("target_classes", None),
                    }
                    valid_polys.append(p_data)
                elif etype == "tripwire" and elem.get("p1") and elem.get("p2"):
                    tw_data = {
                        "tripwire_id": elem.get("tripwire_id", f"tw_{len(valid_tripwires)+1:02d}"),
                        "label": elem.get("label", f"Tripwire {len(valid_tripwires)+1}"),
                        "active": elem.get("active", True),
                        "color_hex": elem.get("color_hex", "#FF5500"),
                        "p1": elem.get("p1"),
                        "p2": elem.get("p2"),
                        "direction": elem.get("direction", "A_TO_B") if elem.get("direction") in ("A_TO_B", "B_TO_A", "BOTH") else "A_TO_B",
                        "target_classes": elem.get("target_classes", ["person", "car", "motorcycle", "bus", "truck"]),
                        "debounce_sec": float(elem.get("debounce_sec", 3.0)),
                    }
                    valid_tripwires.append(tw_data)
                elif etype == "barrier" and len(elem.get("points", [])) >= 2:
                    b_data = {
                        "barrier_id": elem.get("barrier_id", f"bar_{len(valid_barriers)+1:02d}"),
                        "label": elem.get("label", f"Barrier {len(valid_barriers)+1}"),
                        "active": elem.get("active", True),
                        "color_hex": elem.get("color_hex", "#FF0055"),
                        "points": elem.get("points", []),
                        "direction": elem.get("direction", "BOTH") if elem.get("direction") in ("A_TO_B", "B_TO_A", "BOTH") else "BOTH",
                        "target_classes": elem.get("target_classes", ["person", "car", "motorcycle", "bus", "truck"]),
                        "debounce_sec": float(elem.get("debounce_sec", 3.0)),
                    }
                    valid_barriers.append(b_data)
                elif etype == "exclusion" and len(elem.get("points", [])) >= 3:
                    e_data = {
                        "mask_id": elem.get("mask_id", f"ex_{len(valid_exclusions)+1:02d}"),
                        "label": elem.get("label", f"Exclusion {len(valid_exclusions)+1}"),
                        "active": elem.get("active", True),
                        "color_hex": elem.get("color_hex", "#646464"),
                        "points": elem.get("points", []),
                        "ignore_types": elem.get("ignore_types", ["dwell", "motion"]),
                    }
                    valid_exclusions.append(e_data)

            # 1. Auto-Backup ke .bak
            bak_path = self.roi_path.with_suffix(".json.bak")
            if self.roi_path.exists():
                shutil.copy2(self.roi_path, bak_path)
                logger.info(f"Auto-backup tersimpan di: {bak_path.name}")

            # 2. Susun payload konfigurasi lengkap
            payload = {
                "schema_version": self.roi_config.schema_version,
                "camera_id": self.camera_id,
                "polygons": valid_polys,
                "tripwires": valid_tripwires,
                "barriers": valid_barriers,
                "exclusion_masks": valid_exclusions,
                "safe_walkways": [w.model_dump() for w in self.roi_config.safe_walkways],
                "density_rules": [d.model_dump() for d in self.roi_config.density_rules],
            }

            # 3. Validasi dengan Pydantic
            validated = ROIZonesConfig.model_validate(payload)

            # 4. Tulis berkas JSON terformat
            formatted_json = validated.model_dump_json(indent=2)
            self.roi_path.write_text(formatted_json, encoding="utf-8")

            total_saved = len(valid_polys) + len(valid_tripwires) + len(valid_barriers) + len(valid_exclusions)
            self.status_message = f"Tersimpan ({total_saved} Aturan: {len(valid_polys)}P, {len(valid_tripwires)}T, {len(valid_barriers)}B, {len(valid_exclusions)}E)"
            self.status_color = (100, 220, 80)
            logger.info(
                f"✓ Konfigurasi spasial berhasil disimpan ke {self.roi_path.name}: "
                f"{len(valid_polys)} poligon, {len(valid_tripwires)} tripwire, "
                f"{len(valid_barriers)} barrier, {len(valid_exclusions)} exclusion mask"
            )
            return True

        except Exception as e:
            self.status_message = f"GAGAL SIMPAN: {e}"
            self.status_color = (0, 0, 255)
            logger.error(f"Validasi/Penyimpanan zona gagal: {e}")
            return False

    def run_gui(self) -> None:
        """Jalankan antarmuka interaktif OpenCV GUI dengan kanvas visual modern."""
        window_name = f"Facility Management — ROI Calibrator ({self.camera_id})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self._on_mouse)

        logger.info("=" * 70)
        logger.info(f"  Facility Management — Visual ROI Calibrator: {self.camera_id}")
        logger.info("  [Mode Geometri] : [P] Poligon | [T] Tripwire | [B] Barrier | [E] Exclusion")
        logger.info("  [Klik Kiri]     : Tambah titik vertex / set tripwire p1 & p2")
        logger.info("  [Klik Kanan]    : Undo titik terakhir")
        logger.info("  [1-9 / TAB / []]: Pindah / pilih indeks aktif")
        logger.info("  [N]             : Finalisasi & buat entitas baru")
        logger.info("  [D]             : Ganti arah Tripwire (A_TO_B <-> B_TO_A <-> BOTH)")
        logger.info("  [C]             : Reset titik elemen aktif")
        logger.info("  [S]             : Simpan konfigurasi ke JSON (Auto-backup .bak)")
        logger.info("  [Q / ESC]       : Keluar")
        logger.info("=" * 70)

        while True:
            display = self.base_frame.copy()

            if not self.entities:
                self.entities.append(self._create_empty_element(self.geom_mode, 0))
                self.active_idx = 0

            self.active_idx = max(0, min(self.active_idx, len(self.entities) - 1))
            curr_elem = self.entities[self.active_idx]
            curr_type = curr_elem.get("type", self.geom_mode)
            self.geom_mode = curr_type

            active_color = PALETTE[self.active_idx % len(PALETTE)] if curr_type != "exclusion" else (160, 160, 160)

            # -------------------------------------------------------------
            # 1. DUAL-STROKE CONTRAST GUARD RENDERING (Solid & High-Contrast)
            # Pass Idle   (alpha=0.65): entitas non-aktif terlihat jelas di atas aspal
            # Pass Active (alpha=1.00): entitas aktif solid, tegas, tanpa transparansi
            # Dual-Stroke : Underlay gelap 2px (20, 20, 20) + Foreground warna 1px
            # -------------------------------------------------------------
            idle_overlay = display.copy()

            # Render seluruh entitas non-aktif pada idle_overlay
            for idx, elem in enumerate(self.entities):
                if idx == self.active_idx:
                    continue
                etype = elem.get("type", "polygon")
                raw_color = PALETTE[idx % len(PALETTE)] if etype != "exclusion" else (160, 160, 160)
                self._draw_entity_geometry(idle_overlay, elem, idx, raw_color, is_active=False)

            # Terapkan blending untuk entitas non-aktif dengan alpha = 0.65
            cv2.addWeighted(idle_overlay, 0.65, display, 0.35, 0, display)

            # Render entitas aktif LANGSUNG pada display dengan 100% OPASITAS SOLID (Alpha = 1.0)
            active_elem = self.entities[self.active_idx]
            self._draw_entity_geometry(display, active_elem, self.active_idx, active_color, is_active=True)

            # -------------------------------------------------------------
            # 2. LIVE RUBBER-BAND LINE & CROSSHAIR (Solid & High-Contrast)
            # -------------------------------------------------------------
            if self.mouse_pos is not None:
                mx, my = self.mouse_pos
                rb_color = (0, 235, 255)  # Vivid Amber/Cyan solid 1px

                if curr_type == "tripwire":
                    p1 = curr_elem.get("p1")
                    p2 = curr_elem.get("p2")
                    if p1 and not p2:
                        cv2.line(display, (p1["x"], p1["y"]), (mx, my), (20, 20, 20), 2, cv2.LINE_AA)
                        cv2.line(display, (p1["x"], p1["y"]), (mx, my), rb_color, 1, cv2.LINE_AA)
                else:
                    cur_pts = curr_elem.get("points", [])
                    if len(cur_pts) >= 1:
                        last_pt = (cur_pts[-1]["x"], cur_pts[-1]["y"])
                        cv2.line(display, last_pt, (mx, my), (20, 20, 20), 2, cv2.LINE_AA)
                        cv2.line(display, last_pt, (mx, my), rb_color, 1, cv2.LINE_AA)
                        if curr_type in ("polygon", "exclusion") and len(cur_pts) >= 2:
                            start_pt = (cur_pts[0]["x"], cur_pts[0]["y"])
                            cv2.line(display, (mx, my), start_pt, (20, 20, 20), 2, cv2.LINE_AA)
                            cv2.line(display, (mx, my), start_pt, (100, 180, 210), 1, cv2.LINE_AA)

                # Compact Crosshair (radius 3px dengan titik tengah solid)
                cv2.circle(display, (mx, my), 3, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(display, (mx, my), 2, active_color, -1, cv2.LINE_AA)
                cv2.circle(display, (mx, my), 6, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.circle(display, (mx, my), 6, (255, 255, 255), 1, cv2.LINE_AA)

                for (ax1, ay1), (ax2, ay2) in [
                    ((mx - 12, my), (mx - 7, my)),
                    ((mx + 7, my), (mx + 12, my)),
                    ((mx, my - 12), (mx, my - 7)),
                    ((mx, my + 7), (mx, my + 12)),
                ]:
                    cv2.line(display, (ax1, ay1), (ax2, ay2), (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.line(display, (ax1, ay1), (ax2, ay2), (230, 235, 240), 1, cv2.LINE_AA)

                coord_text = f"{mx}, {my}"
                (cw, ch), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cx1 = min(mx + 12, 1919 - cw - 8)
                cy1 = max(my - ch - 4, 4)
                cx2 = cx1 + cw + 8
                cy2 = cy1 + ch + 6
                cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (16, 20, 28), -1)
                cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (60, 70, 85), 1, cv2.LINE_AA)
                cv2.putText(display, coord_text, (cx1 + 4, cy2 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 230, 240), 1, cv2.LINE_AA)

            # -------------------------------------------------------------
            # 3. MODERN FLOATING PILL HUD (POJOK KANAN ATAS)
            # -------------------------------------------------------------
            cur_num = self.active_idx + 1
            total_elems = len(self.entities)
            cam_display = self.camera_id.upper()
            mode_display = curr_type.upper()

            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.41
            font_thick = 1

            line1_cam = f"{cam_display} (1080p)  |  MODE: {mode_display}  |  "
            line1_zone = f"#{cur_num}/{total_elems} AKTIF"

            if curr_type == "tripwire":
                has_p1 = curr_elem.get("p1") is not None
                has_p2 = curr_elem.get("p2") is not None
                curr_dir = curr_elem.get("direction", "BOTH")
                if not has_p1:
                    line2 = f"Titik: 0/2 | [Klik] Set P1 | [D] Arah: {curr_dir} | {self.status_message}"
                elif not has_p2:
                    line2 = f"Titik: 1/2 | [Klik] Set P2 | [R-Click/U] Undo P1 | [D] Arah: {curr_dir} | {self.status_message}"
                else:
                    line2 = f"Titik: 2/2 (Kunci) | [R-Click/U] Edit P2 | [C] Gambar Ulang | [D] Arah: {curr_dir}"
            else:
                pts_cnt = len(curr_elem.get("points", []))
                min_req = 3 if curr_type in ("polygon", "exclusion") else 2
                if pts_cnt < min_req:
                    line2 = f"Titik: {pts_cnt}/{min_req} | [Klik] Tambah | [R-Click/U] Undo | [C] Reset | {self.status_message}"
                else:
                    line2 = f"Titik: {pts_cnt} (Sah) | [Klik] Tambah | [R-Click/U] Undo | [C] Redraw | {self.status_message}"

            (w_cam, h_cam), _ = cv2.getTextSize(line1_cam, font_face, font_scale, font_thick)
            (w_zone, h_zone), _ = cv2.getTextSize(line1_zone, font_face, font_scale, font_thick)
            w1 = w_cam + w_zone
            h1 = max(h_cam, h_zone)
            (w2, h2), _ = cv2.getTextSize(line2, font_face, font_scale, font_thick)

            dot_radius = 4
            dot_gap = 10
            dot_space = (dot_radius * 2) + dot_gap
            content_w = max(dot_space + w1, w2)
            pad_h = 12
            pad_v = 8
            line_spacing = 8

            pill_w = content_w + (pad_h * 2)
            pill_h = h1 + h2 + line_spacing + (pad_v * 2)

            pill_x2 = 1920 - 20
            pill_x1 = pill_x2 - pill_w
            pill_y1 = 20
            pill_y2 = pill_y1 + pill_h

            # -------------------------------------------------------------
            # 4. FOOTER PANDUAN PINTASAN
            # -------------------------------------------------------------
            guide_text = (
                "[P/T/B/E] Mode  |  [Klik L] Titik  |  [Klik R / U] Undo  |  "
                "[C] Gambar Ulang  |  [D] Arah TW  |  [N] Baru  |  [1-9/TAB] Nav  |  [S] Simpan"
            )
            (gw, gh), _ = cv2.getTextSize(guide_text, font_face, 0.38, font_thick)
            foot_pad_h = 16
            foot_w = gw + (foot_pad_h * 2)
            foot_h = 26
            foot_x1 = (1920 - foot_w) // 2
            foot_x2 = foot_x1 + foot_w
            foot_y2 = 1080 - 14
            foot_y1 = foot_y2 - foot_h

            hud_overlay = display.copy()
            cv2.rectangle(hud_overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), (24, 24, 27), -1)
            cv2.rectangle(hud_overlay, (foot_x1, foot_y1), (foot_x2, foot_y2), (24, 24, 27), -1)
            cv2.addWeighted(hud_overlay, 0.60, display, 0.40, 0, display)

            cv2.rectangle(display, (pill_x1, pill_y1), (pill_x2, pill_y2), (60, 60, 65), 1, cv2.LINE_AA)
            cv2.rectangle(display, (foot_x1, foot_y1), (foot_x2, foot_y2), (60, 60, 65), 1, cv2.LINE_AA)

            y_line1 = pill_y1 + pad_v + h1
            dot_cx = pill_x1 + pad_h + dot_radius
            dot_cy = y_line1 - (h1 // 2)
            cv2.circle(display, (dot_cx, dot_cy), dot_radius, active_color, -1, cv2.LINE_AA)

            text_cam_x = dot_cx + dot_radius + dot_gap
            cv2.putText(display, line1_cam, (text_cam_x, y_line1), font_face, font_scale, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(display, line1_zone, (text_cam_x + w_cam, y_line1), font_face, font_scale, active_color, 1, cv2.LINE_AA)

            y_line2 = y_line1 + line_spacing + h2
            cv2.putText(display, line2, (pill_x1 + pad_h, y_line2), font_face, font_scale, (180, 185, 195), 1, cv2.LINE_AA)

            text_foot_x = foot_x1 + foot_pad_h
            text_foot_y = foot_y1 + (foot_h + gh) // 2
            cv2.putText(display, guide_text, (text_foot_x, text_foot_y), font_face, 0.38, (215, 220, 225), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display)
            raw_key = cv2.waitKey(20) & 0xFF

            if raw_key != 255:
                # Tangani tombol non-printable terlebih dahulu
                if raw_key == 9:  # TAB Key (ASCII 9)
                    self._cycle_active_entity(direction=1)
                elif raw_key == 27 or raw_key == ord("q") or raw_key == ord("Q"):  # ESC / Q
                    break
                else:
                    try:
                        key_char = chr(raw_key).lower()
                    except Exception:
                        key_char = ""

                    if key_char == "t":
                        self._switch_mode("TRIPWIRE")
                    elif key_char == "p":
                        self._switch_mode("POLYGON")
                    elif key_char == "b":
                        self._switch_mode("BARRIER")
                    elif key_char == "e":
                        self._switch_mode("EXCLUSION")
                    elif key_char == "d":
                        self._toggle_direction()
                    elif key_char == "n":
                        self._create_new_entity()
                    elif raw_key == 26 or key_char in ("u", "z"):
                        self._undo_active_entity()
                    elif key_char == "c":
                        self._clear_active_entity()
                    elif key_char == "s":
                        self.save_zones()
                    elif key_char in "123456789":
                        self._select_entity(int(key_char) - 1)
                    elif key_char == "]":
                        self._cycle_active_entity(direction=1)
                    elif key_char == "[":
                        self._cycle_active_entity(direction=-1)

        cv2.destroyAllWindows()

    def _draw_entity_geometry(
        self,
        canvas: np.ndarray,
        elem: dict,
        idx: int,
        color: Tuple[int, int, int],
        is_active: bool,
    ) -> None:
        """Render entitas spasial dengan dual-stroke dark contrast guard (solid 1px)."""
        etype = elem.get("type", "polygon")

        if etype == "polygon":
            pts = elem.get("points", [])
            if len(pts) >= 2:
                np_pts = np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)
                is_closed = (len(pts) >= 3)
                # Dual-stroke contrast guard: 2px dark underlay + 1px foreground
                cv2.polylines(canvas, [np_pts], isClosed=is_closed,
                              color=(20, 20, 20), thickness=2, lineType=cv2.LINE_AA)
                cv2.polylines(canvas, [np_pts], isClosed=is_closed,
                              color=color, thickness=1, lineType=cv2.LINE_AA)

            for pt in pts:
                px, py = pt["x"], pt["y"]
                if is_active:
                    cv2.circle(canvas, (px, py), 5, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (px, py), 3, (20, 20, 20), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, color, 1, cv2.LINE_AA)

            if len(pts) >= 1:
                cx_l = int(np.mean([p["x"] for p in pts]))
                cy_l = int(np.mean([p["y"] for p in pts]))
                lbl = f"#{idx + 1} {elem.get('label', elem.get('zone_id'))} ({len(pts)}pts)"
                self._draw_label_pill(canvas, lbl, cx_l, cy_l, color, is_active)

        elif etype == "tripwire":
            p1, p2 = elem.get("p1"), elem.get("p2")
            if p1:
                p1x, p1y = p1["x"], p1["y"]
                if is_active:
                    cv2.circle(canvas, (p1x, p1y), 5, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (p1x, p1y), 5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (p1x, p1y), 2, color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (p1x, p1y), 3, (20, 20, 20), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (p1x, p1y), 2, color, 1, cv2.LINE_AA)
                # Label letter A dengan dark shadow
                cv2.putText(canvas, "A", (p1x - 14, p1y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.putText(canvas, "A", (p1x - 14, p1y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            if p2:
                p2x, p2y = p2["x"], p2["y"]
                if is_active:
                    cv2.circle(canvas, (p2x, p2y), 5, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (p2x, p2y), 5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (p2x, p2y), 2, color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (p2x, p2y), 3, (20, 20, 20), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (p2x, p2y), 2, color, 1, cv2.LINE_AA)
                # Label letter B dengan dark shadow
                cv2.putText(canvas, "B", (p2x + 7, p2y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.putText(canvas, "B", (p2x + 7, p2y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            if p1 and p2:
                pt1 = (p1["x"], p1["y"])
                pt2 = (p2["x"], p2["y"])
                # Dual-stroke contrast guard line
                cv2.line(canvas, pt1, pt2, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

                direction = elem.get("direction", "A_TO_B")
                mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                norm = np.hypot(dx, dy) or 1.0
                nx, ny = -dy / norm, dx / norm
                if direction == "B_TO_A":
                    nx, ny = -nx, -ny

                if direction in ("A_TO_B", "B_TO_A"):
                    tip_x = int(mid_x + nx * 28)
                    tip_y = int(mid_y + ny * 28)
                    cv2.arrowedLine(canvas, (mid_x, mid_y), (tip_x, tip_y),
                                    (20, 20, 20), 2, tipLength=0.35)
                    cv2.arrowedLine(canvas, (mid_x, mid_y), (tip_x, tip_y),
                                    color, 1, tipLength=0.35)
                else:
                    tip1_x, tip1_y = int(mid_x + nx * 20), int(mid_y + ny * 20)
                    tip2_x, tip2_y = int(mid_x - nx * 20), int(mid_y - ny * 20)
                    cv2.arrowedLine(canvas, (mid_x, mid_y), (tip1_x, tip1_y),
                                    (20, 20, 20), 2, tipLength=0.3)
                    cv2.arrowedLine(canvas, (mid_x, mid_y), (tip1_x, tip1_y),
                                    color, 1, tipLength=0.3)
                    cv2.arrowedLine(canvas, (mid_x, mid_y), (tip2_x, tip2_y),
                                    (20, 20, 20), 2, tipLength=0.3)
                    cv2.arrowedLine(canvas, (mid_x, mid_y), (tip2_x, tip2_y),
                                    color, 1, tipLength=0.3)

                lbl = f"#{idx + 1} {elem.get('label', elem.get('tripwire_id', 'TW'))} [{direction}]"
                self._draw_label_pill(canvas, lbl, mid_x, mid_y, color, is_active)

        elif etype == "barrier":
            pts = elem.get("points", [])
            if len(pts) >= 2:
                np_pts = np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)
                cv2.polylines(canvas, [np_pts], isClosed=False,
                              color=(20, 20, 20), thickness=2, lineType=cv2.LINE_AA)
                cv2.polylines(canvas, [np_pts], isClosed=False,
                              color=color, thickness=1, lineType=cv2.LINE_AA)

            for pt in pts:
                px, py = pt["x"], pt["y"]
                if is_active:
                    cv2.circle(canvas, (px, py), 5, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (px, py), 3, (20, 20, 20), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, color, 1, cv2.LINE_AA)

            if len(pts) >= 2:
                cx_l = (pts[0]["x"] + pts[-1]["x"]) // 2
                cy_l = (pts[0]["y"] + pts[-1]["y"]) // 2
                lbl = f"#{idx + 1} {elem.get('label', elem.get('barrier_id', 'Bar'))} ({len(pts)}pts)"
                self._draw_label_pill(canvas, lbl, cx_l, cy_l, color, is_active)

        elif etype == "exclusion":
            pts = elem.get("points", [])
            ex_color = (220, 220, 220) if is_active else (140, 140, 150)
            if len(pts) >= 2:
                np_pts = np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)
                is_closed = (len(pts) >= 3)
                cv2.polylines(canvas, [np_pts], isClosed=is_closed,
                              color=(20, 20, 20), thickness=2, lineType=cv2.LINE_AA)
                cv2.polylines(canvas, [np_pts], isClosed=is_closed,
                              color=ex_color, thickness=1, lineType=cv2.LINE_AA)

            for pt in pts:
                px, py = pt["x"], pt["y"]
                if is_active:
                    cv2.circle(canvas, (px, py), 5, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, ex_color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (px, py), 3, (20, 20, 20), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, ex_color, 1, cv2.LINE_AA)

            if len(pts) >= 1:
                cx_l = int(np.mean([p["x"] for p in pts]))
                cy_l = int(np.mean([p["y"] for p in pts]))
                lbl = f"#{idx + 1} EXCL: {elem.get('label', elem.get('mask_id', 'EX'))}"
                self._draw_label_pill(canvas, lbl, cx_l, cy_l, ex_color, is_active)

    def _draw_label_pill(self, canvas: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], is_active: bool) -> None:
        """Helper render label badge pill hairline — centered pada titik x,y, semi-transparan via outer blending."""
        font_sc = 0.40 if is_active else 0.34
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_sc, 1)
        pad_h, pad_v = 4, 2
        lx1 = max(0, x - tw // 2 - pad_h)
        ly1 = max(0, y - th - pad_v * 2)
        lx2 = min(1919, lx1 + tw + pad_h * 2)
        ly2 = min(1079, y + pad_v)
        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (16, 20, 26), -1)
        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (lx1 + pad_h, ly2 - pad_v),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sc,
                    (255, 255, 255) if is_active else (160, 165, 175), 1, cv2.LINE_AA)


def run_test_save(camera_id: str = "cam_01") -> int:
    """Mode pengujian headless untuk automated testing tanpa GUI window."""
    print(f"[TEST-SAVE] Menjalankan pengujian headless auto-backup & validasi untuk '{camera_id}'...")
    calibrator = ROICalibrator(camera_id=camera_id, skip_frame_load=True)

    # Simpan state asli
    orig_polys = [dict(z) for z in calibrator.polygons]
    orig_tws = [dict(t) for t in calibrator.tripwires]

    # Set data uji valid
    calibrator.polygons = [
        {
            "zone_id": "zone_01",
            "label": "Test Polygon",
            "type": "polygon",
            "active": True,
            "color_hex": "#00FF00",
            "points": [{"x": 100, "y": 100}, {"x": 400, "y": 100}, {"x": 400, "y": 400}, {"x": 100, "y": 400}],
            "trigger_on": ["linger"],
            "linger_threshold_sec": 3,
        }
    ]
    calibrator.tripwires = [
        {
            "tripwire_id": "tw_01",
            "label": "Test Tripwire",
            "active": True,
            "color_hex": "#FFCC00",
            "p1": {"x": 200, "y": 100},
            "p2": {"x": 200, "y": 500},
            "direction": "A_TO_B",
            "target_classes": ["person"],
            "debounce_sec": 3.0,
        }
    ]

    success = calibrator.save_zones()
    assert success is True, "Penyimpanan konfigurasi spasial harus berhasil"

    bak_path = calibrator.roi_path.with_suffix(".json.bak")
    assert bak_path.exists(), f"Berkas backup tidak ditemukan: {bak_path}"

    # Kembalikan state asli
    calibrator.polygons = orig_polys
    calibrator.tripwires = orig_tws
    calibrator.save_zones()

    print(f"  [PASS] Auto-backup '{bak_path.name}' dan validasi Pydantic terverifikasi.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Facility Management — Visual ROI Calibrator (1080p)")
    parser.add_argument("--camera", "-c", default="cam_01", help="ID kamera (default: cam_01)")
    parser.add_argument("--source", "-s", default=None, help="Override URL RTSP atau file video")
    parser.add_argument("--test-save", action="store_true", help="Mode headless test auto-backup & validasi skema")

    args = parser.parse_args()

    if args.test_save:
        sys.exit(run_test_save(camera_id=args.camera))

    calibrator = ROICalibrator(camera_id=args.camera, source_override=args.source)
    calibrator.run_gui()


if __name__ == "__main__":
    main()
