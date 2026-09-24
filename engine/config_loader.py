from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic import model_validator
from typing import Literal, List, Any, Tuple

load_dotenv(override=False)

class ResolutionConfig(BaseModel):
    capture_width: int = Field(1920, ge=1, description="Lebar frame capture penuh")
    capture_height: int = Field(1080, ge=1, description="Tinggi frame capture penuh")
    ai_width: int = Field(640, ge=32, description="Lebar canvas untuk inferensi AI")
    ai_height: int = Field(360, ge=32, description="Tinggi canvas untuk inferensi AI")

    @model_validator(mode="after")
    def ai_dims_smaller_than_capture(self) -> "ResolutionConfig":
        if self.ai_width > self.capture_width or self.ai_height > self.capture_height:
            raise ValueError(
                f"AI canvas ({self.ai_width}x{self.ai_height}) tidak boleh lebih besar "
                f"dari capture resolution ({self.capture_width}x{self.capture_height})"
            )
        return self


class CaptureConfig(BaseModel):
    queue_maxsize: int = Field(1, ge=1, le=4, description="Ukuran queue RTSP (1=drop-on-full)")
    reconnect_delay_sec: float = Field(3.0, ge=0.5, description="Jeda antar percobaan reconnect (detik)")
    max_reconnect_attempts: int = Field(10, ge=1, description="Maksimum percobaan reconnect sebelum berhenti")
    frame_timeout_sec: float = Field(8.0, ge=1.0, description="Batas timeout frame read sebelum reconnect (detik)")
    keyframe_warmup_frames: int = Field(5, ge=0, description="Jumlah frame awal yang diabaikan saat stream baru")


class DiskGuardCameraConfig(BaseModel):
    snapshot_dir: str = Field(..., description="Path relatif direktori snapshot")
    max_size_gb: float = Field(2.0, gt=0, description="Batas ukuran snapshot (GB)")
    purge_oldest_pct: float = Field(20.0, gt=0, le=100, description="Persentase file tertua yang dihapus saat trigger")


class CameraConfig(BaseModel):
    camera_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    rtsp_url: str = Field(..., description="URL RTSP sudah resolved — JANGAN LOG")
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    disk_guard: DiskGuardCameraConfig

    class Config:
        json_schema_extra = {"sensitive_fields": ["rtsp_url"]}

    def safe_repr(self) -> str:
        return (
            f"CameraConfig(camera_id={self.camera_id!r}, "
            f"display_name={self.display_name!r}, "
            f"rtsp_url='rtsp://***:***@***', "
            f"ai_res={self.resolution.ai_width}x{self.resolution.ai_height})"
        )


class ROIPoint(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)

    @model_validator(mode="before")
    @classmethod
    def parse_point(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            return {"x": int(data[0]), "y": int(data[1])}
        return data


class ROIZone(BaseModel):

    zone_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    type: Literal["polygon", "line", "rectangle"] = "polygon"
    active: bool = True
    color_hex: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    points: List[ROIPoint] = Field(..., min_length=3, description="Minimal 3 titik untuk poligon")
    trigger_on: List[Literal["enter", "exit", "linger"]] = Field(..., min_length=1)
    linger_threshold_sec: Optional[int] = Field(None, ge=1)
    target_classes: Optional[List[str]] = None

    @model_validator(mode="after")
    def linger_requires_threshold(self) -> "ROIZone":
        if "linger" in self.trigger_on and self.linger_threshold_sec is None:
            raise ValueError(
                f"Zone '{self.zone_id}': trigger 'linger' membutuhkan linger_threshold_sec"
            )
        return self


class TripwireRule(BaseModel):
    tripwire_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    active: bool = True
    color_hex: str = Field("#FF5500", pattern=r"^#[0-9A-Fa-f]{6}$")
    p1: ROIPoint
    p2: ROIPoint
    direction: Literal["BOTH", "A_TO_B", "B_TO_A"] = "BOTH"
    target_classes: List[str] = Field(
        default_factory=lambda: ["person", "car", "motorcycle", "bus", "truck"]
    )
    debounce_sec: float = Field(3.0, ge=0.1)


class BarrierRule(BaseModel):
    barrier_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    active: bool = True
    color_hex: str = Field("#FF0055", pattern=r"^#[0-9A-Fa-f]{6}$")
    points: List[ROIPoint] = Field(..., min_length=2, description="Polyline minimal 2 titik")
    direction: Literal["BOTH", "A_TO_B", "B_TO_A"] = "BOTH"
    target_classes: List[str] = Field(
        default_factory=lambda: ["person", "car", "motorcycle", "bus", "truck"]
    )
    debounce_sec: float = Field(3.0, ge=0.1)


class ExclusionMask(BaseModel):
    mask_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    active: bool = True
    color_hex: str = Field("#646464", pattern=r"^#[0-9A-Fa-f]{6}$")
    points: List[ROIPoint] = Field(..., min_length=3, description="Poligon mask minimal 3 titik")
    ignore_types: List[Literal["motion", "dwell", "all"]] = Field(
        default_factory=lambda: ["all"]
    )


class SafeWalkwayRule(BaseModel):
    walkway_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    active: bool = True
    color_hex: str = Field("#00FF88", pattern=r"^#[0-9A-Fa-f]{6}$")
    points: List[ROIPoint] = Field(..., min_length=3, description="Poligon walkway minimal 3 titik")
    violation_timeout_sec: float = Field(30.0, ge=1.0)
    vehicle_proximity_filter: bool = True
    proximity_radius_px: float = Field(90.0, ge=10.0)


class DensityRule(BaseModel):
    rule_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    active: bool = True
    zone_ref: str = Field(..., min_length=1, description="Referensi ke zone_id poligon target")
    max_allowed_objects: int = Field(5, ge=1)
    min_dwell_sec: float = Field(5.0, ge=0.5)


class ROIZonesConfig(BaseModel):

    schema_version: str = Field(..., pattern=r"^\d+\.\d+$")
    camera_id: str = Field(..., min_length=1)
    polygons: List[ROIZone] = Field(default_factory=list)
    tripwires: List[TripwireRule] = Field(default_factory=list)
    barriers: List[BarrierRule] = Field(default_factory=list)
    exclusion_masks: List[ExclusionMask] = Field(default_factory=list)
    safe_walkways: List[SafeWalkwayRule] = Field(default_factory=list)
    density_rules: List[DensityRule] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_zones(cls, data: Any) -> Any:
        """Migrasi transparan: map key 'zones' legacy ke 'polygons'."""
        if isinstance(data, dict):
            if "zones" in data and "polygons" not in data:
                data["polygons"] = data.pop("zones")
        return data

    @property
    def zones(self) -> List[ROIZone]:
        """Backward-compatibility alias: mengembalikan self.polygons."""
        return self.polygons

    @property
    def active_zones(self) -> List[ROIZone]:
        """Mengembalikan hanya zona poligon yang aktif."""
        return [z for z in self.polygons if z.active]

    @property
    def active_tripwires(self) -> List[TripwireRule]:
        return [t for t in self.tripwires if t.active]

    @property
    def active_barriers(self) -> List[BarrierRule]:
        return [b for b in self.barriers if b.active]

    @property
    def active_exclusion_masks(self) -> List[ExclusionMask]:
        return [m for m in self.exclusion_masks if m.active]

    @property
    def active_safe_walkways(self) -> List[SafeWalkwayRule]:
        return [w for w in self.safe_walkways if w.active]

    @property
    def active_density_rules(self) -> List[DensityRule]:
        return [d for d in self.density_rules if d.active]

_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")

_PASSWORD_VAR_PATTERN = re.compile(r".*_PASS$", re.IGNORECASE)


def _resolve_env_vars(raw: str) -> str:

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(
                f"Variabel environment '{var_name}' tidak ditemukan. "
                f"Pastikan .env sudah dikonfigurasi (lihat .env.example)."
            )
        if _PASSWORD_VAR_PATTERN.match(var_name):
            value = quote(value, safe="")
        return value

    return _PLACEHOLDER_PATTERN.sub(replacer, raw)

def load_camera_config(config_path: str | Path) -> CameraConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config kamera tidak ditemukan: {path}")

    raw = path.read_text(encoding="utf-8")
    resolved = _resolve_env_vars(raw)
    data = json.loads(resolved)
    return CameraConfig.model_validate(data)


def load_roi_zones(roi_path: str | Path) -> ROIZonesConfig:
    path = Path(roi_path)
    if not path.exists():
        raise FileNotFoundError(f"File ROI zones tidak ditemukan: {path}")

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    return ROIZonesConfig.model_validate(data)
