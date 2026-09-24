
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Tambahkan root project ke sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from engine.config_loader import load_camera_config, load_roi_zones, CameraConfig, ROIZonesConfig
from pydantic import ValidationError


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI Colors
# ═══════════════════════════════════════════════════════════════════════════════

def _ok(s: str) -> str:
    return f"\033[92m{s}\033[0m"

def _fail(s: str) -> str:
    return f"\033[91m{s}\033[0m"

def _warn(s: str) -> str:
    return f"\033[93m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AuditResult:
    cam_id: str
    config_valid: bool = False
    roi_valid: bool = False
    config_error: Optional[str] = None
    roi_error: Optional[str] = None
    zone_count: int = 0
    active_zone_count: int = 0
    ai_resolution: Optional[str] = None

    @property
    def all_valid(self) -> bool:
        return self.config_valid and self.roi_valid

    def error_detail(self) -> str:
        errors = []
        if self.config_error:
            errors.append(f"config: {self.config_error}")
        if self.roi_error:
            errors.append(f"roi: {self.roi_error}")
        return " | ".join(errors) if errors else "-"

# AUDIT FUNCTIONS
def _truncate(s: str, max_len: int = 60) -> str:
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def audit_camera(cam_dir: Path) -> AuditResult:
    cam_id = cam_dir.name
    result = AuditResult(cam_id=cam_id)
    config_path = cam_dir / "config.json"
    if not config_path.exists():
        result.config_error = "config.json tidak ditemukan"
    else:
        try:
            cfg = load_camera_config(config_path)
            result.config_valid = True
            result.ai_resolution = f"{cfg.resolution.ai_width}x{cfg.resolution.ai_height}"
        except FileNotFoundError as exc:
            result.config_error = str(exc)
        except ValueError as exc:
            result.config_error = _truncate(str(exc))
        except ValidationError as exc:
            # Ambil pesan error pertama yang paling relevan
            errors = exc.errors()
            first = errors[0]
            loc = " → ".join(str(l) for l in first["loc"])
            result.config_error = _truncate(f"{loc}: {first['msg']}")
        except Exception as exc:
            result.config_error = _truncate(f"{type(exc).__name__}: {exc}")

    roi_path = cam_dir / "roi_zones.json"
    if not roi_path.exists():
        result.roi_error = "roi_zones.json tidak ditemukan"
    else:
        try:
            roi = load_roi_zones(roi_path)
            result.roi_valid = True
            result.zone_count = len(roi.zones)
            result.active_zone_count = len(roi.active_zones)
            if result.config_valid:
                try:
                    cfg_again = load_camera_config(config_path)
                    if roi.camera_id != cfg_again.camera_id:
                        result.roi_valid = False
                        result.roi_error = (
                            f"Mismatch camera_id: roi={roi.camera_id!r} "
                            f"vs config={cfg_again.camera_id!r}"
                        )
                except Exception:
                    pass  # Sudah di-handle di config audit

        except FileNotFoundError as exc:
            result.roi_error = str(exc)
        except ValidationError as exc:
            errors = exc.errors()
            first = errors[0]
            loc = " → ".join(str(l) for l in first["loc"])
            result.roi_error = _truncate(f"{loc}: {first['msg']}")
        except Exception as exc:
            result.roi_error = _truncate(f"{type(exc).__name__}: {exc}")

    return result

# TABLE RENDERER
def _render_table(results: list[AuditResult]) -> None:
    # Column widths
    w_id = max(8, max(len(r.cam_id) for r in results) + 2)
    w_cfg = 14
    w_roi = 11
    w_zones = 12
    w_ai = 12
    w_err = 42

    # Header
    header = (
        f"  {'cam_id':<{w_id}} │ {'config_valid':<{w_cfg}} │ "
        f"{'roi_valid':<{w_roi}} │ {'zones(act)':<{w_zones}} │ "
        f"{'ai_canvas':<{w_ai}} │ {'error_detail':<{w_err}}"
    )
    sep = "  " + "-" * (w_id + w_cfg + w_roi + w_zones + w_ai + w_err + 20)

    print()
    print(_bold(header))
    print(sep)

    for r in results:
        cfg_icon = _ok("✓") if r.config_valid else _fail("✗")
        roi_icon = _ok("✓") if r.roi_valid else (_fail("✗") if not r.roi_valid and r.config_valid else _warn("-"))
        zones_str = f"{r.active_zone_count}/{r.zone_count}" if r.roi_valid else "-"
        ai_str = r.ai_resolution or "-"
        err_str = _truncate(r.error_detail(), w_err)

        print(
            f"  {r.cam_id:<{w_id}} │ {cfg_icon:<{w_cfg}} │ "
            f"{roi_icon:<{w_roi}} │ {zones_str:<{w_zones}} │ "
            f"{ai_str:<{w_ai}} │ {err_str}"
        )

    print(sep)
    print()

# MAIN RUNNER

def run_audit(camera_ids: Optional[list[str]] = None, silent: bool = False) -> int:
    cameras_dir = PROJECT_ROOT / "cameras"
    cam_dirs = [
        d for d in cameras_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ]

    if not cam_dirs:
        if not silent:
            print(_warn("Tidak ada direktori kamera ditemukan di cameras/"))
        return 1

    if camera_ids:
        cam_dirs = [d for d in cam_dirs if d.name in camera_ids]
        if not cam_dirs:
            if not silent:
                print(_fail(f"Kamera tidak ditemukan: {camera_ids}"))
            return 1

    results = [audit_camera(d) for d in sorted(cam_dirs)]
    valid_count = sum(1 for r in results if r.all_valid)
    invalid_count = len(results) - valid_count

    if not silent or invalid_count > 0:
        print(f"\n{'=' * 70}")
        print(f"  FACILITY MANAGEMENT -- Schema Audit Tool")
        print(f"  Kamera ditemukan: {len(cam_dirs)}")
        print(f"{'=' * 70}")
        _render_table(results)

    if invalid_count == 0:
        if not silent:
            print(_ok(f"  Audit selesai -- Semua {valid_count} kamera valid [OK]"))
            print()
        return 0
    else:
        print(_fail(f"  Audit selesai -- {invalid_count} dari {len(results)} kamera bermasalah [FAIL]"))
        print(_warn("  Perbaiki error di atas sebelum menjalankan engine utama."))
        print()
        return 1


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Facility Management — Schema Audit Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--camera", "-c",
        nargs="+",
        metavar="CAM_ID",
        help="ID kamera yang akan diaudit (default: semua)",
    )
    args = parser.parse_args()

    import os
    os.system("")  # Aktifkan ANSI di Windows

    sys.exit(run_audit(camera_ids=args.camera))
