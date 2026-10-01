"""
tools/verify_cam01_live_frame.py
================================
Memverifikasi alokasi slot pada snapshot cam_01 terkini pasca perbaikan S2 & S3.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from engine.config_loader import load_roi_zones
from engine.detector_impl import YOLO11nDetector
from engine.parking.car_slot import CarSlotTracker
from engine.tracker_interface import TrackResult


def main():
    print("=" * 70)
    print("VERIFIKASI ALOKASI SLOT CAM_01 PADA FRAME SNAPSHOT TERBARU")
    print("=" * 70)

    snap_dir = Path("cameras/cam_01/snapshots")
    snaps = sorted(snap_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snaps:
        print("ERROR: Snapshot tidak ditemukan!")
        return

    frame_path = snaps[0]
    print(f"Snapshot: {frame_path.name}")
    frame = cv2.imread(str(frame_path))

    detector = YOLO11nDetector(confidence_threshold=0.20, model_path=Path("weights/yolo11n.onnx"))
    dets = detector.detect(frame)
    print(f"Deteksi mentah: {len(dets)} objek")

    tracks = [
        TrackResult(
            track_id=i + 1,
            class_label=d.class_label,
            class_id=d.class_id,
            confidence=d.confidence,
            bbox=d.bbox,
            is_confirmed=True,
        )
        for i, d in enumerate(dets)
    ]

    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    tracker = CarSlotTracker(dwell_threshold_sec=0.0, debug_diagnostics=True)
    stats = tracker.update(tracks, roi_cfg.polygons, scale_x=3.0, scale_y=3.0, current_time=100.0)

    print("\nHasil Alokasi Slot:")
    for zid in ["zone_01", "zone_02", "zone_03", "zone_04", "zone_05", "zone_06", "zone_07", "zone_08"]:
        st = tracker.slot_states[zid]
        status_str = f"[{st.phase}]"
        tid_str = f"Track #{st.track_id}" if st.track_id else "None"
        print(f"  * {zid} ({st.label}): {status_str:<12} | {tid_str:<10} | Class: {st.vehicle_class}")

    print(f"\nTotal Okupansi: {stats['occupied_slots']}/8 Slots")

    # Assertions
    s2 = tracker.slot_states["zone_02"]
    s3 = tracker.slot_states["zone_03"]
    assert s2.phase == "OCCUPIED", f"S2 harus OCCUPIED, didapat {s2.phase}"
    assert s3.phase == "VACANT", f"S3 harus VACANT (bebas crosstalk!), didapat {s3.phase}"
    print("\n✓ SUKSES: Mobil di S2 sah terakuisisi ke S2, dan S3 bersih VACANT!")


if __name__ == "__main__":
    main()
