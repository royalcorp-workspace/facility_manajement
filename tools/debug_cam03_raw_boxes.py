"""
tools/debug_cam03_raw_boxes.py
==============================
Skrip audit diagnostik 1-frame untuk cam_03:
1. Ekstraksi raw detections YOLO11n (conf >= 0.10) pada AI Canvas 1280x720.
2. Pelacakan bertahap (Tahap A s.d. E) di mana proposal motor gugur dalam pipeline motorcycle_block.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from engine.config_loader import load_roi_zones
from engine.detector_impl import YOLO11nDetector
from engine.parking.motorcycle_block import (
    MotorcycleBlockTracker,
    deduplicate_motorcycle_tracks,
    is_valid_motorcycle_anatomy,
)
from engine.parking.spatial import bbox_ios, bbox_polygon_overlap_ratio
from engine.tracker_interface import TrackResult


def get_live_frame() -> Tuple[np.ndarray, str]:
    """Ambil frame raw langsung dari RTSP cam_03 atau snapshot disk."""
    rtsp_url = "rtsp://user:royalpusat314@192.212.160.174:554/Streaming/Channels/101"
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame, "Live RTSP cam_03 (1920x1080 raw)"

    # Fallback jika RTSP offline
    snap_dir = Path("cameras/cam_03/snapshots")
    snaps = sorted(snap_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if snaps:
        f = cv2.imread(str(snaps[0]))
        if f is not None:
            return f, f"Snapshot disk: {snaps[0].name}"

    raise RuntimeError("Tidak dapat mengambil frame live maupun snapshot!")


def main():
    print("=" * 100)
    print("LAPORAN AUDIT DIAGNOSTIK EMPIRIS: UNDER-DETECTION MOTOR cam_03 (Branch v2)")
    print("=" * 100)

    # 1. Akuisisi Frame
    raw_img, source_desc = get_live_frame()
    h_orig, w_orig = raw_img.shape[:2]
    print(f"[1] Sumber Frame: {source_desc}")
    print(f"    Resolusi Asli Kamera: {w_orig}x{h_orig}")

    ai_w, ai_h = 1280, 720
    ai_frame = cv2.resize(raw_img, (ai_w, ai_h))
    print(f"    AI Canvas Target : {ai_w}x{ai_h}")

    # Simpan frame untuk verifikasi visual jika dibutuhkan
    cv2.imwrite("tools/cam03_audit_canvas_1280x720.jpg", ai_frame)

    # 2. Muat Poligon ROI
    roi_cfg = load_roi_zones(Path("cameras/cam_03/roi_zones.json"))
    poly = roi_cfg.polygons[0]
    sx = ai_w / float(w_orig)
    sy = ai_h / float(h_orig)
    scale_x = float(w_orig) / ai_w  # 1.5 jika capture 1920x1080
    scale_y = float(h_orig) / ai_h  # 1.5 jika capture 1920x1080

    pts_720 = np.array([[int(round(pt.x * sx)), int(round(pt.y * sy))] for pt in poly.points], dtype=np.int32)
    print(f"    Poligon 'zone_01' (720p Canvas): {len(pts_720)} titik koordinat")

    # 3. Inferensi Mentah YOLO11n (conf >= 0.10)
    detector = YOLO11nDetector(
        confidence_threshold=0.10,
        iou_threshold=0.45,
        target_classes=["motorcycle", "bicycle"],
        model_path=Path("weights/yolo11n.onnx"),
    )
    raw_dets = detector.detect(ai_frame)

    total_frame_dets = len(raw_dets)
    print(f"\n[2] RAW DETECTIONS YOLO11n (conf >= 0.10):")
    print(f"    Total Bounding Box Motor/Sepeda di Seluruh Frame: {total_frame_dets} unit")

    # Filter yang menyentuh atau berada di dalam area poligon
    audit_rows = []
    for i, d in enumerate(raw_dets):
        rx1, ry1, rx2, ry2 = d.bbox
        bw = float(rx2 - rx1)
        bh = float(ry2 - ry1)
        cx = float((rx1 + rx2) / 2.0)
        cy = float((ry1 + ry2) / 2.0)
        y2 = float(ry2)
        ar = bh / bw if bw > 0 else 0.0

        wheel_pt = (int(round(cx)), int(round(y2)))
        center_pt = (int(round(cx)), int(round(cy)))

        d_wheel = float(cv2.pointPolygonTest(pts_720, wheel_pt, True))
        d_center = float(cv2.pointPolygonTest(pts_720, center_pt, True))
        ovl_ratio = bbox_polygon_overlap_ratio(d.bbox, pts_720)

        # Skala konversi 1080p untuk evaluasi anatomi
        # pipeline mengirim scale_x = ai_w / ref_w = 0.6667
        # motorcycle_block menghitung: sx = (1.0 / scale_x) if scale_x > 1.0 else scale_x
        sx_block = (ai_w / 1920.0)
        sy_block = (ai_h / 1080.0)
        bw_1080p = bw / sx_block
        bh_1080p = bh / sy_block
        area_1080p = bw_1080p * bh_1080p

        # Detail Evaluasi Anatomi
        anat_reasons = []
        if bw_1080p > 580.0:
            anat_reasons.append(f"w_1080p={bw_1080p:.0f}>580")
        if bh_1080p > 380.0:
            anat_reasons.append(f"h_1080p={bh_1080p:.0f}>380")
        if area_1080p > 140000.0:
            anat_reasons.append(f"area={area_1080p:.0f}>140k")
        if ar < 0.35:
            anat_reasons.append(f"ar={ar:.2f}<0.35")
        if ar > 2.25:
            anat_reasons.append(f"ar={ar:.2f}>2.25")
        pass_anatomy = is_valid_motorcycle_anatomy(d.bbox, sx_block, sy_block)

        # Tahap A: Conf Gate (Acquisition 0.10, Retention 0.08)
        pass_conf_acq = (d.confidence >= 0.10)
        pass_conf_ret = (d.confidence >= 0.08)

        # Tahap C: Dual Containment Gate
        if d_wheel < -15.0:
            pass_containment = False
        elif d_wheel >= 0.0:
            pass_containment = (ovl_ratio >= 0.08)
        else:
            pass_containment = not (d_center < -8.0 and ovl_ratio < 0.02)

        pass_wheel_in = (d_wheel >= -15.0)
        pass_ovl = (ovl_ratio >= 0.02)

        # Status Keseluruhan Tahap A-C
        fail_reasons = []
        if not pass_conf_acq:
            fail_reasons.append(f"CONF_LOW({d.confidence:.2f}<0.10)")
        if not pass_anatomy:
            fail_reasons.append(f"ANATOMY({','.join(anat_reasons)})")
        if not pass_containment:
            fail_reasons.append(f"CONTAINMENT_FAIL(dw={d_wheel:.1f},ovl={ovl_ratio*100:.0f}%)")

        status_str = "LOLOS TAHAP A-C" if not fail_reasons else " | ".join(fail_reasons)

        audit_rows.append({
            "raw_id": i + 1,
            "class": d.class_label,
            "conf": d.confidence,
            "bbox": [round(c, 1) for c in d.bbox],
            "w": round(bw, 1),
            "h": round(bh, 1),
            "ar": round(ar, 2),
            "cx": round(cx, 1),
            "y2": round(y2, 1),
            "d_wheel": round(d_wheel, 1),
            "d_center": round(d_center, 1),
            "ovl_pct": round(ovl_ratio * 100.0, 1),
            "pass_conf_acq": pass_conf_acq,
            "pass_conf_ret": pass_conf_ret,
            "pass_anatomy": pass_anatomy,
            "anat_reasons": anat_reasons,
            "pass_wheel_in": pass_wheel_in,
            "pass_ovl": pass_ovl,
            "pass_containment": pass_containment,
            "fail_reasons": fail_reasons,
            "status_str": status_str,
            "det": d,
        })

    # Sort berdasarkan posisi horizontal (cx)
    audit_rows.sort(key=lambda r: r["cx"])

    # Filter yang relevan dengan poligon (overlap > 0 ATAU d_wheel >= -30)
    polygon_candidates = [r for r in audit_rows if r["ovl_pct"] > 0.0 or r["d_wheel"] >= -30.0]

    print(f"    Total Bounding Box Terkait Poligon 'zone_01': {len(polygon_candidates)} unit")

    print("\n" + "=" * 135)
    print("TABEL 1: DATA DETEKSI MENTAH YOLO (conf >= 0.10) DI AREA POLIGON PARKIR MOTOR KIRI")
    print("=" * 135)
    print(
        f"{'ID':<4} | {'Class':<10} | {'Conf':<5} | {'Bbox [x1,y1,x2,y2]':<23} | "
        f"{'w':<4} | {'h':<4} | {'h/w':<4} | {'Tapak (cx,y2)':<14} | "
        f"{'d_whl':<6} | {'Ovl %':<6} | {'Status Tahap A-C'}"
    )
    print("-" * 135)

    for r in polygon_candidates:
        b = r["bbox"]
        b_str = f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"
        tp_str = f"({r['cx']:.0f},{r['y2']:.0f})"
        print(
            f"#{r['raw_id']:<3} | {r['class']:<10} | {r['conf']:<5.2f} | {b_str:<23} | "
            f"{r['w']:<4.0f} | {r['h']:<4.0f} | {r['ar']:<4.2f} | {tp_str:<14} | "
            f"{r['d_wheel']:<6.1f} | {r['ovl_pct']:<5.1f}% | {r['status_str']}"
        )
    print("-" * 135)

    # 4. Rangkuman Keguguran Tahap A - C
    gugur_a = sum(1 for r in polygon_candidates if not r["pass_conf_acq"])
    gugur_b = sum(1 for r in polygon_candidates if not r["pass_anatomy"])
    gugur_c = sum(1 for r in polygon_candidates if not r["pass_containment"])
    lolos_abc = [r for r in polygon_candidates if len(r["fail_reasons"]) == 0]

    print(f"\nRANGKUMAN GUGUR TAHAP A s.d. C:")
    print(f"  * Tahap A (Confidence Gate < 0.10)  : {gugur_a} motor gugur")
    print(f"  * Tahap B (Anatomy & Size Filter)   : {gugur_b} motor gugur")
    print(f"  * Tahap C (Polygon Containment Gate): {gugur_c} motor gugur")
    print(f"  * Lolos Tahap A, B, C               : {len(lolos_abc)} proposal")

    # 5. Evaluasi Tahap D: Local Deduplication (IoS & IoU Suppression)
    print("\n" + "=" * 100)
    print("[3] AUDIT TAHAP D: DEDUPLIKASI SPASIAL (deduplicate_motorcycle_tracks)")
    print("=" * 100)

    # Format lolos_abc sebagai TrackResult
    tracks_for_dedup = [
        TrackResult(
            track_id=r["raw_id"],
            class_label=r["class"],
            class_id=3 if r["class"] == "motorcycle" else 1,
            confidence=r["conf"],
            bbox=tuple(r["bbox"]),
            is_confirmed=True,
        )
        for r in lolos_abc
    ]

    scale_factor = max(1.0, (ai_w / 1920.0) / (1.0 / 3.0))  # 2.0
    deduped_tracks = deduplicate_motorcycle_tracks(
        tracks_for_dedup,
        iou_thresh=0.55,
        ios_thresh=0.85,
        min_dx_px=6.0 * scale_factor,
        cumulative_overlap_thresh=0.85,
    )

    deduped_ids = {t.track_id for t in deduped_tracks}
    dropped_by_dedup = [t for t in tracks_for_dedup if t.track_id not in deduped_ids]

    print(f"  * Input ke Tahap D           : {len(tracks_for_dedup)} unit")
    print(f"  * Lolos pasca-deduplikasi    : {len(deduped_tracks)} unit")
    print(f"  * Gugur di Tahap D (Eliminasi): {len(dropped_by_dedup)} unit")
    for dt in dropped_by_dedup:
        print(f"    - Gugur: ID #{dt.track_id} conf={dt.confidence:.2f} bbox={dt.bbox}")

    # 6. Evaluasi Tahap E: Spatial Anchor Memory & Latch
    print("\n" + "=" * 100)
    print("[4] AUDIT TAHAP E: SPATIAL ANCHOR MEMORY & LATCHING (MotorcycleBlockTracker)")
    print("=" * 100)

    tracker_warmup = MotorcycleBlockTracker(block_capacity=30, stationary_dwell_sec=0.0)
    stats_warm = tracker_warmup.update(
        tracks=deduped_tracks,
        polygons=roi_cfg.polygons,
        scale_x=sx_block,
        scale_y=sy_block,
        current_time=100.0,
        is_warmup=True,
    )
    print(f"  * Mode Cold-Start (is_warmup=True):")
    print(f"    - Terdaftar di Memory : {len(tracker_warmup._stationary_motor_units)} unit")
    print(f"    - Occupied Slots      : {stats_warm['occupied_slots']} / 30")

    tracker_normal = MotorcycleBlockTracker(block_capacity=30, stationary_dwell_sec=0.0)
    stats_normal = tracker_normal.update(
        tracks=deduped_tracks,
        polygons=roi_cfg.polygons,
        scale_x=sx_block,
        scale_y=sy_block,
        current_time=100.0,
        is_warmup=False,
    )
    print(f"\n  * Mode Normal Streaming 1-Frame (is_warmup=False, hits=1):")
    print(f"    - Terdaftar di Memory : {len(tracker_normal._stationary_motor_units)} unit")
    print(f"    - Unit Ter-latch      : {sum(1 for u in tracker_normal._stationary_motor_units.values() if u.is_latched)} unit")
    print(f"    - Occupied Slots      : {stats_normal['occupied_slots']} / 30")

    # Render Visualisasi Audit
    vis = ai_frame.copy()
    cv2.polylines(vis, [pts_720], isClosed=True, color=(0, 255, 0), thickness=2)
    for r in polygon_candidates:
        x1, y1, x2, y2 = [int(round(c)) for c in r["bbox"]]
        color = (0, 255, 255) if len(r["fail_reasons"]) == 0 else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        label = f"#{r['raw_id']} {r['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite("tools/cam03_audit_detections_vis.jpg", vis)
    print(f"\n[5] Gambar visualisasi audit disimpan di: tools/cam03_audit_detections_vis.jpg")


if __name__ == "__main__":
    main()

