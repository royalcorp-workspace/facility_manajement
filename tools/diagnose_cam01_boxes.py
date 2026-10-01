"""
tools/diagnose_cam01_boxes.py
==============================
Skrip audit diagnostik 1-frame empiris untuk cam_01.
Mengekstrak seluruh deteksi mentah YOLO11n pada AI Canvas (640x360),
dan memetakan geometri/stance probe ke poligon S1, S2, dan S3 dari roi_zones.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from engine.detector_impl import YOLO11nDetector
from engine.parking.spatial import bbox_polygon_overlap_ratio


def main():
    print("=" * 80)
    print("AUDIT DIAGNOSTIK 1-FRAME EMPIRIS: BOUNDING BOX & POLIGON CAM_01 (S1, S2, S3)")
    print("=" * 80)

    # 1. Temukan frame snapshot terbaru dari cam_01
    snap_dir = Path("cameras/cam_01/snapshots")
    snaps = sorted(snap_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snaps:
        print("ERROR: Tidak ada snapshot frame di cameras/cam_01/snapshots!")
        return

    frame_path = snaps[0]
    print(f"[1] Membaca snapshot terbaru: {frame_path.name} (Ukuran: {frame_path.stat().st_size} bytes)")
    raw_frame = cv2.imread(str(frame_path))
    if raw_frame is None:
        print(f"ERROR: cv2.imread gagal membaca {frame_path}")
        return

    raw_h, raw_w = raw_frame.shape[:2]
    print(f"    Dimensi Raw Frame Snapshot: {raw_w} x {raw_h}")

    ai_w, ai_h = 640, 360
    if raw_w != ai_w or raw_h != ai_h:
        ai_frame = cv2.resize(raw_frame, (ai_w, ai_h))
    else:
        ai_frame = raw_frame

    # Poligon di roi_zones.json berbasis 1920x1080 (Base Canvas)
    base_w, base_h = 1920.0, 1080.0
    poly_scale_x = ai_w / base_w  # 1/3
    poly_scale_y = ai_h / base_h  # 1/3
    print(f"    Skala Transformasi Poligon (1080p -> 360p): sx={poly_scale_x:.4f}, sy={poly_scale_y:.4f}")

    # 2. Muat ROI zones cam_01
    roi_path = Path("cameras/cam_01/roi_zones.json")
    with open(roi_path, "r", encoding="utf-8") as f:
        roi_data = json.load(f)

    polygons_dict = {p["zone_id"]: p for p in roi_data.get("polygons", [])}

    zones_info = {}
    print("\n[2] Geometri Poligon Target pada AI Canvas (640x360):")
    for zid in ["zone_01", "zone_02", "zone_03"]:
        poly_cfg = polygons_dict[zid]
        pts_raw = [(pt["x"], pt["y"]) for pt in poly_cfg["points"]]
        pts_ai = np.array(
            [[round(pt["x"] * poly_scale_x, 2), round(pt["y"] * poly_scale_y, 2)] for pt in poly_cfg["points"]],
            dtype=np.float32,
        )
        pts_ai_int = np.array(np.round(pts_ai), dtype=np.int32)
        cx = float(np.mean(pts_ai[:, 0]))
        cy = float(np.mean(pts_ai[:, 1]))
        min_x, max_x = float(np.min(pts_ai[:, 0])), float(np.max(pts_ai[:, 0]))
        min_y, max_y = float(np.min(pts_ai[:, 1])), float(np.max(pts_ai[:, 1]))

        zones_info[zid] = {
            "label": poly_cfg.get("label", zid),
            "pts_ai": pts_ai,
            "pts_ai_int": pts_ai_int,
            "centroid": (cx, cy),
            "span_x": (min_x, max_x),
            "span_y": (min_y, max_y),
        }
        print(f"    [{zid}] {poly_cfg.get('label', zid)}:")
        print(f"      Points AI: {pts_ai.tolist()}")
        print(f"      Centroid : ({cx:.1f}, {cy:.1f}) | Span X: [{min_x:.1f}, {max_x:.1f}] | Span Y: [{min_y:.1f}, {max_y:.1f}] | Max Y (Tapak Dasar): {max_y:.1f}")

    # 3. Ekstrak seluruh deteksi mentah YOLO (conf >= 0.15)
    model_path = Path("weights/yolo11n.onnx")
    detector = YOLO11nDetector(
        confidence_threshold=0.15,
        iou_threshold=0.45,
        target_classes=["car", "truck", "bus"],
        model_path=model_path,
    )

    detections = detector.detect(ai_frame)
    print(f"\n[3] Ekstraksi Deteksi YOLO Mentah (conf >= 0.15): Ditemukan total {len(detections)} deteksi di frame")

    # Evaluasi deteksi
    analyzed_dets = []
    for idx, d in enumerate(detections):
        x1, y1, x2, y2 = d.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        ar = bw / bh
        bottom_center = (cx, float(y2))

        # 5-point stance probes
        probes = {
            "center": (cx, float(y2)),
            "inset": (cx, float(y2 - bh * 0.05)),
            "left": (float(x1 + bw * 0.22), float(y2 - bh * 0.04)),
            "right": (float(x2 - bw * 0.22), float(y2 - bh * 0.04)),
            "axle": (cx, float(y2 - bh * 0.12)),
        }

        # Evaluasi terhadap S1, S2, S3
        eval_zones = {}
        for zid, zdata in zones_info.items():
            zpts = zdata["pts_ai_int"]
            zcx, zcy = zdata["centroid"]

            d_probes = {k: float(cv2.pointPolygonTest(zpts, pt, True)) for k, pt in probes.items()}
            d_ground = max(d_probes.values())

            overlap_ratio = bbox_polygon_overlap_ratio((x1, y1, x2, y2), zpts)
            dist_to_centroid = math.hypot(cx - zcx, cy - zcy)

            eval_zones[zid] = {
                "d_probes": d_probes,
                "d_ground": d_ground,
                "overlap_pct": overlap_ratio * 100.0,
                "dist_centroid": dist_to_centroid,
            }

        # Klasifikasi target fisik sebenarnya
        if cx <= 100 and y2 <= 210:
            target_phys = "Mobil Silver S1 / Aspal S1"
        elif 80 <= cx <= 165 and y2 <= 210:
            target_phys = "Mobil S2 (Putih/Silver)"
        elif 165 < cx <= 240 and y2 <= 210:
            target_phys = "Slot S3 (Area Mobil/Aspal S3)"
        elif y2 >= 220:
            target_phys = "Kendaraan Koridor Depan"
        elif cx > 240 and y2 <= 210:
            target_phys = "Slot S4 s.d. S8"
        else:
            target_phys = "Objek Sekitar"

        analyzed_dets.append({
            "idx": idx + 1,
            "class_label": d.class_label,
            "confidence": d.confidence,
            "bbox": [round(c, 1) for c in d.bbox],
            "bw": round(bw, 1),
            "bh": round(bh, 1),
            "ar": round(ar, 2),
            "cx": round(cx, 1),
            "cy": round(cy, 1),
            "bottom_center": (round(cx, 1), round(y2, 1)),
            "probes": probes,
            "eval_zones": eval_zones,
            "target_phys": target_phys,
        })

    # Urutkan berdasarkan x1 (dari kiri ke kanan)
    analyzed_dets.sort(key=lambda item: item["bbox"][0])

    print("\n" + "=" * 145)
    print("TABEL ANALISIS EMPIRIS DETEKSI MENTAH YOLO (cam_01 AI Canvas 640x360)")
    print("=" * 145)
    header = (
        f"{'ID':<4} | {'Class':<5} | {'Conf':<5} | {'BBox [x1,y1,x2,y2]':<21} | "
        f"{'bw':<5} | {'bh':<5} | {'bw/bh':<5} | {'Bottom_Ctr':<13} | "
        f"{'Ovl S1':<7} | {'Ovl S2':<7} | {'Ovl S3':<7} | {'d_gnd S2':<8} | {'d_gnd S3':<8} | {'Target Fisik Sebenarnya'}"
    )
    print(header)
    print("-" * 145)

    for item in analyzed_dets:
        b = item["bbox"]
        b_str = f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"
        bc = item["bottom_center"]
        bc_str = f"({bc[0]:.0f},{bc[1]:.0f})"

        ovl_s1 = item["eval_zones"]["zone_01"]["overlap_pct"]
        ovl_s2 = item["eval_zones"]["zone_02"]["overlap_pct"]
        ovl_s3 = item["eval_zones"]["zone_03"]["overlap_pct"]

        dg_s2 = item["eval_zones"]["zone_02"]["d_ground"]
        dg_s3 = item["eval_zones"]["zone_03"]["d_ground"]

        line = (
            f"#{item['idx']:<3} | {item['class_label']:<5} | {item['confidence']:<5.2f} | {b_str:<21} | "
            f"{item['bw']:<5.1f} | {item['bh']:<5.1f} | {item['ar']:<5.2f} | {bc_str:<13} | "
            f"{ovl_s1:<6.1f}% | {ovl_s2:<6.1f}% | {ovl_s3:<6.1f}% | {dg_s2:<8.1f} | {dg_s3:<8.1f} | {item['target_phys']}"
        )
        print(line)

    print("-" * 145)

    # Analisis mendalam khusus untuk deteksi di area S1..S3
    print("\n[4] DETAIL EVALUASI GEOMETRIS STANCE PROBE & AFINITAS (DETEKSI AREA S1 - S3):")
    for item in analyzed_dets:
        if item["cx"] <= 260 and item["bbox"][3] <= 220:
            print(f"\n==========================================================================")
            print(f"DETEKSI #{item['idx']}: {item['class_label'].upper()} (Conf={item['confidence']:.2f})")
            print(f"  BBox: {item['bbox']} | Dimensi: bw={item['bw']}, bh={item['bh']}, bw/bh={item['ar']}")
            print(f"  Bottom-Center (Tapak): {item['bottom_center']}")
            print(f"  Target Fisik: {item['target_phys']}")
            print("--------------------------------------------------------------------------")
            for zid in ["zone_01", "zone_02", "zone_03"]:
                ez = item["eval_zones"][zid]
                z_lbl = zones_info[zid]["label"]
                z_cx, z_cy = zones_info[zid]["centroid"]
                z_sy = zones_info[zid]["span_y"]
                print(f"  * Zona {zid} ({z_lbl}) [Centroid: ({z_cx:.1f}, {z_cy:.1f}), Tapak Bawah max_y={z_sy[1]:.1f}]:")
                print(f"      Overlap BBox terhadap Poligon: {ez['overlap_pct']:.1f}%")
                print(f"      Jarak Centroid BBox ke Centroid Zona: {ez['dist_centroid']:.1f} px")
                print(f"      d_ground (Max Stance Probe): {ez['d_ground']:+.2f} px")
                p_strs = [f"{k}={v:+.1f}" for k, v in ez["d_probes"].items()]
                print(f"      Detail 5 Probes: {', '.join(p_strs)}")


if __name__ == "__main__":
    main()
