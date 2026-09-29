"""
tools/debug_s4_detection.py
===========================
Skrip diagnostik offline 1-frame untuk audit deteksi YOLO mentah (raw proposals)
dan gating spasial pada Slot S4 (zone_04) di live feed cam_01.
"""

from __future__ import annotations

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from engine.config_loader import load_roi_zones, ROIPoint, ROIZone
from engine.detector_impl import COCO_CLASSES, YOLO11nDetector
from engine.smart_parking import (
    bbox_iou,
    bbox_polygon_overlap_ratio,
    SmartParkingTracker,
)


def capture_frame() -> np.ndarray:
    raw_path = Path("storage/cam_01_raw.jpg")
    rtsp_url = "rtsp://parkir:Bandung*@192.212.160.210:554/Streaming/Channels/102"
    
    print("[1] Mencoba capture live frame dari RTSP cam_01...")
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        print(f"  ✓ Berhasil capture live frame dari RTSP: shape={frame.shape}")
        cv2.imwrite(str(raw_path), frame)
        return frame
    
    if raw_path.exists():
        print(f"  [i] Menggunakan cached frame dari {raw_path}")
        frame = cv2.imread(str(raw_path))
        if frame is not None:
            return frame
            
    raise RuntimeError("Gagal mengambil frame dari RTSP maupun storage/cam_01_raw.jpg")


def extract_raw_yolo_proposals(
    frame: np.ndarray,
    model_path: str = "weights/yolo11n.onnx",
    conf_min: float = 0.05,
) -> list[dict]:
    """Ekstraksi deteksi mentah langsung dari output tensor ONNX tanpa filter threshold tinggi."""
    print(f"\n[2] Menjalankan inferensi YOLO mentah (conf >= {conf_min})...")
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    h_img, w_img = frame.shape[:2]
    scale = min(640.0 / w_img, 640.0 / h_img)
    new_w = int(round(w_img * scale))
    new_h = int(round(h_img * scale))
    pad_x = (640 - new_w) // 2
    pad_y = (640 - new_h) // 2

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    blob = cv2.dnn.blobFromImage(
        canvas,
        scalefactor=1.0 / 255.0,
        size=(640, 640),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    output = net.forward()
    if isinstance(output, (tuple, list)):
        output = output[0]
    if len(output.shape) == 3:
        output = output[0]
    if output.shape[0] == 84:
        output = output.T

    raw_proposals = []
    for row in output:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])
        if conf < conf_min:
            continue

        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"cls_{class_id}"
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        if cx <= 1.0 and cy <= 1.0:
            cx *= 640.0
            cy *= 640.0
            w *= 640.0
            h *= 640.0

        cx = (cx - pad_x) / scale
        cy = (cy - pad_y) / scale
        w = w / scale
        h = h / scale

        x1 = max(0.0, cx - w / 2.0)
        y1 = max(0.0, cy - h / 2.0)
        x2 = min(float(w_img), cx + w / 2.0)
        y2 = min(float(h_img), cy + h / 2.0)

        raw_proposals.append({
            "class_id": class_id,
            "class_label": label,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "raw_box": [int(x1), int(y1), int(w), int(h)],
        })

    # NMS per class / global
    boxes = [p["raw_box"] for p in raw_proposals]
    confs = [p["confidence"] for p in raw_proposals]
    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_min, 0.45)
    
    nms_proposals = []
    if len(indices) > 0:
        flat = indices.flatten() if isinstance(indices, np.ndarray) else [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
        for idx in flat:
            nms_proposals.append(raw_proposals[idx])

    print(f"  ✓ Total proposal mentah: {len(raw_proposals)} | Setelah NMS (iou=0.45): {len(nms_proposals)}")
    return nms_proposals


def audit_s4_zone(frame: np.ndarray, proposals: list[dict]) -> None:
    print("\n[3] Mengaudit Geometri Poligon S4 (zone_04)...")
    roi_cfg = load_roi_zones(Path("cameras/cam_01/roi_zones.json"))
    s4_poly = next((p for p in roi_cfg.polygons if p.zone_id == "zone_04"), None)
    if s4_poly is None:
        raise ValueError("zone_04 tidak ditemukan di roi_zones.json")

    # Frame AI adalah 640x360 (scale = 3.0 dari 1920x1080)
    sx = 640.0 / 1920.0
    sy = 360.0 / 1080.0
    pts_scaled = np.array(
        [[int(round(pt.x * sx)), int(round(pt.y * sy))] for pt in s4_poly.points],
        dtype=np.int32,
    )
    rx, ry, rw, rh = cv2.boundingRect(pts_scaled)
    s4_bbox = (float(rx), float(ry), float(rx + rw), float(ry + rh))

    print(f"  Poligon S4 (1080p): {[(p.x, p.y) for p in s4_poly.points]}")
    print(f"  Poligon S4 (640x360): {pts_scaled.tolist()}")
    print(f"  Bounding Rect S4 (640x360): {s4_bbox}")

    # Temukan semua proposal yang beririsan dengan bounding rect S4
    candidates_s4 = []
    for p in proposals:
        bx1, by1, bx2, by2 = p["bbox"]
        overlap = bbox_polygon_overlap_ratio((bx1, by1, bx2, by2), pts_scaled)
        iou_slot = bbox_iou((bx1, by1, bx2, by2), s4_bbox)
        
        # Cek jika bbox berada di sekitar area S4
        in_vicinity = not (bx2 < rx - 30 or bx1 > rx + rw + 30 or by2 < ry - 30 or by1 > ry + rh + 30)
        if in_vicinity or overlap > 0.05 or iou_slot > 0.05:
            candidates_s4.append((p, overlap, iou_slot))

    print(f"\n[4] Proposal YOLO di sekitar Slot S4 (zone_04): {len(candidates_s4)} terdeteksi:")
    print("=" * 80)
    print(f"{'Class':<12} | {'Conf':<6} | {'BBox (640p)':<26} | {'Lower-IoZ':<10} | {'Slot-IoU':<8} | {'Status Pipeline'}")
    print("-" * 80)

    for p, lower_overlap, iou_slot in candidates_s4:
        bx1, by1, bx2, by2 = p["bbox"]
        cx = float((bx1 + bx2) / 2.0)
        bh = max(1.0, by2 - by1)
        bw = max(1.0, bx2 - bx1)

        p_center = (cx, float(by2))
        p_inset = (cx, float(by2 - bh * 0.05))
        p_left = (float(bx1 + bw * 0.22), float(by2 - bh * 0.04))
        p_right = (float(bx2 - bw * 0.22), float(by2 - bh * 0.04))
        p_axle = (cx, float(by2 - bh * 0.12))

        d_center = cv2.pointPolygonTest(pts_scaled, p_center, True)
        d_inset = cv2.pointPolygonTest(pts_scaled, p_inset, True)
        d_left = cv2.pointPolygonTest(pts_scaled, p_left, True)
        d_right = cv2.pointPolygonTest(pts_scaled, p_right, True)
        d_axle = cv2.pointPolygonTest(pts_scaled, p_axle, True)
        d_ground = max(d_center, d_inset, d_left, d_right, d_axle)

        lower_bbox = (bx1, by1 + bh * 0.5, bx2, by2)
        actual_lower_overlap = bbox_polygon_overlap_ratio(lower_bbox, pts_scaled)

        # Evaluasi gating
        det_threshold = 0.20  # detector pipeline threshold
        pass_det = p["confidence"] >= det_threshold
        eff_margin = 8.0
        has_stance = (
            (d_ground >= -eff_margin)
            or (actual_lower_overlap >= 0.15)
            or (d_ground >= -12.0 and actual_lower_overlap >= 0.10)
        )
        pass_acq = p["confidence"] >= 0.20

        status_str = "PASS DETECTOR & GATE" if (pass_det and has_stance and pass_acq) else (
            "DROPPED AT DETECTOR" if not pass_det else ("FAILED STANCE GATE" if not has_stance else "FAILED ACQ")
        )

        bbox_str = f"[{bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}]"
        print(f"{p['class_label']:<12} | {p['confidence']:<6.3f} | {bbox_str:<26} | {actual_lower_overlap:<10.3f} | {iou_slot:<8.3f} | {status_str}")
        print(f"   -> Probes: d_center={d_center:.1f}, d_inset={d_inset:.1f}, d_left={d_left:.1f}, d_right={d_right:.1f}, d_ground={d_ground:.1f}")

    print("=" * 80)

    # Simpan annotated visual frame
    vis_frame = frame.copy()
    cv2.polylines(vis_frame, [pts_scaled], True, (0, 255, 255), 2)
    cv2.putText(vis_frame, "S4 (zone_04)", (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for p, lower_overlap, iou_slot in candidates_s4:
        bx1, by1, bx2, by2 = [int(v) for v in p["bbox"]]
        color = (0, 255, 0) if p["confidence"] >= 0.20 else (0, 0, 255)
        cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), color, 1)
        lbl = f"{p['class_label']} {p['confidence']:.2f}"
        cv2.putText(vis_frame, lbl, (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out_path = Path("storage/debug_s4_annotated.jpg")
    cv2.imwrite(str(out_path), vis_frame)
    print(f"\n[5] Frame diagnostik visual disimpan ke: {out_path.resolve()}")


if __name__ == "__main__":
    frame = capture_frame()
    proposals = extract_raw_yolo_proposals(frame, conf_min=0.05)
    audit_s4_zone(frame, proposals)
