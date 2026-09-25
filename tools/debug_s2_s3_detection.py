import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from engine.detector_impl import YOLO11nDetector
from engine.geometry import bbox_bottom_center, bbox_iou, scale_points

def main():
    print("==================================================================")
    print("DEBUG DIAGNOSTIC: INSPEKSI RAW YOLO DETECTIONS DI SEKITAR S2/S3")
    print("==================================================================")

    # 1. Temukan frame snapshot terkini
    snap_dir = Path("cameras/cam_01/snapshots")
    snaps = sorted(snap_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if snaps:
        snap_path = snaps[0]
        print(f"Menggunakan snapshot: {snap_path}")
    else:
        snap_path = Path("scratch/cam01_live.jpg")
        print(f"Snapshot tidak ditemukan, menggunakan live frame: {snap_path}")

    frame = cv2.imread(str(snap_path))
    if frame is None:
        print("ERROR: Gagal membaca gambar frame!")
        return

    h_raw, w_raw = frame.shape[:2]
    print(f"Resolusi raw frame: {w_raw}x{h_raw}")

    base_w, base_h = 1920, 1080
    ai_w, ai_h = 640, 360
    ai_frame = cv2.resize(frame, (ai_w, ai_h))
    scale_x = ai_w / base_w
    scale_y = ai_h / base_h
    print(f"Canvas Basis: {base_w}x{base_h} -> AI: {ai_w}x{ai_h} (scale_x={scale_x:.4f}, scale_y={scale_y:.4f})")

    # 2. Muat ROI zones cam_01
    roi_path = Path("cameras/cam_01/roi_zones.json")
    with open(roi_path, "r", encoding="utf-8") as f:
        roi_data = json.load(f)

    zones = {z["zone_id"]: z for z in roi_data.get("polygons", [])}

    print("\n--- KOORDINAT POLIGON S1, S2, S3, S4 (RAW & AI) ---")
    for zid in ["zone_01", "zone_02", "zone_03", "zone_04"]:
        if zid not in zones:
            continue
        z = zones[zid]
        pts_raw = [(pt["x"], pt["y"]) for pt in z["points"]]
        pts_ai = [(round(pt["x"] * scale_x, 1), round(pt["y"] * scale_y, 1)) for pt in z["points"]]
        cx_raw = np.mean([p[0] for p in pts_raw])
        cy_raw = np.mean([p[1] for p in pts_raw])
        cx_ai = np.mean([p[0] for p in pts_ai])
        cy_ai = np.mean([p[1] for p in pts_ai])
        print(f"[{zid}] ({z.get('label', zid)}):")
        print(f"  Raw: {pts_raw} | Centroid: ({cx_raw:.1f}, {cy_raw:.1f})")
        print(f"  AI : {pts_ai} | Centroid: ({cx_ai:.1f}, {cy_ai:.1f})")

    # 3. Direct inspection of raw ONNX network outputs
    model_path = Path("weights/yolo11n.onnx")
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    h_img, w_img = ai_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        ai_frame,
        scalefactor=1.0 / 255.0,
        size=(640, 640),
        swapRB=True,
        crop=False
    )
    net.setInput(blob)
    outputs = net.forward()
    if isinstance(outputs, (tuple, list)):
        output = outputs[0]
    else:
        output = outputs
    if len(output.shape) == 3:
        output = output[0]
    if output.shape[0] == 84:
        output = output.T

    from engine.detector_impl import COCO_CLASSES

    print("\n--- ANALISIS SEMUA PROPOSAL BBOX DI SLOT 3 (x_ai: 140..230, y_ai: 100..210) ---")
    x_scale = w_img / 640.0
    y_scale = h_img / 640.0

    slot3_proposals = []
    for row in output:
        cx, cy, w, h = row[0] * x_scale, row[1] * y_scale, row[2] * x_scale, row[3] * y_scale
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(w_img, int(cx + w / 2))
        y2 = min(h_img, int(cy + h / 2))

        # Check if proposal center is in Slot 3 area
        if 130 <= cx <= 240 and 110 <= cy <= 210:
            scores = row[4:]
            max_cls_id = int(np.argmax(scores))
            max_conf = float(scores[max_cls_id])
            car_conf = float(scores[2]) # Class 2 = car
            truck_conf = float(scores[7]) # Class 7 = truck
            bus_conf = float(scores[5]) # Class 5 = bus
            pm_conf = float(scores[12]) # Class 12 = parking meter

            if max_conf >= 0.15 or car_conf >= 0.10:
                slot3_proposals.append({
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "max_cls": COCO_CLASSES[max_cls_id],
                    "max_conf": max_conf,
                    "car_conf": car_conf,
                    "truck_conf": truck_conf,
                    "bus_conf": bus_conf,
                    "pm_conf": pm_conf,
                })

    # Sort by car_conf descending
    slot3_proposals.sort(key=lambda p: p["max_conf"], reverse=True)
    print(f"Ditemukan {len(slot3_proposals)} proposal di area Slot 3:")
    for i, p in enumerate(slot3_proposals[:15]):
        print(f"  [{i+1}] Bbox={p['bbox']} | Top={p['max_cls']} ({p['max_conf']:.3f}) | Car={p['car_conf']:.3f} | Truck={p['truck_conf']:.3f} | ParkingMeter={p['pm_conf']:.3f}")

    print("\n--- PRODUKSI YOLO11n DETECTOR (conf=0.20, target_classes=['car', 'truck', 'bus']) ---")
    detector = YOLO11nDetector(
        confidence_threshold=0.20,
        iou_threshold=0.45,
        target_classes=["car", "truck", "bus"],
        model_path=model_path,
    )
    detections = detector.detect(ai_frame)
    print(f"\n--- TOTAL DETEKSI RAW YOLO (conf >= 0.10): {len(detections)} ---")

    # Filter deteksi di area x_ai antara 50 dan 350 (mencakup S1, S2, S3, S4)
    # y_ai sekitar 120 - 250
    candidates_s2_s3 = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # Konversi ke raw
        raw_x1 = x1 / scale_x
        raw_y1 = y1 / scale_y
        raw_x2 = x2 / scale_x
        raw_y2 = y2 / scale_y

        print(f"Deteksi #{d.class_id} ({d.class_label}): conf={d.confidence:.3f} | AI Bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] | RAW Bbox=[{raw_x1:.0f}, {raw_y1:.0f}, {raw_x2:.0f}, {raw_y2:.0f}]")
        candidates_s2_s3.append(d)

    # 4. Evaluasi Spasial terhadap zone_02 dan zone_03
    print("\n--- EVALUASI SPASIAL TERHADAP ZONE_02 & ZONE_03 ---")
    for zid in ["zone_02", "zone_03"]:
        z = zones[zid]
        pts_ai_np = np.array(
            [[pt["x"] * scale_x, pt["y"] * scale_y] for pt in z["points"]],
            dtype=np.float32,
        )
        xs = [pt[0] for pt in pts_ai_np]
        ys = [pt[1] for pt in pts_ai_np]
        slot_bbox = (min(xs), min(ys), max(xs), max(ys))
        slot_cx = (min(xs) + max(xs)) / 2.0
        slot_cy = (min(ys) + max(ys)) / 2.0
        slot_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0

        print(f"\nAnalisis untuk {zid} ({z.get('label')}):")
        for d in detections:
            if d.class_label not in ["car", "truck", "bus", "parking meter"]:
                continue
            x1, y1, x2, y2 = d.bbox
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            wheel_pt = (float(cx), float(y2 - h * 0.10))
            lower_pt = (float(cx), float(y2 - h * 0.25))
            center_pt = (float(cx), float(cy))

            d_wheel = cv2.pointPolygonTest(pts_ai_np, wheel_pt, True)
            d_lower = cv2.pointPolygonTest(pts_ai_np, lower_pt, True)
            d_center = cv2.pointPolygonTest(pts_ai_np, center_pt, True)
            d_max = max(d_wheel, d_lower, d_center)

            iou_slot = bbox_iou(d.bbox, slot_bbox)
            dist_to_center = math.hypot(cx - slot_cx, cy - slot_cy)
            dist_norm = dist_to_center / slot_diag

            # Kriteria kandidat SmartParkingTracker:
            # d_max >= -3.0 or iou_anchor >= 0.25
            is_valid_candidate = (d_max >= -3.0)

            score = (d_max * 2.5) - (dist_norm * 30.0) + (iou_slot * 35.0)

            print(f"  -> Kendaraan conf={d.confidence:.2f} bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]:")
            print(f"     d_wheel={d_wheel:.2f}, d_lower={d_lower:.2f}, d_center={d_center:.2f} => d_max={d_max:.2f}")
            print(f"     iou_slot={iou_slot:.3f}, dist_norm={dist_norm:.3f}")
            print(f"     KANDIDAT VALID (d_max >= -3.0)? {is_valid_candidate} | Score={score:.2f}")

    # 5. Simulasi SmartParkingTracker dengan deteksi produksi
    print("\n--- SIMULASI SMART PARKING TRACKER DENGAN DETEKSI TERAKHIR ---")
    from engine.smart_parking import SmartParkingTracker
    from engine.tracker_interface import TrackResult
    from engine.config_loader import ROIZonesConfig

    with open(roi_path, "r", encoding="utf-8") as f:
        roi_cfg = ROIZonesConfig(**json.load(f))

    tracker = SmartParkingTracker(dwell_threshold_sec=5.0)

    # Convert detections to TrackResult
    tracks = [
        TrackResult(
            track_id=idx + 1,
            bbox=d.bbox,
            class_id=d.class_id,
            class_label=d.class_label,
            confidence=d.confidence,
        )
        for idx, d in enumerate(detections)
    ]

    # Warmup update (seperti cold-start kamera)
    t = 1000.0
    res = tracker.update(tracks, roi_cfg.polygons, scale_x, scale_y, t, is_warmup=True)

    print("\nHasil Status Okupansi Tiap Slot (Cold-Start Evaluation):")
    for s_id in sorted(tracker.slot_states.keys()):
        st = tracker.slot_states[s_id]
        print(f"  [{s_id}] {st.label}: {st.phase} | track_id={st.track_id} | dwell={st.dwell_duration:.1f}s | latched={st.latch_occupied}")

    print(f"\nTotal Terisi (OCCUPIED): {res['occupied_slots']} / {res['total_slots']}")
    print(f"Total Kosong (AVAILABLE): {res['available_slots']} / {res['total_slots']}")

if __name__ == "__main__":
    main()
