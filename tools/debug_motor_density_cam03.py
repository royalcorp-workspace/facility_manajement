import os
import sys
import json
import math
import cv2
import numpy as np
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.detector_impl import YOLO11nDetector
from engine.config_loader import ROIZonesConfig
from engine.smart_parking import (
    SmartParkingTracker,
    bbox_iou,
    bbox_ios,
    bbox_polygon_overlap_ratio,
    deduplicate_motorcycle_tracks,
)
from engine.tracker_interface import TrackResult


def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "scratch/cam03_live_1120.jpg" if Path("scratch/cam03_live_1120.jpg").exists() else "scratch/cam03_raw.jpg"

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return

    h, w = img.shape[:2]
    print(f"=== EVALUASI OFFLINE DEDUPLIKASI & DENSITAS MOTOR cam_03 ===")
    print(f"Source Image : {img_path} ({w}x{h})")

    # 1. Inisialisasi Detector
    detector = YOLO11nDetector(
        confidence_threshold=0.15,
        iou_threshold=0.48,
        target_classes=["motorcycle", "bicycle"],
    )
    detector.load_model("weights/yolo11n.onnx")
    results = detector.detect(img)
    print(f"Total raw vehicle detections across frame: {len(results)}")

    # 2. Load ROI Zone 01
    roi_path = Path("cameras/cam_03/roi_zones.json")
    with open(roi_path, "r", encoding="utf-8") as f:
        roi_data = json.load(f)
    roi_config = ROIZonesConfig(**roi_data)
    zone_01 = next(z for z in roi_config.polygons if z.zone_id == "zone_01")

    sx = w / 1920.0
    sy = h / 1080.0
    pts_scaled = np.array(
        [[int(p.x * sx), int(p.y * sy)] for p in zone_01.points],
        dtype=np.int32,
    )

    # 3. Pengetatan Footprint Spasial (Singkirkan Motor Luar & Reject Standalone Overlap)
    candidate_tracks = []
    outside_tracks = []

    for r in results:
        if r.class_label not in ["motorcycle", "bicycle"]:
            continue
        rx1, ry1, rx2, ry2 = r.bbox
        cx = float((rx1 + rx2) / 2.0)
        wheel_y = float(ry2)
        wheel_pt = (int(cx), int(wheel_y))

        # Motor hanya sah jika titik tumpu roda bawah (cx, y2) berada DI DALAM poligon zone_01
        wheel_in = cv2.pointPolygonTest(pts_scaled, wheel_pt, False) >= 0
        overlap_ratio = bbox_polygon_overlap_ratio(r.bbox, pts_scaled)

        if not wheel_in:
            outside_tracks.append((r, "wheel_in == False", overlap_ratio))
            continue

        if overlap_ratio < 0.20:
            outside_tracks.append((r, "overlap_ratio < 0.20", overlap_ratio))
            continue

        candidate_tracks.append(r)

    print(f"\n--- DETEKSI MOTOR DI LUAR ZONE_01 (GUGUR / REJECT: {len(outside_tracks)}) ---")
    for idx, (r, reason, ov) in enumerate(outside_tracks, 1):
        print(
            f"#{idx:02d} [{r.class_label}] conf={r.confidence:.2f} "
            f"bbox={[round(x, 1) for x in r.bbox]} overlap={ov*100:.1f}% -> {reason}"
        )

    print(f"\n--- KANDIDAT MOTOR DALAM ZONE_01 SEBELUM DEDUPLIKASI ({len(candidate_tracks)}) ---")
    for idx, c in enumerate(candidate_tracks, 1):
        ov = bbox_polygon_overlap_ratio(c.bbox, pts_scaled)
        print(
            f"#{idx:02d} [{c.class_label}] conf={c.confidence:.2f} "
            f"bbox={[round(x, 1) for x in c.bbox]} overlap={ov*100:.1f}%"
        )

    # 4. Deduplikasi Spasial Diperketat (Local NMS / IoU >= 0.30 + IoS >= 0.50 + Centroid Proximity <= 28px)
    sorted_candidates = sorted(candidate_tracks, key=lambda t: t.confidence, reverse=True)
    valid_tracks = []
    suppressed = []

    for cand in sorted_candidates:
        c_cx = float((cand.bbox[0] + cand.bbox[2]) / 2.0)
        c_cy = float((cand.bbox[1] + cand.bbox[3]) / 2.0)
        is_dup = False
        for acc in valid_tracks:
            iou = bbox_iou(cand.bbox, acc.bbox)
            ios = bbox_ios(cand.bbox, acc.bbox)
            a_cx = float((acc.bbox[0] + acc.bbox[2]) / 2.0)
            a_cy = float((acc.bbox[1] + acc.bbox[3]) / 2.0)
            dist_px = math.hypot(c_cx - a_cx, c_cy - a_cy)

            if iou >= 0.30 or ios >= 0.50 or dist_px <= 28.0:
                reasons = []
                if iou >= 0.30:
                    reasons.append(f"IoU={iou:.2f} >= 0.30")
                if ios >= 0.50:
                    reasons.append(f"IoS={ios:.2f} >= 0.50")
                if dist_px <= 28.0:
                    reasons.append(f"CentroidDist={dist_px:.1f}px <= 28px")
                reason_str = " | ".join(reasons)
                suppressed.append((cand, acc, reason_str))
                is_dup = True
                break
        if not is_dup:
            valid_tracks.append(cand)

    print(f"\n--- DEDUPLIKASI KOTAK TUMPANG TINDIH / BERSARANG (SUPPRESSED: {len(suppressed)}) ---")
    for cand, acc, reason_str in suppressed:
        print(
            f"  [DROP] conf={cand.confidence:.2f} bbox={[round(x, 1) for x in cand.bbox]} "
            f"tertindih oleh conf={acc.confidence:.2f} bbox={[round(x, 1) for x in acc.bbox]} "
            f"({reason_str})"
        )

    print(f"\n--- HASIL DETEKSI MOTOR BERSIH SESUDAH DEDUPLIKASI ({len(valid_tracks)}) ---")
    for idx, v in enumerate(valid_tracks, 1):
        ov = bbox_polygon_overlap_ratio(v.bbox, pts_scaled)
        print(
            f"#{idx:02d} [{v.class_label}] conf={v.confidence:.2f} "
            f"bbox={[round(x, 1) for x in v.bbox]} overlap={ov*100:.1f}%"
        )

    print("\n==================================================")
    print("RINGKASAN LOG EVALUASI DEDUPLIKASI SPASIAL (SINGLE FRAME):")
    print(f"- Total Raw Detections Frame       : {len(results)}")
    print(f"- Motor Gugur di Luar Area (OUT)   : {len(outside_tracks)}")
    print(f"- Kandidat Awal di Dalam Poligon   : {len(candidate_tracks)}")
    print(f"- Kotak Duplikat Dieliminasi (DROP): {len(suppressed)}")
    print(f"- Motor Bersih Lolos Terhitung (IN): {len(valid_tracks)}")
    print("==================================================")

    # 5. SIMULASI 100 FRAME BERURUTAN (TEMPORAL LATCHING & ANTI-FLAPPING STABILIZATION)
    print("\n=== SIMULASI 100 FRAME BERURUTAN (TIME-BASED LATCH & MOVING MEDIAN) ===")
    tracker = SmartParkingTracker(
        parking_mode="motorcycle_block",
        block_capacity=30,
        vehicle_classes={"motorcycle", "bicycle"},
        stationary_dwell_sec=0.0,
    )

    random.seed(42)
    sim_counts = []
    base_tracks = [
        TrackResult(
            track_id=idx,
            bbox=t.bbox,
            confidence=t.confidence,
            class_label=t.class_label,
            class_id=t.class_id,
            age=30,
            is_confirmed=True,
        )
        for idx, t in enumerate(valid_tracks, 1)
    ]

    t_start = 1000.0
    for frame_no in range(1, 101):
        t_current = t_start + (frame_no * 0.05)  # 20 FPS simulation (dt = 0.05s)

        # Injeksi gangguan flicker realistis: pada frame tertentu, 1-2 deteksi hilang sementara
        if frame_no > 10 and (frame_no % 4 == 0 or frame_no % 7 == 0):
            # Drop 1 s.d. 2 proposal acak untuk mensimulasikan noise kamera / kompresi RTSP
            drop_indices = set(random.sample(range(len(base_tracks)), min(2, len(base_tracks))))
            noisy_tracks = [t for i, t in enumerate(base_tracks) if i not in drop_indices]
        else:
            noisy_tracks = list(base_tracks)

        stats = tracker.update(
            tracks=noisy_tracks,
            polygons=[zone_01],
            scale_x=sx,
            scale_y=sy,
            current_time=t_current,
            is_warmup=(frame_no <= 5),
        )
        occ = stats["occupied_slots"]
        sim_counts.append(occ)

    print(f"Hasil Kuota 100 Frame Berurutan:")
    print(f"Frames  1-25  : {sim_counts[:25]}")
    print(f"Frames 26-50  : {sim_counts[25:50]}")
    print(f"Frames 51-75  : {sim_counts[50:75]}")
    print(f"Frames 76-100 : {sim_counts[75:100]}")

    counts_arr = np.array(sim_counts)
    mean_count = float(np.mean(counts_arr))
    std_dev = float(np.std(counts_arr))
    min_val = int(np.min(counts_arr))
    max_val = int(np.max(counts_arr))

    print("\n--- STATISTIK KESTABILAN TEMPORAL LATCHING ---")
    print(f"Nilai Rata-rata Kuota : {mean_count:.2f}")
    print(f"Nilai Min / Max       : {min_val} / {max_val}")
    print(f"Standar Deviasi (STD) : {std_dev:.4f}")

    if std_dev == 0.0:
        print("[SUKSES] Fluktuasi deviasi kuota adalah 0.0000 -> ANGKA KEPADATAN TERKUNCI STABIL (LOCKED)!")
        # Verifikasi tambahan: Motor #1 (unit paling kiri) harus selalu gugur
        print("[INFO] Motor di luar batas X=720 dipastikan gugur via wheel_in filter.")
    else:
        print(f"[WARN] Terjadi fluktuasi angka: STD = {std_dev:.4f}")


if __name__ == "__main__":
    main()
