"""
Smoke test untuk YOLO11nDetector via OpenCV DNN module.
"""

import time
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.detector_impl import YOLO11nDetector


def smoke_test_detector():
    print("[SMOKE TEST] Menguji YOLO11nDetector dengan OpenCV DNN...")

    detector = YOLO11nDetector(
        confidence_threshold=0.25,
        iou_threshold=0.45,
        target_classes=["person", "car", "motorcycle", "bus", "truck", "backpack", "handbag"],
        model_path="weights/yolo11n.onnx",
    )

    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    t0 = time.time()
    detector.warmup((360, 640, 3))
    warmup_dt = (time.time() - t0) * 1000.0
    print(f"  Warmup selesai dalam {warmup_dt:.1f} ms")

    t0 = time.time()
    results = detector.detect(dummy_frame)
    infer_dt = (time.time() - t0) * 1000.0
    print(f"  Inferensi dummy frame selesai dalam {infer_dt:.1f} ms (Hasil: {len(results)} deteksi)")

    print("  ✓ PASS: YOLO11nDetector smoke test sukses.")


if __name__ == "__main__":
    smoke_test_detector()
