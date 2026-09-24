"""
Script pengunduh otomatis model YOLO ONNX untuk Facility Management Vision Engine.
Mengunduh yolov8n.onnx/yolo11n.onnx yang kompatibel dengan OpenCV 4.10.0 DNN module.
"""

import sys
import urllib.request
from pathlib import Path
import cv2

V8_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "weights"
MODEL_FILE = OUTPUT_DIR / "yolo11n.onnx"
V8_MODEL_FILE = OUTPUT_DIR / "yolov8n.onnx"


def download_model(force: bool = False) -> Path:
    """Unduh file model ONNX jika belum ada atau jika force=True."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not V8_MODEL_FILE.exists() or force:
        print(f"[DOWNLOAD] Mengunduh model YOLO ONNX dari {V8_URL}...")

        def _progress(count, block_size, total_size):
            percent = count * block_size * 100 / total_size if total_size > 0 else 0
            sys.stdout.write(f"\r progress: {percent:.1f}% ({count * block_size / 1024:.0f} KB)")
            sys.stdout.flush()

        try:
            urllib.request.urlretrieve(V8_URL, V8_MODEL_FILE, reporthook=_progress)
            print(f"\n[DOWNLOAD] Sukses! Model tersimpan di {V8_MODEL_FILE} ({V8_MODEL_FILE.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            if V8_MODEL_FILE.exists():
                V8_MODEL_FILE.unlink()
            print(f"\n[ERROR] Gagal mengunduh model: {e}")
            sys.exit(1)

    # Verifikasi loading via OpenCV DNN
    try:
        net = cv2.dnn.readNetFromONNX(str(V8_MODEL_FILE))
        print(f"[MODEL CHECK] OpenCV DNN berhasil memuat {V8_MODEL_FILE.name} ({len(net.getLayerNames())} layer).")
    except Exception as e:
        print(f"[ERROR] Model ONNX gagal dibaca oleh OpenCV DNN: {e}")
        sys.exit(1)

    # Copy / alias ke yolo11n.onnx agar kompatibel dengan settings.json
    if not MODEL_FILE.exists() or force:
        import shutil
        shutil.copy(V8_MODEL_FILE, MODEL_FILE)
        print(f"[MODEL ALIAS] Disalin ke {MODEL_FILE}")

    return MODEL_FILE


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    download_model(force=force_download)
