from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np

from engine.detector_interface import DetectorBase, DetectionResult

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class YOLO11nDetector(DetectorBase):
    def __init__(
        self,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        target_classes: Optional[list[str]] = None,
        model_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            target_classes=target_classes,
        )
        self.net: Optional[cv2.dnn.Net] = None
        if model_path:
            self.load_model(str(model_path))

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file tidak ditemukan: {path}")

        try:
            self.net = cv2.dnn.readNetFromONNX(str(path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model ONNX via OpenCV DNN: {e}") from e

    def warmup(self, ai_frame_shape: tuple[int, int, int] = (360, 640, 3)) -> None:
        if self.net is None:
            return
        dummy = np.zeros(ai_frame_shape, dtype=np.uint8)
        self.detect(dummy)

    def detect(self, ai_frame: np.ndarray) -> list[DetectionResult]:
        if self.net is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        h_img, w_img = ai_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            ai_frame,
            scalefactor=1.0 / 255.0,
            size=(640, 640),
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        if isinstance(outputs, (tuple, list)):
            output = outputs[0]
        else:
            output = outputs

        if len(output.shape) == 3:
            output = output[0] 

        if output.shape[0] == 84:
            output = output.T

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        class_labels: list[str] = []
        x_scale = w_img / 640.0
        y_scale = h_img / 640.0

        for row in output:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < self.confidence_threshold:
                continue

            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"cls_{class_id}"
            if not self.is_target_class(label):
                continue

            cx, cy, w, h = row[0], row[1], row[2], row[3]
            if cx <= 1.0 and cy <= 1.0:
                cx *= 640.0
                cy *= 640.0
                w *= 640.0
                h *= 640.0

            cx *= x_scale
            cy *= y_scale
            w *= x_scale
            h *= y_scale

            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            box_w = int(w)
            box_h = int(h)

            boxes.append([x1, y1, box_w, box_h])
            confidences.append(confidence)
            class_ids.append(class_id)
            class_labels.append(label)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.iou_threshold
        )

        results: list[DetectionResult] = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if isinstance(indices, np.ndarray) else [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
            for idx in flat_indices:
                x, y, w, h = boxes[idx]
                x1 = float(x)
                y1 = float(y)
                x2 = min(float(w_img), float(x + w))
                y2 = min(float(h_img), float(y + h))

                results.append(
                    DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidences[idx],
                        class_id=class_ids[idx],
                        class_label=class_labels[idx],
                    )
                )

        return results


class SyntheticMockDetector(DetectorBase):
    def __init__(
        self,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        target_classes: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            target_classes=target_classes,
        )
        self.scripted_detections: Dict[int, List[DetectionResult]] = {}
        self._current_frame_number: int = 0

    def load_model(self, model_path: str) -> None:
        pass

    def warmup(self, ai_frame_shape: tuple[int, int, int]) -> None:
        pass

    def add_scripted_detection(self, frame_number: int, detection: DetectionResult) -> None:
        if frame_number not in self.scripted_detections:
            self.scripted_detections[frame_number] = []
        self.scripted_detections[frame_number].append(detection)

    def set_current_frame(self, frame_number: int) -> None:
        self._current_frame_number = frame_number

    def detect(self, ai_frame: np.ndarray) -> list[DetectionResult]:
        results = self.scripted_detections.get(self._current_frame_number, [])
        filtered = [r for r in results if r.confidence >= self.confidence_threshold and self.is_target_class(r.class_label)]
        return filtered
