from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_label: str
    frame_number: int = 0
    camera_id: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence harus dalam [0, 1], dapat: {self.confidence}")

class DetectorBase(ABC):

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[list[str]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or []

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        ...

    @abstractmethod
    def detect(self, ai_frame: np.ndarray) -> list[DetectionResult]:

        ...

    @abstractmethod
    def warmup(self, ai_frame_shape: tuple[int, int, int]) -> None:
        ...

    def is_target_class(self, class_label: str) -> bool:
        if not self.target_classes:
            return True   
        return class_label in self.target_classes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"conf={self.confidence_threshold}, "
            f"iou={self.iou_threshold}, "
            f"targets={self.target_classes or 'all'})"
        )
