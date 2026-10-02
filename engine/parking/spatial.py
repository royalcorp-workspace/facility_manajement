from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from engine.geometry import bbox_iou


def bbox_polygon_overlap_ratio(
    bbox: Tuple[float, float, float, float],
    pts_scaled: np.ndarray,
) -> float:
    bx1, by1, bx2, by2 = bbox
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    bbox_area = bw * bh

    px, py, pw, ph = cv2.boundingRect(pts_scaled)
    ix1 = max(bx1, float(px))
    iy1 = max(by1, float(py))
    ix2 = min(bx2, float(px + pw))
    iy2 = min(by2, float(py + ph))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    rx1, ry1 = int(math.floor(ix1)), int(math.floor(iy1))
    rx2, ry2 = int(math.ceil(ix2)), int(math.ceil(iy2))
    mw = rx2 - rx1
    mh = ry2 - ry1
    if mw <= 0 or mh <= 0:
        return 0.0

    mask = np.zeros((mh, mw), dtype=np.uint8)
    poly_shifted = pts_scaled - np.array([rx1, ry1], dtype=np.int32)
    cv2.fillPoly(mask, [poly_shifted], 255)

    bx_s1 = max(0, int(round(bx1 - rx1)))
    by_s1 = max(0, int(round(by1 - ry1)))
    bx_s2 = min(mw, int(round(bx2 - rx1)))
    by_s2 = min(mh, int(round(by2 - ry1)))

    if bx_s2 <= bx_s1 or by_s2 <= by_s1:
        return 0.0

    box_submask = mask[by_s1:by_s2, bx_s1:bx_s2]
    intersection_area = float(cv2.countNonZero(box_submask))
    return intersection_area / bbox_area


def bbox_polygon_ios(
    bbox: Tuple[float, float, float, float],
    pts_scaled: np.ndarray,
) -> float:
    """
    Menghitung rasio irisan area (Intersection over Slot Area / IoS):
    Luas irisan antara bounding box kendaraan dan poligon slot dibagi luas poligon slot.
    """
    if len(pts_scaled) < 3:
        return 0.0

    slot_area = float(cv2.contourArea(pts_scaled))
    if slot_area <= 0.0:
        return 0.0

    bx1, by1, bx2, by2 = bbox
    px, py, pw, ph = cv2.boundingRect(pts_scaled)
    ix1 = max(bx1, float(px))
    iy1 = max(by1, float(py))
    ix2 = min(bx2, float(px + pw))
    iy2 = min(by2, float(py + ph))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    rx1, ry1 = int(math.floor(ix1)), int(math.floor(iy1))
    rx2, ry2 = int(math.ceil(ix2)), int(math.ceil(iy2))
    mw = rx2 - rx1
    mh = ry2 - ry1
    if mw <= 0 or mh <= 0:
        return 0.0

    mask = np.zeros((mh, mw), dtype=np.uint8)
    poly_shifted = pts_scaled - np.array([rx1, ry1], dtype=np.int32)
    cv2.fillPoly(mask, [poly_shifted], 255)

    bx_s1 = max(0, int(round(bx1 - rx1)))
    by_s1 = max(0, int(round(by1 - ry1)))
    bx_s2 = min(mw, int(round(bx2 - rx1)))
    by_s2 = min(mh, int(round(by2 - ry1)))

    if bx_s2 <= bx_s1 or by_s2 <= by_s1:
        return 0.0

    box_submask = mask[by_s1:by_s2, bx_s1:bx_s2]
    intersection_area = float(cv2.countNonZero(box_submask))
    return intersection_area / slot_area


def bbox_ios(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(intersection / min(area_a, area_b))
