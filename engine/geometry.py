from __future__ import annotations

from typing import Sequence


Point = tuple[float, float]
Polygon = Sequence[Point]
BBox = tuple[float, float, float, float]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def points_any_in_polygon(points: Sequence[Point], polygon: Polygon) -> bool:

    return any(point_in_polygon(p, polygon) for p in points)


def points_all_in_polygon(points: Sequence[Point], polygon: Polygon) -> bool:
    return all(point_in_polygon(p, polygon) for p in points)

def bbox_centroid(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_bottom_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))

def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def bbox_corners(bbox: BBox) -> list[Point]:
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def polygon_area(polygon: Polygon) -> float:
    n = len(polygon)
    if n < 3:
        return 0.0

    area = 0.0
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        area += (xj + xi) * (yj - yi)
        j = i

    return abs(area) / 2.0


def scale_points(
    points: Sequence[Point],
    scale_x: float,
    scale_y: float,
) -> list[Point]:
    return [(x / scale_x, y / scale_y) for x, y in points]


def roi_zone_to_polygon(zone_points: list[dict]) -> Polygon:
    return [(float(p["x"]), float(p["y"])) for p in zone_points]

def check_bbox_in_zone(
    bbox: BBox,
    polygon: Polygon,
    method: str = "bottom_center",
) -> bool:

    match method:
        case "bottom_center":
            return point_in_polygon(bbox_bottom_center(bbox), polygon)
        case "centroid":
            return point_in_polygon(bbox_centroid(bbox), polygon)
        case "any_corner":
            return points_any_in_polygon(bbox_corners(bbox), polygon)
        case "all_corners":
            return points_all_in_polygon(bbox_corners(bbox), polygon)
        case _:
            raise ValueError(f"Method tidak dikenal: '{method}'. Pilih: bottom_center, centroid, any_corner, all_corners")


def ccw(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(p1: Point, p2: Point, q: Point) -> bool:
    return (
        min(p1[0], p2[0]) <= q[0] <= max(p1[0], p2[0])
        and min(p1[1], p2[1]) <= q[1] <= max(p1[1], p2[1])
    )


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    # Kasus umum: titik ujung berada di sisi berseberangan
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Kasus batas: kolinier dan menyentuh segmen
    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True

    return False


def check_line_crossing(
    p_prev: Point,
    p_curr: Point,
    line_start: Point,
    line_end: Point,
    allowed_direction: str = "BOTH",
) -> tuple[bool, str]:
    if not segments_intersect(p_prev, p_curr, line_start, line_end):
        return False, "NONE"

    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1

    vx = p_curr[0] - p_prev[0]
    vy = p_curr[1] - p_prev[1]

    dot = vx * (-dy) + vy * dx

    if dot > 0:
        actual_direction = "A_TO_B"
    elif dot < 0:
        actual_direction = "B_TO_A"
    else:
        actual_direction = "BOTH"

    if allowed_direction == "BOTH" or allowed_direction == actual_direction:
        return True, actual_direction
    return False, actual_direction


def check_polyline_crossing(
    p_prev: Point,
    p_curr: Point,
    polyline: Sequence[Point],
    allowed_direction: str = "BOTH",
) -> tuple[bool, Optional[int], str]:
    n = len(polyline)
    if n < 2:
        return False, None, "NONE"

    for i in range(n - 1):
        s_start = polyline[i]
        s_end = polyline[i + 1]
        crossed, direction = check_line_crossing(p_prev, p_curr, s_start, s_end, allowed_direction)
        if crossed:
            return True, i, direction

    return False, None, "NONE"


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Jarak Euclidean 2D antara dua titik."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (dx * dx + dy * dy) ** 0.5
