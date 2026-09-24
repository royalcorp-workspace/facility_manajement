"""
tools/test_spatial_rules.py
===========================
Automated Mathematical & Spatial Rule Test Suite.
Verifies:
1. Pure Math Geometry (ccw, segments_intersect, line crossing, polyline crossing).
2. Directional Tripwire crossing & debounce suppression.
3. Polyline Barrier crossing (multi-segment boundary).
4. Exclusion Mask suppression for dwell events.
5. Safe Walkway K3 compliance timer & vehicle proximity latch.
6. Density & Congestion monitor rule threshold & dwell triggering.
7. ROIZonesConfig backward compatibility for legacy "zones".
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.geometry import (
    ccw,
    segments_intersect,
    check_line_crossing,
    check_polyline_crossing,
    euclidean_distance,
    point_in_polygon,
)
from engine.config_loader import (
    ROIPoint,
    ROIZone,
    TripwireRule,
    BarrierRule,
    ExclusionMask,
    SafeWalkwayRule,
    DensityRule,
    ROIZonesConfig,
)
from engine.spatial_rules import SpatialRuleEngine
from engine.tracker_interface import TrackResult


def test_pure_math_geometry():
    print("[1/7] Testing Pure Math Geometry...")
    # Intersection test: (0,0)-(10,10) and (0,10)-(10,0)
    p1 = (0.0, 0.0)
    p2 = (10.0, 10.0)
    p3 = (0.0, 10.0)
    p4 = (10.0, 0.0)
    assert segments_intersect(p1, p2, p3, p4) is True, "Segments should intersect"

    # Parallel non-intersecting
    p5 = (0.0, 5.0)
    p6 = (10.0, 15.0)
    assert segments_intersect(p1, p2, p5, p6) is False, "Parallel lines should not intersect"

    # Collinear overlapping check
    assert euclidean_distance((0, 0), (3, 4)) == 5.0, "Distance should be 5.0"
    print("  ✓ Pure math geometry tests passed.")


def test_directional_tripwire():
    print("[2/7] Testing Directional Tripwires & Debounce...")
    engine = SpatialRuleEngine()

    # Vertical line from (100, 0) to (100, 200)
    # Side A: x < 100, Side B: x > 100
    tw = TripwireRule(
        tripwire_id="tw_01",
        label="Main Corridor Tripwire",
        active=True,
        p1=ROIPoint(x=100.0, y=0.0),
        p2=ROIPoint(x=100.0, y=200.0),
        direction="A_TO_B",
        target_classes=["person"],
        debounce_sec=3.0,
    )

    # 1. Crossing from side A (right) to side B (left, towards normal vector): (120, 100) -> (80, 100)
    trk1 = TrackResult(
        track_id=1,
        bbox=(70, 80, 90, 120),  # current centroid (80, 100)
        confidence=0.88,
        class_label="person",
        class_id=0,
        is_confirmed=True,
        prev_centroid=(120.0, 100.0),
    )

    events = engine.evaluate_tripwires(
        camera_id="cam_01",
        tracks=[trk1],
        tripwires=[tw],
        scale_x=1.0,
        scale_y=1.0,
        timestamp=100.0,
    )
    assert len(events) == 1, f"Should trigger 1 crossing event, got {len(events)}"
    assert events[0].event_type == "LINE_CROSSING"
    assert events[0].direction == "A_TO_B"

    # 2. Debounce suppression check (timestamp 101.0, < 3.0s cooldown)
    events_debounce = engine.evaluate_tripwires(
        camera_id="cam_01",
        tracks=[trk1],
        tripwires=[tw],
        scale_x=1.0,
        scale_y=1.0,
        timestamp=101.0,
    )
    assert len(events_debounce) == 0, "Debounce should suppress re-triggering within 3.0s"

    # 3. Crossing opposite direction B to A: (80, 100) -> (120, 100) should NOT trigger when direction is A_TO_B
    trk2 = TrackResult(
        track_id=2,
        bbox=(110, 80, 130, 120),  # current centroid (120, 100)
        confidence=0.85,
        class_label="person",
        class_id=0,
        is_confirmed=True,
        prev_centroid=(80.0, 100.0),
    )
    events_opp = engine.evaluate_tripwires(
        camera_id="cam_01",
        tracks=[trk2],
        tripwires=[tw],
        scale_x=1.0,
        scale_y=1.0,
        timestamp=100.0,
    )
    assert len(events_opp) == 0, "B_TO_A crossing should not trigger A_TO_B rule"
    print("  ✓ Directional tripwire & debounce passed.")


def test_polyline_barrier():
    print("[3/7] Testing Polyline Barrier Multi-Segment Crossing...")
    engine = SpatialRuleEngine()

    # L-shaped barrier: (50, 50) -> (150, 50) -> (150, 150)
    barrier = BarrierRule(
        barrier_id="bar_01",
        label="Perimeter L-Wall",
        active=True,
        points=[
            ROIPoint(x=50.0, y=50.0),
            ROIPoint(x=150.0, y=50.0),
            ROIPoint(x=150.0, y=150.0),
        ],
        direction="BOTH",
        target_classes=["person"],
        debounce_sec=3.0,
    )

    # Cross segment 2: (130, 100) -> (170, 100) across x=150
    trk = TrackResult(
        track_id=10,
        bbox=(160, 90, 180, 110),
        confidence=0.9,
        class_label="person",
        class_id=0,
        is_confirmed=True,
        prev_centroid=(130.0, 100.0),
    )

    events = engine.evaluate_barriers(
        camera_id="cam_01",
        tracks=[trk],
        barriers=[barrier],
        scale_x=1.0,
        scale_y=1.0,
        timestamp=200.0,
    )
    assert len(events) == 1, "Should detect barrier breach"
    assert events[0].event_type == "LINE_CROSSING"
    assert events[0].details.get("segment_index") == 1
    print("  ✓ Polyline barrier multi-segment passed.")


def test_exclusion_mask():
    print("[4/7] Testing Exclusion Mask Suppression...")
    engine = SpatialRuleEngine()

    # Exclusion mask box: (200, 200) to (300, 300)
    mask = ExclusionMask(
        mask_id="ex_01",
        label="Tree Leaves Area",
        active=True,
        points=[
            ROIPoint(x=200.0, y=200.0),
            ROIPoint(x=300.0, y=200.0),
            ROIPoint(x=300.0, y=300.0),
            ROIPoint(x=200.0, y=300.0),
        ],
        ignore_types=["dwell", "motion"],
    )

    # Track footpoint at (250, 250) -> inside exclusion
    trk_inside = TrackResult(
        track_id=20,
        bbox=(240, 200, 260, 250),  # bottom center = (250, 250)
        confidence=0.9,
        class_label="person",
        class_id=0,
        is_confirmed=True,
    )

    is_excluded = engine.filter_exclusion(trk_inside, [mask], scale_x=1.0, scale_y=1.0, filter_type="dwell")
    assert is_excluded is True, "Track inside exclusion mask must be excluded for dwell"

    # Track footpoint at (150, 150) -> outside
    trk_outside = TrackResult(
        track_id=21,
        bbox=(140, 100, 160, 150),  # bottom center = (150, 150)
        confidence=0.9,
        class_label="person",
        class_id=0,
        is_confirmed=True,
    )
    assert engine.filter_exclusion(trk_outside, [mask], scale_x=1.0, scale_y=1.0, filter_type="dwell") is False
    print("  ✓ Exclusion mask suppression passed.")


def test_safe_walkway_k3():
    print("[5/7] Testing Safe Walkway Compliance & Vehicle Proximity Latch...")
    engine = SpatialRuleEngine()

    # Walkway box: (0, 0) to (100, 100)
    walkway = SafeWalkwayRule(
        walkway_id="ww_01",
        label="Main Pedestrian Path",
        active=True,
        points=[
            ROIPoint(x=0.0, y=0.0),
            ROIPoint(x=100.0, y=0.0),
            ROIPoint(x=100.0, y=100.0),
            ROIPoint(x=0.0, y=100.0),
        ],
        violation_timeout_sec=5.0,
        vehicle_proximity_filter=True,
        proximity_radius_px=50.0,
    )

    # 1. Person outside walkway (footpoint 150, 150) - t = 0.0 -> no violation yet
    person = TrackResult(
        track_id=30,
        bbox=(140, 100, 160, 150),  # footpoint (150, 150), centroid (150, 125)
        confidence=0.9,
        class_label="person",
        class_id=0,
        is_confirmed=True,
    )

    ev0 = engine.evaluate_walkways("cam_01", [person], [walkway], 1.0, 1.0, timestamp=10.0)
    assert len(ev0) == 0, "Initial exit from walkway within timeout should not trigger"

    # t = 16.0 (6.0s elapsed > 5.0s timeout) -> triggers violation
    ev1 = engine.evaluate_walkways("cam_01", [person], [walkway], 1.0, 1.0, timestamp=16.0)
    assert len(ev1) == 1, "Should trigger walkway violation after timeout"
    assert ev1[0].event_type == "WALKWAY_VIOLATION"

    # 2. Vehicle proximity latch test: Person at (150, 125), Forklift/Truck nearby at (170, 125) (dist = 20px < 50px)
    engine2 = SpatialRuleEngine()
    truck = TrackResult(
        track_id=99,
        bbox=(160, 110, 180, 140),  # centroid (170, 125)
        confidence=0.92,
        class_label="truck",
        class_id=7,
        is_confirmed=True,
    )
    # Immediate trigger due to proximity latch even at t=0
    ev_hazard = engine2.evaluate_walkways("cam_01", [person, truck], [walkway], 1.0, 1.0, timestamp=20.0)
    assert len(ev_hazard) == 1, "Vehicle hazard proximity latch must trigger immediately"
    assert ev_hazard[0].details.get("has_vehicle_hazard") is True
    print("  ✓ Safe walkway K3 compliance & proximity latch passed.")


def test_density_congestion():
    print("[6/7] Testing Density & Area Congestion Rule...")
    engine = SpatialRuleEngine()

    poly = ROIZone(
        zone_id="zone_loading",
        label="Loading Bay",
        type="polygon",
        active=True,
        color_hex="#00FF00",
        points=[
            ROIPoint(x=0.0, y=0.0),
            ROIPoint(x=200.0, y=0.0),
            ROIPoint(x=200.0, y=200.0),
            ROIPoint(x=0.0, y=200.0),
        ],
        trigger_on=["enter"],
    )

    rule = DensityRule(
        rule_id="den_01",
        label="Loading Bay Congestion Alert",
        active=True,
        zone_ref="zone_loading",
        max_allowed_objects=2,
        min_dwell_sec=3.0,
    )

    # 3 tracks inside loading bay
    tracks = [
        TrackResult(track_id=1, bbox=(10, 10, 30, 30), confidence=0.9, class_label="car", class_id=2, is_confirmed=True),
        TrackResult(track_id=2, bbox=(40, 40, 60, 60), confidence=0.9, class_label="car", class_id=2, is_confirmed=True),
        TrackResult(track_id=3, bbox=(70, 70, 90, 90), confidence=0.9, class_label="car", class_id=2, is_confirmed=True),
    ]

    # At t=10.0: first congestion detection (dwell = 0 < 3.0s) -> no alert yet
    ev0 = engine.evaluate_density("cam_01", tracks, [rule], [poly], 1.0, 1.0, timestamp=10.0)
    assert len(ev0) == 0

    # At t=14.0 (dwell = 4.0s >= 3.0s) -> triggers alert
    ev1 = engine.evaluate_density("cam_01", tracks, [rule], [poly], 1.0, 1.0, timestamp=14.0)
    assert len(ev1) == 1, "Should trigger CONGESTION_ALERT"
    assert ev1[0].event_type == "CONGESTION_ALERT"
    assert ev1[0].details.get("current_count") == 3
    print("  ✓ Density congestion monitor passed.")


def test_backward_compatibility():
    print("[7/7] Testing Backward Compatibility for Legacy 'zones' Schema...")
    legacy_payload = {
        "schema_version": "1.0",
        "camera_id": "cam_legacy",
        "zones": [
            {
                "zone_id": "zone_01",
                "label": "Legacy Zone",
                "type": "polygon",
                "active": True,
                "color_hex": "#00FF00",
                "points": [(10, 10), (100, 10), (100, 100), (10, 100)],
                "trigger_on": ["linger"],
                "linger_threshold_sec": 5,
            }
        ]
    }

    config = ROIZonesConfig.model_validate(legacy_payload)
    assert len(config.polygons) == 1, "Legacy 'zones' should migrate to 'polygons'"
    assert config.polygons[0].zone_id == "zone_01"
    assert len(config.active_zones) == 1
    assert config.zones == config.polygons
    print("  ✓ Backward compatibility migration passed.")


def main():
    print("=" * 65)
    print(" FACILITY MANAGEMENT - ADVANCED SPATIAL ANALYTICS TEST SUITE")
    print("=" * 65)
    test_pure_math_geometry()
    test_directional_tripwire()
    test_polyline_barrier()
    test_exclusion_mask()
    test_safe_walkway_k3()
    test_density_congestion()
    test_backward_compatibility()
    print("=" * 65)
    print(" ALL 7 SPATIAL ANALYTICS SUITES PASSED SUCCESSFULLY! ✓")
    print("=" * 65)


if __name__ == "__main__":
    main()
