"""Footprint geofence overlay tests (no GPU / cameras required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofence_manager import GeofenceManager, RestrictedZone, footprint_from_bbox


SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]


def test_footprint_is_bottom_center():
    assert footprint_from_bbox((10, 20, 50, 80)) == (30, 80)


def test_intrusion_alerts_once_per_global_id():
    mgr = GeofenceManager(zone_points=SQUARE, cooldown_seconds=60.0)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    inside = (180, 160, 220, 250)
    mgr.check_intrusion(frame, inside, global_id=7, camera_id="cam_a", timestamp=10.0, draw=False)
    mgr.check_intrusion(frame, inside, global_id=7, camera_id="cam_a", timestamp=11.0, draw=False)
    alerts = mgr.pop_alerts()
    assert len(alerts) == 1
    assert alerts[0].global_id == 7
    assert alerts[0].footprint == (200, 250)


def test_outside_polygon_does_not_alert():
    mgr = GeofenceManager(zone_points=SQUARE, cooldown_seconds=5.0)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    mgr.check_intrusion(frame, (350, 350, 380, 390), global_id=3, timestamp=1.0, draw=False)
    assert mgr.pop_alerts() == []


def test_cooldown_expiry_allows_new_alert():
    mgr = GeofenceManager(zone_points=SQUARE, cooldown_seconds=5.0)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    bbox = (180, 160, 220, 250)
    mgr.check_intrusion(frame, bbox, global_id=1, timestamp=1.0, draw=False)
    mgr.check_intrusion(frame, bbox, global_id=1, timestamp=7.0, draw=False)
    assert len(mgr.pop_alerts()) == 2


def test_camera_scoped_zone_ignores_other_feeds():
    zone = RestrictedZone(SQUARE, zone_id="gate", name="Gate", camera_id="cam_a")
    mgr = GeofenceManager(zones=[zone], cooldown_seconds=1.0)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    bbox = (180, 160, 220, 250)
    mgr.check_intrusion(frame, bbox, global_id=9, camera_id="cam_b", timestamp=1.0, draw=False)
    assert mgr.pop_alerts() == []
    mgr.check_intrusion(frame, bbox, global_id=9, camera_id="cam_a", timestamp=1.0, draw=False)
    assert len(mgr.pop_alerts()) == 1
