"""Dwell-time / loitering analyzer tests (no GPU / cameras required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activity_analyzer import ActivityAnalyzer, numeric_entity_id
from alert_logger import AlertLogger


def test_numeric_entity_id_maps_vehicle_tokens():
    assert numeric_entity_id(17) == 17
    assert numeric_entity_id("V-4") == 10_000_004


def test_loitering_alerts_once_after_threshold(tmp_path: Path):
    logger = AlertLogger(db_path=tmp_path / "a.db", snapshot_dir=tmp_path / "s")
    analyzer = ActivityAnalyzer(dwell_threshold=30.0, logger=logger)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = (20, 20, 80, 120)
    analyzer.analyze_behavior(frame, bbox, 7, "Person", "Cam_1_Outpost", timestamp=10.0, draw=False)
    analyzer.analyze_behavior(frame, bbox, 7, "Person", "Cam_1_Outpost", timestamp=20.0, draw=False)
    assert analyzer.pop_alerts() == []
    analyzer.analyze_behavior(frame, bbox, 7, "Person", "Cam_1_Outpost", timestamp=41.0, draw=False)
    analyzer.analyze_behavior(frame, bbox, 7, "Person", "Cam_1_Outpost", timestamp=42.0, draw=False)
    alerts = analyzer.pop_alerts()
    assert len(alerts) == 1
    assert alerts[0]["entity_id"] == 7
    rows = logger.search(alert_type="Loitering")
    assert len(rows) == 1
    assert rows[0]["camera_id"] == "Cam_1_Outpost"
    assert rows[0]["zone_name"] == "Person"


def test_vehicle_loitering_and_stale_cleanup(tmp_path: Path):
    logger = AlertLogger(db_path=tmp_path / "v.db", snapshot_dir=tmp_path / "s")
    analyzer = ActivityAnalyzer(dwell_threshold=5.0, logger=logger)
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    bbox = (10, 10, 90, 80)
    vid = "V-3"
    analyzer.analyze_behavior(frame, bbox, vid, "Truck", "Cam_2_Gate", timestamp=1.0, draw=False)
    analyzer.analyze_behavior(frame, bbox, vid, "Truck", "Cam_2_Gate", timestamp=7.0, draw=False)
    assert len(analyzer.pop_alerts()) == 1
    rows = logger.search(alert_type="Loitering")
    assert rows[0]["global_id"] == numeric_entity_id(vid)
    removed = analyzer.cleanup_stale_tracks([], camera_id="Cam_2_Gate")
    assert removed == 1
    analyzer.analyze_behavior(frame, bbox, vid, "Truck", "Cam_2_Gate", timestamp=8.0, draw=False)
    analyzer.analyze_behavior(frame, bbox, vid, "Truck", "Cam_2_Gate", timestamp=14.0, draw=False)
    assert len(analyzer.pop_alerts()) == 1
