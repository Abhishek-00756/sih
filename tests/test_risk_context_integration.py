"""Integration tests for directional tripwire + dwell context propagation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activity_analyzer import ActivityAnalyzer
from geofence_manager import GeofenceManager, VirtualTripwire
from risk_scorer import RiskScorer


class CaptureLogger:
    def __init__(self) -> None:
        self.calls = []

    def log_intrusion(self, **kwargs):
        self.calls.append(kwargs)
        return "evidence.jpg"


def test_tripwire_alert_receives_current_dwell_time():
    logger = CaptureLogger()
    wire = VirtualTripwire((100, 200), (300, 200), wire_id="border")
    mgr = GeofenceManager(tripwires=[wire], logger=logger, cooldown_seconds=1.0)
    analyzer = ActivityAnalyzer(dwell_threshold=30.0, logger=logger)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    bbox_above = (180, 100, 220, 150)
    bbox_below = (180, 210, 220, 260)

    analyzer.analyze_behavior(frame, bbox_above, 7, camera_id="Cam_1_Outpost", timestamp=100.0, draw=False)
    analyzer.analyze_behavior(frame, bbox_above, 7, camera_id="Cam_1_Outpost", timestamp=161.0, draw=False)

    # Mapped geofence camera ID differs from source camera ID, matching main.py.
    mgr.check_tripwire(
        frame,
        bbox_above,
        global_id=7,
        camera_id="cam_01",
        source_camera_id="Cam_1_Outpost",
        timestamp=160.0,
        draw=False,
    )
    mgr.check_tripwire(
        frame,
        bbox_below,
        global_id=7,
        camera_id="cam_01",
        source_camera_id="Cam_1_Outpost",
        timestamp=161.0,
        draw=False,
    )

    tripwire_calls = [c for c in logger.calls if str(c.get("alert_type", "")).startswith("Tripwire")]
    assert len(tripwire_calls) == 1
    call = tripwire_calls[0]
    assert call["direction"] == "INBOUND"
    assert call["dwell_time"] == 61.0


def test_loitering_alert_reuses_previous_tripwire_direction():
    logger = CaptureLogger()
    wire = VirtualTripwire((100, 200), (300, 200), wire_id="border")
    mgr = GeofenceManager(tripwires=[wire], logger=logger, cooldown_seconds=1.0)
    analyzer = ActivityAnalyzer(dwell_threshold=30.0, logger=logger)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    above = (180, 100, 220, 150)
    below = (180, 210, 220, 260)

    analyzer.analyze_behavior(frame, above, 9, camera_id="Cam_1_Outpost", timestamp=100.0, draw=False)
    mgr.check_tripwire(
        frame,
        above,
        global_id=9,
        camera_id="cam_01",
        source_camera_id="Cam_1_Outpost",
        timestamp=101.0,
        draw=False,
    )
    mgr.check_tripwire(
        frame,
        below,
        global_id=9,
        camera_id="cam_01",
        source_camera_id="Cam_1_Outpost",
        timestamp=102.0,
        draw=False,
    )

    analyzer.analyze_behavior(frame, below, 9, camera_id="Cam_1_Outpost", timestamp=161.0, draw=False)

    loiter_calls = [c for c in logger.calls if c.get("alert_type") == "Loitering"]
    assert len(loiter_calls) == 1
    call = loiter_calls[0]
    assert call["direction"] == "INBOUND"
    assert call["dwell_time"] == 61.0


def test_combined_inbound_night_loiter_reaches_critical_score():
    scorer = RiskScorer(timezone_name="Asia/Kolkata")
    night_ist = datetime(2026, 9, 12, 22, 0, tzinfo=__import__("zoneinfo").ZoneInfo("Asia/Kolkata"))
    score, reasons = scorer.score(
        event_type="Tripwire INBOUND",
        direction="INBOUND",
        timestamp=night_ist.timestamp(),
        dwell_time=61.0,
    )
    assert score == 100
    assert reasons == ["base", "inbound_tripwire", "night", "loitering"]
    assert scorer.label(score) == "CRITICAL"
