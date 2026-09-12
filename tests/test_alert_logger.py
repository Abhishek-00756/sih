"""SQLite evidence logger tests (no GPU / cameras required)."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alert_logger import AlertLogger
from geofence_manager import GeofenceManager


SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]


def test_log_intrusion_writes_snapshot_and_row(tmp_path: Path):
    db = tmp_path / "alerts.db"
    snaps = tmp_path / "snaps"
    logger = AlertLogger(db_path=db, snapshot_dir=snaps)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 200
    path = logger.log_intrusion(
        camera_id="Cam_1_Outpost",
        global_id=17,
        frame=frame,
        bbox=(20, 20, 80, 80),
        timestamp=1_710_000_001.5,
        zone_id="zone_restricted_gate",
        zone_name="Restricted Gate",
        footprint=(50, 80),
    )
    assert Path(path).exists()
    rows = logger.recent(limit=5)
    assert len(rows) == 1
    row = rows[0]
    assert row["global_id"] == 17
    assert row["camera_id"] == "Cam_1_Outpost"
    assert row["alert_type"] == "Geofence Intrusion"
    assert json.loads(row["bbox"]) == [20, 20, 80, 80]
    assert json.loads(row["footprint"]) == [50, 80]
    assert Path(row["snapshot_path"]).exists()
    assert row["crop_path"] and Path(row["crop_path"]).exists()
    assert row["incident_dir"] and Path(row["incident_dir"]).is_dir()
    assert (Path(row["incident_dir"]) / "meta.json").exists()
    assert logger.search(camera_id="Cam_1_Outpost", global_id=17)
    assert logger.stats()["total"] == 1


def test_geofence_hooks_logger_once(tmp_path: Path):
    logger = AlertLogger(db_path=tmp_path / "a.db", snapshot_dir=tmp_path / "s")
    mgr = GeofenceManager(zone_points=SQUARE, cooldown_seconds=60.0, logger=logger)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    inside = (180, 160, 220, 250)
    mgr.check_intrusion(frame, inside, global_id=7, camera_id="cam_a", timestamp=10.0, source_camera_id="Cam_1_Outpost")
    mgr.check_intrusion(frame, inside, global_id=7, camera_id="cam_a", timestamp=11.0, source_camera_id="Cam_1_Outpost")
    rows = logger.recent()
    assert len(rows) == 1
    assert rows[0]["camera_id"] == "Cam_1_Outpost"
    assert rows[0]["global_id"] == 7


def test_schema_has_audit_columns(tmp_path: Path):
    db = tmp_path / "schema.db"
    AlertLogger(db_path=db, snapshot_dir=tmp_path / "s")
    with sqlite3.connect(str(db)) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(security_alerts)")}
    for name in (
        "timestamp",
        "camera_id",
        "global_id",
        "bbox",
        "snapshot_path",
        "footprint",
        "incident_dir",
        "image_hash",
        "previous_hash",
        "block_hash",
    ):
        assert name in cols


def test_ledger_chains_events_and_detects_image_tamper(tmp_path: Path):
    logger = AlertLogger(db_path=tmp_path / "l.db", snapshot_dir=tmp_path / "s")
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    logger.log_intrusion("Cam_1", 1, frame, (10, 10, 40, 40), timestamp=1.0)
    logger.log_intrusion("Cam_1", 2, frame, (10, 10, 40, 40), timestamp=2.0)
    result = logger.verify_ledger()
    assert result["ok"] is True
    assert result["blocks"] == 3

    rows = logger.recent()
    snap = Path(rows[0]["snapshot_path"])
    snap.write_bytes(snap.read_bytes() + b"tamper")
    broken = logger.verify_ledger()
    assert broken["ok"] is False
    assert "modified" in broken["message"]


def test_ledger_detects_row_tamper(tmp_path: Path):
    db = tmp_path / "t.db"
    logger = AlertLogger(db_path=db, snapshot_dir=tmp_path / "s")
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    logger.log_intrusion("Cam_2", 9, frame, (5, 5, 20, 20), timestamp=3.0)
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE security_ledger SET camera_id = 'HACKED' WHERE id = 2")
        conn.commit()
    broken = logger.verify_ledger()
    assert broken["ok"] is False
    assert "altered" in broken["message"]
