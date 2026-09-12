"""C2 incident review console tests (no GPU / cameras required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alert_logger import AlertLogger
from c2_console import create_app


def _seed(tmp_path: Path) -> AlertLogger:
    logger = AlertLogger(db_path=tmp_path / "alerts.db", snapshot_dir=tmp_path / "snaps")
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[10:50, 10:50] = 180
    logger.log_intrusion(
        camera_id="Cam_1_Outpost",
        global_id=17,
        frame=frame,
        bbox=(10, 10, 50, 50),
        timestamp=1_710_000_001.5,
        footprint=(30, 50),
    )
    return logger


def test_console_lists_and_serves_evidence(tmp_path: Path):
    logger = _seed(tmp_path)
    app = create_app(db_path=tmp_path / "alerts.db", snapshot_dir=tmp_path / "snaps")
    client = app.test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert b"INCIDENT REVIEW" in home.data
    stats = client.get("/api/stats").get_json()
    assert stats["total"] == 1
    rows = client.get("/api/alerts?camera_id=Cam_1_Outpost").get_json()
    assert len(rows) == 1
    assert rows[0]["global_id"] == 17
    detail = client.get(f"/api/alerts/{rows[0]['id']}").get_json()
    assert detail["camera_id"] == "Cam_1_Outpost"
    evidence = client.get("/evidence", query_string={"path": detail["snapshot_path"]})
    assert evidence.status_code == 200
    assert evidence.mimetype.startswith("image/")
    blocked = client.get("/evidence", query_string={"path": "/etc/passwd"})
    assert blocked.status_code in (403, 404)
    assert logger.get(rows[0]["id"]) is not None
