#!/usr/bin/env python3
"""Write a few synthetic incidents so the C2 console is reviewable without cameras."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from alert_logger import AlertLogger


def seed(db_path: Path, snapshot_dir: Path) -> int:
    logger = AlertLogger(db_path=db_path, snapshot_dir=snapshot_dir)
    samples = [
        ("Cam_1_Outpost", 17, (80, 60, 180, 220), (130, 220), 1_710_000_001.0),
        ("Cam_2_Gate", 4, (200, 90, 310, 260), (255, 260), 1_710_000_040.0),
        ("Sector_Alpha", 17, (40, 120, 140, 300), (90, 300), 1_710_000_090.0),
    ]
    for cam, gid, bbox, foot, ts in samples:
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        frame[:] = (28, 36, 28)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (40, 90, 180), -1)
        logger.log_intrusion(
            camera_id=cam,
            global_id=gid,
            frame=frame,
            bbox=bbox,
            timestamp=ts,
            zone_id="zone_restricted_gate",
            zone_name="Restricted Gate",
            footprint=foot,
        )
    return logger.stats()["total"]


if __name__ == "__main__":
    n = seed(ROOT / "border_alerts.db", ROOT / "alert_snapshots")
    print(f"seeded {n} incidents")
