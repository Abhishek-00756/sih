"""Persistent C2 audit trail for geofence intrusions.

Creates a per-incident evidence folder, writes timestamped full-frame and
crop snapshots, and records structured metadata (Global ID, camera, bbox,
footprint) in SQLite for command-center search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

LOGGER = logging.getLogger("perception.alerts")

BBox = Tuple[int, int, int, int]


def _utc_now(ts: Optional[float] = None) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _safe_token(value: object) -> str:
    text = "".join(ch if str(ch).isalnum() or ch in "-_" else "_" for ch in str(value))
    return text.strip("_") or "unknown"


class AlertLogger:
    def __init__(
        self,
        db_path: Union[str, Path] = "border_alerts.db",
        snapshot_dir: Union[str, Path] = "alert_snapshots",
    ) -> None:
        self.db_path = Path(db_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    unix_time REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    global_id INTEGER NOT NULL,
                    alert_type TEXT NOT NULL,
                    zone_id TEXT,
                    zone_name TEXT,
                    bbox TEXT NOT NULL,
                    footprint TEXT,
                    snapshot_path TEXT NOT NULL,
                    crop_path TEXT,
                    incident_dir TEXT
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(security_alerts)")}
            if "incident_dir" not in cols:
                conn.execute("ALTER TABLE security_alerts ADD COLUMN incident_dir TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_camera ON security_alerts(camera_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_gid ON security_alerts(global_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_time ON security_alerts(unix_time)"
            )
            conn.commit()

    def _annotate(self, frame: np.ndarray, bbox: BBox, global_id: int, camera_id: str) -> np.ndarray:
        evidence = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            evidence,
            f"INTRUSION ALERT - GID: {global_id} CAM: {camera_id}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return evidence

    def _crop(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _incident_dir(self, camera_id: str, global_id: int, now: datetime) -> Path:
        file_timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        folder = (
            self.snapshot_dir
            / now.strftime("%Y%m%d")
            / f"cam_{_safe_token(camera_id)}_gid_{int(global_id)}_{file_timestamp}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def log_intrusion(
        self,
        camera_id: str,
        global_id: int,
        frame: np.ndarray,
        bbox: Sequence[float],
        timestamp: Optional[float] = None,
        alert_type: str = "Geofence Intrusion",
        zone_id: str = "",
        zone_name: str = "",
        footprint: Optional[Sequence[float]] = None,
    ) -> str:
        """Save visual evidence into an incident folder and insert a searchable row."""
        now = _utc_now(timestamp)
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        box: BBox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        gid = int(global_id)
        incident_dir = self._incident_dir(camera_id, gid, now)
        filepath = incident_dir / "full.jpg"
        crop_path: Optional[Path] = None

        evidence = self._annotate(frame, box, gid, str(camera_id))
        if not cv2.imwrite(str(filepath), evidence):
            raise RuntimeError(f"failed to write evidence snapshot {filepath}")

        crop = self._crop(frame, box)
        if crop is not None and crop.size:
            crop_path = incident_dir / "crop.jpg"
            cv2.imwrite(str(crop_path), crop)

        payload = {
            "timestamp": timestamp_str,
            "unix_time": float(now.timestamp()),
            "camera_id": str(camera_id),
            "global_id": gid,
            "alert_type": alert_type,
            "zone_id": zone_id or None,
            "zone_name": zone_name or None,
            "bbox": list(box),
            "footprint": [int(footprint[0]), int(footprint[1])] if footprint else None,
            "snapshot_path": str(filepath),
            "crop_path": str(crop_path) if crop_path else None,
        }
        (incident_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO security_alerts (
                    timestamp, unix_time, camera_id, global_id, alert_type,
                    zone_id, zone_name, bbox, footprint, snapshot_path, crop_path, incident_dir
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_str,
                    float(now.timestamp()),
                    str(camera_id),
                    gid,
                    alert_type,
                    zone_id or None,
                    zone_name or None,
                    json.dumps(list(box)),
                    json.dumps([int(footprint[0]), int(footprint[1])]) if footprint else None,
                    str(filepath),
                    str(crop_path) if crop_path else None,
                    str(incident_dir),
                ),
            )
            conn.commit()

        LOGGER.warning(
            "[SECURITY EVENT SAVED] G-ID %s on %s logged. Snapshot: %s",
            gid,
            camera_id,
            filepath,
        )
        return str(filepath)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.search(limit=limit)

    def get(self, alert_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM security_alerts WHERE id = ?",
                (int(alert_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def search(
        self,
        camera_id: Optional[str] = None,
        global_id: Optional[int] = None,
        alert_type: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(str(camera_id))
        if global_id is not None:
            clauses.append("global_id = ?")
            params.append(int(global_id))
        if alert_type:
            clauses.append("alert_type = ?")
            params.append(str(alert_type))
        if since is not None:
            clauses.append("unix_time >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("unix_time <= ?")
            params.append(float(until))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT * FROM security_alerts {where} "
            "ORDER BY id DESC LIMIT ? OFFSET ?"
        )
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM security_alerts").fetchone()[0]
            cameras = conn.execute(
                "SELECT camera_id, COUNT(*) AS n FROM security_alerts GROUP BY camera_id ORDER BY n DESC"
            ).fetchall()
            types = conn.execute(
                "SELECT alert_type, COUNT(*) AS n FROM security_alerts GROUP BY alert_type ORDER BY n DESC"
            ).fetchall()
            latest = conn.execute(
                "SELECT timestamp FROM security_alerts ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return {
            "total": int(total),
            "latest": None if latest is None else latest[0],
            "by_camera": {row[0]: int(row[1]) for row in cameras},
            "by_type": {row[0]: int(row[1]) for row in types},
        }
