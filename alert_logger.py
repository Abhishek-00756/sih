"""Persistent C2 audit trail for geofence intrusions.

Creates a per-incident evidence folder, writes timestamped full-frame and
crop snapshots, and records structured metadata (Global ID, camera, bbox,
footprint) in SQLite for command-center search. Each event is chained with
SHA-256 image and block hashes so operators can prove the ledger was not
tampered with.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from risk_scorer import RiskScorer

import cv2
import numpy as np

LOGGER = logging.getLogger("perception.alerts")

BBox = Tuple[int, int, int, int]
GENESIS_HASH = "GENESIS_HASH"


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
        self.scorer = RiskScorer()
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
            for col, decl in (
                ("image_hash", "TEXT"),
                ("previous_hash", "TEXT"),
                ("block_hash", "TEXT"),
                ("direction", "TEXT"),
                ("risk_score", "INTEGER"),
                ("risk_label", "TEXT"),
                ("risk_factors", "TEXT"),
            ):
                if col not in cols:
                    conn.execute(f"ALTER TABLE security_alerts ADD COLUMN {col} {decl}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_camera ON security_alerts(camera_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_gid ON security_alerts(global_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_time ON security_alerts(unix_time)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    global_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    image_hash TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    block_hash TEXT NOT NULL
                )
                """
            )
            count = conn.execute("SELECT COUNT(*) FROM security_ledger").fetchone()[0]
            if int(count) == 0:
                conn.execute(
                    """
                    INSERT INTO security_ledger (
                        timestamp, camera_id, global_id, event_type, snapshot_path,
                        image_hash, previous_hash, block_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "2024-01-01 00:00:00",
                        "SYSTEM",
                        0,
                        "GENESIS",
                        "none",
                        "none",
                        "0",
                        GENESIS_HASH,
                    ),
                )
            conn.commit()

    def _annotate(
        self,
        frame: np.ndarray,
        bbox: BBox,
        global_id: int,
        camera_id: str,
        risk_score: Optional[int] = None,
        risk_label: Optional[str] = None,
    ) -> np.ndarray:
        evidence = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)
        title = f"INTRUSION ALERT - GID: {global_id} CAM: {camera_id}"
        if risk_score is not None:
            title = f"{title} RISK {risk_score} {risk_label or ''}".strip()
        cv2.putText(
            evidence,
            title,
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

    @staticmethod
    def file_hash(filepath: Union[str, Path]) -> str:
        sha256 = hashlib.sha256()
        with Path(filepath).open("rb") as handle:
            while True:
                chunk = handle.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def block_hash(
        timestamp: str,
        camera_id: str,
        global_id: int,
        event_type: str,
        image_hash: str,
        previous_hash: str,
    ) -> str:
        data = f"{timestamp}{camera_id}{int(global_id)}{event_type}{image_hash}{previous_hash}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _latest_block_hash(self, conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT block_hash FROM security_ledger ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return GENESIS_HASH
        return str(row[0])

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
        direction: Optional[str] = None,
        dwell_time: Optional[float] = None,
    ) -> str:
        """Save visual evidence into an incident folder and insert a searchable row."""
        now = _utc_now(timestamp)
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        box: BBox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        gid = int(global_id)
        incident_dir = self._incident_dir(camera_id, gid, now)
        filepath = incident_dir / "full.jpg"
        crop_path: Optional[Path] = None

        risk_score, risk_factors = self.scorer.score(
            event_type=alert_type,
            direction=direction,
            timestamp=float(now.timestamp()),
            dwell_time=dwell_time,
        )
        risk_label = self.scorer.label(risk_score)
        evidence = self._annotate(
            frame,
            box,
            gid,
            str(camera_id),
            risk_score=risk_score,
            risk_label=risk_label,
        )
        if not cv2.imwrite(str(filepath), evidence):
            raise RuntimeError(f"failed to write evidence snapshot {filepath}")

        crop = self._crop(frame, box)
        if crop is not None and crop.size:
            crop_path = incident_dir / "crop.jpg"
            cv2.imwrite(str(crop_path), crop)

        img_hash = self.file_hash(filepath)
        with self._lock, self._connect() as conn:
            prev_hash = self._latest_block_hash(conn)
            chain_hash = self.block_hash(
                timestamp_str,
                str(camera_id),
                gid,
                alert_type,
                img_hash,
                prev_hash,
            )
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
                "image_hash": img_hash,
                "previous_hash": prev_hash,
                "block_hash": chain_hash,
                "direction": direction,
                "risk_score": risk_score,
                "risk_label": risk_label,
                "risk_factors": risk_factors,
            }
            (incident_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            conn.execute(
                """
                INSERT INTO security_alerts (
                    timestamp, unix_time, camera_id, global_id, alert_type,
                    zone_id, zone_name, bbox, footprint, snapshot_path, crop_path, incident_dir,
                    image_hash, previous_hash, block_hash, direction, risk_score, risk_label, risk_factors
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    img_hash,
                    prev_hash,
                    chain_hash,
                    direction,
                    risk_score,
                    risk_label,
                    json.dumps(risk_factors),
                ),
            )
            conn.execute(
                """
                INSERT INTO security_ledger (
                    timestamp, camera_id, global_id, event_type, snapshot_path,
                    image_hash, previous_hash, block_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_str,
                    str(camera_id),
                    gid,
                    alert_type,
                    str(filepath),
                    img_hash,
                    prev_hash,
                    chain_hash,
                ),
            )
            conn.commit()

        LOGGER.warning(
            "[SECURE LEDGER] %s G-ID %s on %s. Block %s...",
            alert_type,
            gid,
            camera_id,
            chain_hash[:8],
        )
        return str(filepath)

    def log_secure_event(
        self,
        camera_id: str,
        global_id: int,
        event_type: str,
        frame: np.ndarray,
        bbox: Sequence[float] = (0, 0, 1, 1),
        timestamp: Optional[float] = None,
    ) -> str:
        return self.log_intrusion(
            camera_id=camera_id,
            global_id=global_id,
            frame=frame,
            bbox=bbox,
            timestamp=timestamp,
            alert_type=event_type,
        )

    def verify_ledger(self) -> Dict[str, Any]:
        """Recompute image and block hashes. Broken links expose tampering."""
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM security_ledger ORDER BY id ASC").fetchall()
        if not rows:
            return {"ok": False, "message": "TAMPER DETECTED: ledger is empty."}

        genesis = rows[0]
        if str(genesis["event_type"]) != "GENESIS" or str(genesis["block_hash"]) != GENESIS_HASH:
            return {"ok": False, "message": "TAMPER DETECTED: genesis block is invalid."}

        for index in range(1, len(rows)):
            prev_row = rows[index - 1]
            curr_row = rows[index]
            snapshot = str(curr_row["snapshot_path"])
            if snapshot not in ("", "none"):
                path = Path(snapshot)
                if not path.exists():
                    return {
                        "ok": False,
                        "message": f"TAMPER DETECTED: image file {snapshot} is missing.",
                        "row_id": int(curr_row["id"]),
                    }
                current_file_hash = self.file_hash(path)
                if current_file_hash != str(curr_row["image_hash"]):
                    return {
                        "ok": False,
                        "message": f"TAMPER DETECTED: image file {snapshot} was modified.",
                        "row_id": int(curr_row["id"]),
                    }
            if str(curr_row["previous_hash"]) != str(prev_row["block_hash"]):
                return {
                    "ok": False,
                    "message": (
                        f"TAMPER DETECTED: chain broken between ID {prev_row['id']} "
                        f"and {curr_row['id']}."
                    ),
                    "row_id": int(curr_row["id"]),
                }
            recalculated = self.block_hash(
                str(curr_row["timestamp"]),
                str(curr_row["camera_id"]),
                int(curr_row["global_id"]),
                str(curr_row["event_type"]),
                str(curr_row["image_hash"]),
                str(curr_row["previous_hash"]),
            )
            if recalculated != str(curr_row["block_hash"]):
                return {
                    "ok": False,
                    "message": f"TAMPER DETECTED: database row ID {curr_row['id']} was altered.",
                    "row_id": int(curr_row["id"]),
                }
        return {
            "ok": True,
            "message": "SECURE: Ledger is intact. No tampering detected.",
            "blocks": len(rows),
        }

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
