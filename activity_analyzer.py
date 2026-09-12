"""Suspicious dwell-time / loitering detector for persons and vehicles.

Tracks first-seen time per (camera, entity). When dwell exceeds a configurable
threshold, operators get an orange overlay and a one-shot SQLite evidence row.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

from alert_logger import AlertLogger

LOGGER = logging.getLogger("perception.activity")

EntityId = Union[int, str]
EntityKey = Tuple[str, str]

COCO_CLASSES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

TRACK_CLASSES = (0, 2, 3, 5, 7)
PERSON_CLASS = 0
VEHICLE_CLASSES = (2, 3, 5, 7)


def numeric_entity_id(entity_id: EntityId) -> int:
    """Map person Global IDs and vehicle track tokens onto SQLite INTEGER."""
    if isinstance(entity_id, (int, np.integer)):
        return int(entity_id)
    text = str(entity_id)
    if text.startswith("V-"):
        rest = text[2:]
        if rest.isdigit():
            return 10_000_000 + int(rest)
    if text.lstrip("-").isdigit():
        return int(text)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return 10_000_000 + int(digits)
    total = 0
    for byte in text.encode("utf-8"):
        total = (total * 256 + byte) % 1_000_000_000
    return total


def _footprint(bbox: Sequence[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return (int((x1 + x2) / 2.0), int(y2))


class ActivityAnalyzer:
    def __init__(
        self,
        dwell_threshold: float = 30.0,
        logger: Optional[AlertLogger] = None,
    ) -> None:
        if dwell_threshold <= 0:
            raise ValueError("dwell_threshold must be positive")
        self.dwell_threshold = float(dwell_threshold)
        self.active_entities: Dict[EntityKey, Dict[str, object]] = {}
        self.last_alerts: list[Dict[str, object]] = []
        self.logger = logger if logger is not None else AlertLogger()

    def _key(self, camera_id: str, entity_id: EntityId) -> EntityKey:
        return (str(camera_id), str(entity_id))

    def analyze_behavior(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        entity_id: EntityId,
        entity_type: str = "Person",
        camera_id: str = "Sector_Alpha",
        timestamp: Optional[float] = None,
        draw: bool = True,
    ) -> np.ndarray:
        current_time = time.time() if timestamp is None else float(timestamp)
        key = self._key(camera_id, entity_id)
        record = self.active_entities.get(key)
        if record is None:
            record = {"first_seen": current_time, "alerted": False}
            self.active_entities[key] = record

        dwell_time = current_time - float(record["first_seen"])
        if dwell_time <= self.dwell_threshold:
            return frame

        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            warning_text = f"SUSPICIOUS: {entity_type} {entity_id} (Dwell: {int(dwell_time)}s)"
            cv2.putText(
                frame,
                warning_text,
                (x1, max(20, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        if not record["alerted"]:
            record["alerted"] = True
            LOGGER.warning(
                "[WARNING] Loitering Detected: %s %s at %s (%.1fs).",
                entity_type,
                entity_id,
                camera_id,
                dwell_time,
            )
            self.last_alerts.append(
                {
                    "camera_id": str(camera_id),
                    "entity_id": entity_id,
                    "entity_type": str(entity_type),
                    "dwell_time": dwell_time,
                    "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    "timestamp": current_time,
                }
            )
            if self.logger is not None:
                self.logger.log_intrusion(
                    camera_id=camera_id,
                    global_id=numeric_entity_id(entity_id),
                    frame=frame,
                    bbox=bbox,
                    timestamp=current_time,
                    alert_type="Loitering",
                    zone_name=str(entity_type),
                    footprint=_footprint(bbox),
                )
        return frame

    def pop_alerts(self) -> list[Dict[str, object]]:
        alerts = list(self.last_alerts)
        self.last_alerts.clear()
        return alerts

    def cleanup_stale_tracks(
        self,
        current_active_ids: Iterable[EntityId],
        camera_id: Optional[str] = None,
    ) -> int:
        """Drop IDs that left this camera FOV so dwell clocks reset on re-entry."""
        active: Set[str] = {str(eid) for eid in current_active_ids}
        stale: list[EntityKey] = []
        for key in self.active_entities:
            cam, eid = key
            if camera_id is not None and cam != str(camera_id):
                continue
            if eid not in active:
                stale.append(key)
        for key in stale:
            del self.active_entities[key]
        return len(stale)
