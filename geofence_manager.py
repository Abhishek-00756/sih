"""Footprint geofence overlay keyed by persistent Global IDs.

Uses the bottom-center of each bounding box as a ground-contact proxy
and OpenCV pointPolygonTest for inside/edge/outside. Alerts are rate-
limited per Global ID so the command center is not flooded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

LOGGER = logging.getLogger("perception.geofence")

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class IntrusionAlert:
    global_id: int
    camera_id: str
    zone_id: str
    zone_name: str
    footprint: Point
    bbox: BBox
    timestamp: float


def footprint_from_bbox(bbox: Sequence[float]) -> Point:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return (int((x1 + x2) / 2.0), int(y2))


def _as_contour(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    if arr.shape[0] < 3:
        raise ValueError("geofence polygon needs at least 3 vertices")
    return arr


class RestrictedZone:
    def __init__(
        self,
        points: Sequence[Sequence[float]],
        zone_id: str = "restricted",
        name: str = "Restricted Zone",
        camera_id: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        self.zone_id = str(zone_id)
        self.name = str(name)
        self.camera_id = None if camera_id is None else str(camera_id)
        self.color = color
        self.polygon = _as_contour(points)

    def contains(self, point: Point) -> bool:
        return float(cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)) >= 0.0

    def applies_to(self, camera_id: str) -> bool:
        return self.camera_id is None or self.camera_id == str(camera_id)

    def draw(self, frame: np.ndarray) -> None:
        cv2.polylines(frame, [self.polygon], True, self.color, 2)
        origin = tuple(int(v) for v in self.polygon[0, 0])
        cv2.putText(
            frame,
            self.name,
            (origin[0], max(20, origin[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.color,
            1,
        )


class GeofenceManager:
    """Per-Global-ID intrusion detector with cooldown."""

    def __init__(
        self,
        zone_points: Optional[Sequence[Sequence[float]]] = None,
        cooldown_seconds: float = 60.0,
        zones: Optional[Iterable[RestrictedZone]] = None,
        logger: Optional[object] = None,
    ) -> None:
        self.cooldown_seconds = float(cooldown_seconds)
        self.zones: List[RestrictedZone] = list(zones or [])
        if zone_points:
            self.zones.append(RestrictedZone(zone_points, zone_id="restricted", name="Restricted Zone"))
        if not self.zones:
            raise ValueError("GeofenceManager requires at least one polygon")
        self.alerted_ids: Dict[Tuple[int, str], float] = {}
        self.last_alerts: List[IntrusionAlert] = []
        self.logger = logger

    @classmethod
    def from_zone_store(
        cls,
        zones_path: Union[str, "Path"],
        cooldown_seconds: float = 60.0,
        restricted_only: bool = True,
    ) -> "GeofenceManager":
        from pathlib import Path

        from geofence.models import ZoneType
        from geofence.zones import ZoneStore

        store = ZoneStore.load(Path(zones_path))
        packed: List[RestrictedZone] = []
        colors = {
            "restricted": (0, 0, 255),
            "warning": (0, 165, 255),
            "safe": (0, 200, 0),
        }
        for zone in store.all(enabled_only=True):
            ztype = zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type)
            if restricted_only and ztype != ZoneType.RESTRICTED.value:
                continue
            packed.append(
                RestrictedZone(
                    zone.polygon,
                    zone_id=zone.zone_id,
                    name=zone.name,
                    camera_id=zone.camera_id,
                    color=colors.get(ztype, (0, 0, 255)),
                )
            )
        if not packed:
            raise ValueError(f"no usable geofence polygons in {zones_path}")
        return cls(cooldown_seconds=cooldown_seconds, zones=packed, logger=None)

    def draw_zones(self, frame: np.ndarray, camera_id: Optional[str] = None) -> None:
        for zone in self.zones:
            if camera_id is not None and not zone.applies_to(camera_id):
                continue
            zone.draw(frame)

    def check_intrusion(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        global_id: int,
        camera_id: str = "",
        timestamp: Optional[float] = None,
        draw: bool = True,
        source_camera_id: Optional[str] = None,
    ) -> np.ndarray:
        """Overlay footprint and emit a cooldown-limited alert if inside a zone."""
        if timestamp is None:
            timestamp = time.time()
        foot = footprint_from_bbox(bbox)
        if draw:
            cv2.circle(frame, foot, 4, (0, 255, 255), -1)

        for zone in self.zones:
            if camera_id and not zone.applies_to(camera_id):
                continue
            if not zone.contains(foot):
                continue
            key = (int(global_id), zone.zone_id)
            last = self.alerted_ids.get(key)
            if last is None or timestamp - last > self.cooldown_seconds:
                self.alerted_ids[key] = timestamp
                alert = IntrusionAlert(
                    global_id=int(global_id),
                    camera_id=str(camera_id),
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    footprint=foot,
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    timestamp=timestamp,
                )
                self.last_alerts.append(alert)
                outpost = source_camera_id or camera_id or "unknown-cam"
                LOGGER.warning(
                    "[ALERT] Intrusion Detected! Global ID: %s entered %s on %s",
                    global_id,
                    zone.name,
                    outpost,
                )
                if self.logger is not None:
                    self.logger.log_intrusion(
                        camera_id=outpost,
                        global_id=int(global_id),
                        frame=frame,
                        bbox=alert.bbox,
                        timestamp=timestamp,
                        zone_id=zone.zone_id,
                        zone_name=zone.name,
                        footprint=foot,
                    )
                if draw:
                    cv2.putText(
                        frame,
                        f"INTRUSION: ID {global_id}",
                        (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
            elif draw:
                cv2.putText(
                    frame,
                    f"ZONE {zone.name}",
                    (int(bbox[0]), min(frame.shape[0] - 8, int(bbox[3]) + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    zone.color,
                    1,
                )
        return frame

    def pop_alerts(self) -> List[IntrusionAlert]:
        alerts = list(self.last_alerts)
        self.last_alerts.clear()
        return alerts
