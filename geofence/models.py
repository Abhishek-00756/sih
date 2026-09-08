"""Shared data types for the geofencing module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

Point2D = Tuple[float, float]
BoundingBox = Tuple[float, float, float, float]


class ZoneType(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    RESTRICTED = "restricted"


class EventType(str, Enum):
    GEOFENCE_ENTER = "GEOFENCE_ENTER"
    GEOFENCE_EXIT = "GEOFENCE_EXIT"


class SpatialStatus(str, Enum):
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"


def normalize_object_id(object_id: Union[int, str]) -> str:
    return str(object_id)


def normalize_bbox(bbox: Sequence[float]) -> BoundingBox:
    if len(bbox) != 4:
        raise ValueError("bounding box must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class TrackedObject:
    """Standardized tracking record from an upstream detector/tracker."""

    camera_id: str
    object_id: str
    object_type: str
    bbox: BoundingBox
    confidence: float
    timestamp: float

    @staticmethod
    def from_raw(
        camera_id: str,
        object_id: Union[int, str],
        object_type: str,
        bbox: Sequence[float],
        confidence: float,
        timestamp: float,
    ) -> "TrackedObject":
        return TrackedObject(
            camera_id=str(camera_id),
            object_id=normalize_object_id(object_id),
            object_type=str(object_type),
            bbox=normalize_bbox(bbox),
            confidence=float(confidence),
            timestamp=float(timestamp),
        )


@dataclass
class GeofenceZone:
    zone_id: str
    camera_id: str
    name: str
    zone_type: ZoneType
    polygon: List[Point2D]
    enabled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.zone_type, str):
            self.zone_type = ZoneType(self.zone_type.lower())
        self.polygon = [(float(x), float(y)) for x, y in self.polygon]
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.zone_id} needs at least 3 polygon vertices")

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "camera_id": self.camera_id,
            "name": self.name,
            "zone_type": self.zone_type.value,
            "polygon": [[x, y] for x, y in self.polygon],
            "enabled": self.enabled,
        }

    @staticmethod
    def from_dict(data: dict) -> "GeofenceZone":
        polygon = data.get("polygon") or data.get("polygon_coordinates")
        if polygon is None:
            raise ValueError("zone config missing polygon coordinates")
        return GeofenceZone(
            zone_id=str(data["zone_id"]),
            camera_id=str(data["camera_id"]),
            name=str(data.get("name") or data.get("zone_name") or data["zone_id"]),
            zone_type=ZoneType(str(data.get("zone_type", "restricted")).lower()),
            polygon=[(float(p[0]), float(p[1])) for p in polygon],
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class GeofenceEvent:
    event_type: EventType
    camera_id: str
    object_id: str
    object_type: str
    zone_id: str
    zone_name: str
    zone_type: ZoneType
    timestamp: float
    bounding_box: BoundingBox
    ground_contact_position: Point2D
    confidence: float
    confirmed_status: SpatialStatus = SpatialStatus.INSIDE

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "camera_id": self.camera_id,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "zone_type": self.zone_type.value,
            "timestamp": self.timestamp,
            "bounding_box": list(self.bounding_box),
            "ground_contact_position": list(self.ground_contact_position),
            "confidence": self.confidence,
            "confirmed_status": self.confirmed_status.value,
        }


@dataclass
class Observation:
    """One processed sighting of a tracked object against one zone."""

    track: TrackedObject
    zone: GeofenceZone
    ground_contact: Point2D
    raw_inside: bool
    timestamp: float


@dataclass
class TrackZoneKey:
    camera_id: str
    object_id: str
    zone_id: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.camera_id, self.object_id, self.zone_id)


@dataclass
class TrackZoneState:
    """Persistent spatial state for camera_id + object_id + zone_id."""

    camera_id: str
    object_id: str
    zone_id: str
    object_type: str = ""
    confirmed_inside: bool = False
    consecutive_inside: int = 0
    consecutive_outside: int = 0
    last_seen: float = 0.0
    last_ground_point: Optional[Point2D] = None
    last_bbox: Optional[BoundingBox] = None
    last_confidence: float = 0.0
    last_raw_inside: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def confirmed_status(self) -> SpatialStatus:
        return SpatialStatus.INSIDE if self.confirmed_inside else SpatialStatus.OUTSIDE

    @property
    def key(self) -> TrackZoneKey:
        return TrackZoneKey(self.camera_id, self.object_id, self.zone_id)
