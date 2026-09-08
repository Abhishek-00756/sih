"""Pixel-space geofencing for CCTV tracked objects.

This package consumes tracked-object data (e.g. YOLOX + ByteTrack) and
emits ENTER/EXIT events against polygon zones. It does not implement
detection or tracking.

IMPORTANT LIMITATIONS
---------------------
* Ground-contact is approximated as the bottom-center of the bounding box.
  This is not a true 3D foot position.
* All geometry is in image/pixel coordinates. Pixels are not meters and
  are not geographic (GPS) coordinates.
* Homography / world mapping is an optional future layer and is not
  required by the MVP.
"""

from geofence.models import (
    BoundingBox,
    GeofenceEvent,
    GeofenceZone,
    Point2D,
    TrackedObject,
    ZoneType,
    EventType,
    SpatialStatus,
)
from geofence.engine import GeofenceEngine, EngineConfig
from geofence.zones import ZoneStore
from geofence.adapter import TrackingInterface, dict_to_tracked_object

__all__ = [
    "BoundingBox",
    "GeofenceEvent",
    "GeofenceZone",
    "Point2D",
    "TrackedObject",
    "ZoneType",
    "EventType",
    "SpatialStatus",
    "GeofenceEngine",
    "EngineConfig",
    "ZoneStore",
    "TrackingInterface",
    "dict_to_tracked_object",
]

__version__ = "0.1.0"
