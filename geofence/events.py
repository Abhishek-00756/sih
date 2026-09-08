"""Structured geofence event generation.

This module only produces spatial events. It does not implement blockchain,
encryption, JWT, risk scoring, or evidence storage.
"""

from __future__ import annotations

from typing import List, Optional

from geofence.models import (
    EventType,
    GeofenceEvent,
    GeofenceZone,
    Point2D,
    SpatialStatus,
    TrackedObject,
    TrackZoneState,
)


class EventGenerator:
    def make_event(
        self,
        event_type: EventType,
        track: TrackedObject,
        zone: GeofenceZone,
        ground_contact: Point2D,
        confirmed_inside: bool,
    ) -> GeofenceEvent:
        return GeofenceEvent(
            event_type=event_type,
            camera_id=track.camera_id,
            object_id=track.object_id,
            object_type=track.object_type,
            zone_id=zone.zone_id,
            zone_name=zone.name,
            zone_type=zone.zone_type,
            timestamp=track.timestamp,
            bounding_box=track.bbox,
            ground_contact_position=ground_contact,
            confidence=track.confidence,
            confirmed_status=SpatialStatus.INSIDE if confirmed_inside else SpatialStatus.OUTSIDE,
        )

    def from_state(
        self,
        event_type: EventType,
        state: TrackZoneState,
        zone: GeofenceZone,
        track: Optional[TrackedObject] = None,
        ground_contact: Optional[Point2D] = None,
    ) -> GeofenceEvent:
        if track is not None:
            return self.make_event(
                event_type,
                track,
                zone,
                ground_contact or state.last_ground_point or (0.0, 0.0),
                state.confirmed_inside,
            )
        bbox = state.last_bbox or (0.0, 0.0, 0.0, 0.0)
        point = ground_contact or state.last_ground_point or (0.0, 0.0)
        synthetic = TrackedObject(
            camera_id=state.camera_id,
            object_id=state.object_id,
            object_type=state.object_type,
            bbox=bbox,
            confidence=state.last_confidence,
            timestamp=state.last_seen,
        )
        return self.make_event(event_type, synthetic, zone, point, state.confirmed_inside)


def events_to_dicts(events: List[GeofenceEvent]) -> List[dict]:
    return [e.to_dict() for e in events]
