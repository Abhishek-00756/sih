"""Geofence engine: wire geometry, mapping, state, confirmation, events.

Logical flow
------------
Tracking Interface
  -> Ground-Contact Point Calculation
  -> Optional Coordinate Mapping (identity in MVP)
  -> Polygon Point-in-Polygon Test
  -> Per-Track / Per-Zone State Manager
  -> Temporal Confirmation
  -> Geofence Event Generator

The engine does not run YOLOX or ByteTrack. It consumes TrackedObject records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from geofence.confirmation import TemporalConfirmer
from geofence.events import EventGenerator
from geofence.geometry import ground_contact_from_bbox, point_in_polygon
from geofence.mapping import CoordinateMapper, IdentityMapper
from geofence.models import (
    EventType,
    GeofenceEvent,
    GeofenceZone,
    Point2D,
    SpatialStatus,
    TrackedObject,
    TrackZoneState,
)
from geofence.state import StateManager
from geofence.zones import ZoneStore


@dataclass
class EngineConfig:
    confirm_frames: int = 3
    stale_timeout_sec: float = 5.0
    emit_exit_on_stale: bool = False


@dataclass
class ObjectOverlay:
    """Per-object snapshot for visualization / downstream UI."""

    camera_id: str
    object_id: str
    object_type: str
    bbox: Tuple[float, float, float, float]
    ground_contact: Point2D
    confidence: float
    timestamp: float
    raw_zone_ids: List[str] = field(default_factory=list)
    confirmed_zone_ids: List[str] = field(default_factory=list)
    latest_event: Optional[GeofenceEvent] = None

    @property
    def status_label(self) -> str:
        if self.confirmed_zone_ids:
            return "INSIDE " + ",".join(self.confirmed_zone_ids)
        if self.raw_zone_ids:
            return "PENDING " + ",".join(self.raw_zone_ids)
        return "OUTSIDE"


class GeofenceEngine:
    def __init__(
        self,
        zone_store: ZoneStore,
        config: Optional[EngineConfig] = None,
        mapper: Optional[CoordinateMapper] = None,
    ):
        self.zones = zone_store
        self.config = config or EngineConfig()
        self.mapper = mapper or IdentityMapper()
        self.state = StateManager(stale_timeout_sec=self.config.stale_timeout_sec)
        self.confirmer = TemporalConfirmer(confirm_frames=self.config.confirm_frames)
        self.event_gen = EventGenerator()
        self._recent_events: List[GeofenceEvent] = []

    def process_frame(
        self,
        tracks: Sequence[TrackedObject],
        camera_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> List[GeofenceEvent]:
        """Process one camera frame of tracked objects. Returns new events only."""
        events: List[GeofenceEvent] = []
        now = timestamp
        if now is None and tracks:
            now = max(t.timestamp for t in tracks)
        if now is None:
            now = 0.0

        for track in tracks:
            if camera_id is not None and track.camera_id != camera_id:
                continue
            events.extend(self._process_track(track))

        stale = self.state.prune_stale(now)
        if self.config.emit_exit_on_stale:
            for record in stale:
                if not record.confirmed_inside:
                    continue
                zone = self.zones.get(record.zone_id)
                if zone is None:
                    continue
                record.confirmed_inside = False
                events.append(
                    self.event_gen.from_state(EventType.GEOFENCE_EXIT, record, zone)
                )

        self._recent_events = events
        return events

    def _process_track(self, track: TrackedObject) -> List[GeofenceEvent]:
        events: List[GeofenceEvent] = []
        pixel_point = ground_contact_from_bbox(track.bbox)
        mapped_point = self.mapper.map_point(pixel_point, track.camera_id)
        zones = self.zones.for_camera(track.camera_id, enabled_only=True)

        for zone in zones:
            raw_inside = point_in_polygon(mapped_point, zone.polygon)
            state = self.state.get_or_create(
                track.camera_id,
                track.object_id,
                zone.zone_id,
                object_type=track.object_type,
                timestamp=track.timestamp,
            )
            self.state.touch_from_track(state, track, mapped_point, raw_inside)
            result = self.confirmer.update(state, raw_inside)
            if result.event_type is not None:
                events.append(
                    self.event_gen.make_event(
                        result.event_type,
                        track,
                        zone,
                        mapped_point,
                        result.confirmed_inside,
                    )
                )
        return events

    def overlays_for_tracks(self, tracks: Sequence[TrackedObject]) -> List[ObjectOverlay]:
        overlays: List[ObjectOverlay] = []
        events_by_obj: Dict[Tuple[str, str], GeofenceEvent] = {}
        for ev in self._recent_events:
            events_by_obj[(ev.camera_id, ev.object_id)] = ev

        for track in tracks:
            pixel_point = ground_contact_from_bbox(track.bbox)
            mapped_point = self.mapper.map_point(pixel_point, track.camera_id)
            states = self.state.states_for_object(track.camera_id, track.object_id)
            raw_ids = [s.zone_id for s in states if s.last_raw_inside]
            confirmed_ids = [s.zone_id for s in states if s.confirmed_inside]
            overlays.append(
                ObjectOverlay(
                    camera_id=track.camera_id,
                    object_id=track.object_id,
                    object_type=track.object_type,
                    bbox=track.bbox,
                    ground_contact=mapped_point,
                    confidence=track.confidence,
                    timestamp=track.timestamp,
                    raw_zone_ids=raw_ids,
                    confirmed_zone_ids=confirmed_ids,
                    latest_event=events_by_obj.get((track.camera_id, track.object_id)),
                )
            )
        return overlays

    def confirmed_status(self, camera_id: str, object_id: str, zone_id: str) -> SpatialStatus:
        state = self.state.get(camera_id, object_id, zone_id)
        if state is None:
            return SpatialStatus.OUTSIDE
        return state.confirmed_status

    def reset(self) -> None:
        self.state.reset()
        self._recent_events = []
