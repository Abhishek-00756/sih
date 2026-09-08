"""Per-track / per-zone persistent spatial state.

Object IDs are camera-specific. Key is always (camera_id, object_id, zone_id).
Temporary tracking loss does not immediately delete state; stale records are
removed after `stale_timeout_sec`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from geofence.models import TrackedObject, TrackZoneKey, TrackZoneState


StateKey = Tuple[str, str, str]


def make_key(camera_id: str, object_id: str, zone_id: str) -> StateKey:
    return (str(camera_id), str(object_id), str(zone_id))


class StateManager:
    def __init__(self, stale_timeout_sec: float = 5.0):
        self.stale_timeout_sec = float(stale_timeout_sec)
        self._states: Dict[StateKey, TrackZoneState] = {}

    def get(self, camera_id: str, object_id: str, zone_id: str) -> Optional[TrackZoneState]:
        return self._states.get(make_key(camera_id, object_id, zone_id))

    def get_or_create(
        self,
        camera_id: str,
        object_id: str,
        zone_id: str,
        object_type: str = "",
        timestamp: float = 0.0,
    ) -> TrackZoneState:
        key = make_key(camera_id, object_id, zone_id)
        state = self._states.get(key)
        if state is None:
            state = TrackZoneState(
                camera_id=str(camera_id),
                object_id=str(object_id),
                zone_id=str(zone_id),
                object_type=object_type,
                last_seen=timestamp,
            )
            self._states[key] = state
        return state

    def touch_from_track(self, state: TrackZoneState, track: TrackedObject, ground_point, raw_inside: bool) -> None:
        state.object_type = track.object_type
        state.last_seen = track.timestamp
        state.last_ground_point = ground_point
        state.last_bbox = track.bbox
        state.last_confidence = track.confidence
        state.last_raw_inside = raw_inside

    def all_states(self) -> List[TrackZoneState]:
        return list(self._states.values())

    def states_for_object(self, camera_id: str, object_id: str) -> List[TrackZoneState]:
        cam = str(camera_id)
        oid = str(object_id)
        return [s for s in self._states.values() if s.camera_id == cam and s.object_id == oid]

    def prune_stale(self, now: float) -> List[TrackZoneState]:
        """Drop states not seen for stale_timeout_sec. Returns removed records."""
        removed: List[TrackZoneState] = []
        cutoff = now - self.stale_timeout_sec
        stale_keys = [k for k, s in self._states.items() if s.last_seen < cutoff]
        for key in stale_keys:
            removed.append(self._states.pop(key))
        return removed

    def reset(self) -> None:
        self._states.clear()

    def __len__(self) -> int:
        return len(self._states)
