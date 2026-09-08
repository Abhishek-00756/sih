"""Temporal confirmation for ENTER/EXIT transitions.

An object must remain inside (or outside) for `confirm_frames` consecutive
frames before a transition is confirmed. Remaining INSIDE or OUTSIDE after
confirmation does not emit another event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from geofence.models import EventType, TrackZoneState


@dataclass
class ConfirmationResult:
    event_type: Optional[EventType]
    confirmed_inside: bool
    changed: bool


class TemporalConfirmer:
    def __init__(self, confirm_frames: int = 3):
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        self.confirm_frames = int(confirm_frames)

    def update(self, state: TrackZoneState, raw_inside: bool) -> ConfirmationResult:
        if raw_inside:
            state.consecutive_inside += 1
            state.consecutive_outside = 0
        else:
            state.consecutive_outside += 1
            state.consecutive_inside = 0

        event: Optional[EventType] = None
        was_inside = state.confirmed_inside

        if not was_inside and raw_inside and state.consecutive_inside >= self.confirm_frames:
            state.confirmed_inside = True
            event = EventType.GEOFENCE_ENTER
        elif was_inside and (not raw_inside) and state.consecutive_outside >= self.confirm_frames:
            state.confirmed_inside = False
            event = EventType.GEOFENCE_EXIT

        return ConfirmationResult(
            event_type=event,
            confirmed_inside=state.confirmed_inside,
            changed=event is not None,
        )
