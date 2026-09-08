"""OpenCV overlay of CCTV frame, zones, tracks, ground points, and events."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from geofence.engine import ObjectOverlay
from geofence.models import EventType, GeofenceEvent, GeofenceZone, ZoneType

ZONE_COLORS: Dict[ZoneType, Tuple[int, int, int]] = {
    ZoneType.SAFE: (80, 180, 80),
    ZoneType.WARNING: (0, 200, 255),
    ZoneType.RESTRICTED: (40, 40, 220),
}

BBOX_COLOR = (240, 240, 240)
GROUND_COLOR = (0, 255, 255)
ENTER_COLOR = (0, 80, 255)
EXIT_COLOR = (255, 180, 0)
TEXT_COLOR = (255, 255, 255)


def _draw_filled_poly(frame, pts, color, alpha: float = 0.22) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _label(frame, text: str, org, bg, fg=TEXT_COLOR, scale=0.5, thickness=1) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y + baseline), bg, -1)
    cv2.putText(frame, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


class GeofenceVisualizer:
    def __init__(self, show_raw_pending: bool = True):
        self.show_raw_pending = show_raw_pending

    def draw(
        self,
        frame: np.ndarray,
        zones: Sequence[GeofenceZone],
        overlays: Sequence[ObjectOverlay],
        events: Optional[Sequence[GeofenceEvent]] = None,
        camera_id: Optional[str] = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        self._draw_zones(canvas, zones, camera_id)
        for item in overlays:
            self._draw_overlay(canvas, item)
        banner_events = list(events or [])
        if not banner_events:
            banner_events = [o.latest_event for o in overlays if o.latest_event is not None]
        self._draw_event_banner(canvas, banner_events)
        self._draw_legend(canvas)
        return canvas

    def _draw_zones(self, frame, zones: Sequence[GeofenceZone], camera_id: Optional[str]) -> None:
        h, w = frame.shape[:2]
        for zone in zones:
            if camera_id is not None and zone.camera_id != camera_id:
                continue
            color = ZONE_COLORS.get(zone.zone_type, (180, 180, 180))
            pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            alpha = 0.28 if zone.enabled else 0.08
            _draw_filled_poly(frame, pts, color, alpha=alpha)
            thickness = 2 if zone.enabled else 1
            cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
            label_pt = (int(zone.polygon[0][0]), max(18, int(zone.polygon[0][1]) - 8))
            state = "ON" if zone.enabled else "OFF"
            _label(frame, f"{zone.name} [{zone.zone_type.value}] {state}", label_pt, color)

        cv2.putText(
            frame,
            "Ground point = bbox bottom-center (approx). Pixels are not meters/GPS.",
            (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_overlay(self, frame, item: ObjectOverlay) -> None:
        x1, y1, x2, y2 = [int(v) for v in item.bbox]
        color = BBOX_COLOR
        if item.latest_event is not None:
            color = ENTER_COLOR if item.latest_event.event_type == EventType.GEOFENCE_ENTER else EXIT_COLOR
        elif item.confirmed_zone_ids:
            color = (0, 140, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        gx, gy = int(item.ground_contact[0]), int(item.ground_contact[1])
        cv2.circle(frame, (gx, gy), 5, GROUND_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (gx, gy), 8, GROUND_COLOR, 1, cv2.LINE_AA)
        cv2.line(frame, ((x1 + x2) // 2, y2), (gx, gy), GROUND_COLOR, 1, cv2.LINE_AA)
        title = f"ID {item.object_id} {item.object_type} {item.confidence:.2f}"
        _label(frame, title, (x1, max(18, y1 - 4)), (30, 30, 30))
        _label(frame, item.status_label, (x1, min(frame.shape[0] - 4, y2 + 18)), (40, 90, 40))

    def _draw_event_banner(self, frame, events: Sequence[GeofenceEvent]) -> None:
        y = 24
        for ev in events:
            color = ENTER_COLOR if ev.event_type == EventType.GEOFENCE_ENTER else EXIT_COLOR
            text = (
                f"{ev.event_type.value}  cam={ev.camera_id}  obj={ev.object_id} "
                f"zone={ev.zone_name}/{ev.zone_type.value}"
            )
            _label(frame, text, (8, y), color, scale=0.52)
            y += 24

    def _draw_legend(self, frame) -> None:
        x, y = frame.shape[1] - 210, 18
        items = [
            (ZONE_COLORS[ZoneType.RESTRICTED], "restricted"),
            (ZONE_COLORS[ZoneType.WARNING], "warning"),
            (ZONE_COLORS[ZoneType.SAFE], "safe"),
            (GROUND_COLOR, "ground point"),
        ]
        for color, name in items:
            cv2.rectangle(frame, (x, y - 10), (x + 14, y + 4), color, -1)
            cv2.putText(frame, name, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
            y += 18
