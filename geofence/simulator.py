"""Synthetic tracks and background for demo / tests without YOLOX."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from geofence.models import TrackedObject


def make_background(width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (38, 42, 36)
    cv2.rectangle(frame, (0, int(height * 0.55)), (width, height), (48, 58, 48), -1)
    cv2.line(frame, (0, int(height * 0.55)), (width, int(height * 0.55)), (70, 90, 70), 2)
    cv2.putText(frame, "CCTV BORDER CAM 01", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    cv2.putText(frame, "SIMULATED FEED", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    for x in range(0, width, 80):
        cv2.line(frame, (x, int(height * 0.55)), (x, height), (55, 65, 55), 1)
    return frame


def _person_bbox(cx: float, feet_y: float, w: float = 46, h: float = 120) -> Tuple[float, float, float, float]:
    x1 = cx - w / 2.0
    y2 = feet_y
    y1 = y2 - h
    x2 = cx + w / 2.0
    return (x1, y1, x2, y2)


def synthetic_tracks(frame_idx: int, timestamp: float, camera_id: str = "cam_01") -> List[TrackedObject]:
    """Two objects: one walks into restricted, one stays in safe then exits."""
    t = frame_idx
    tracks: List[TrackedObject] = []

    # Object 17: approaches from left, crosses warning into restricted, then leaves.
    if 0 <= t <= 220:
        cx = 80 + t * 4.2
        feet_y = 620 + 8 * np.sin(t / 9.0)
        tracks.append(
            TrackedObject.from_raw(
                camera_id=camera_id,
                object_id=17,
                object_type="person",
                bbox=_person_bbox(cx, feet_y),
                confidence=0.91,
                timestamp=timestamp,
            )
        )

    # Object 4: patrols the safe zone, briefly lost, then reappears.
    if t < 40 or t > 55:
        cx2 = 90 + 40 * np.sin(t / 18.0)
        feet_y2 = 640
        if t <= 180:
            tracks.append(
                TrackedObject.from_raw(
                    camera_id=camera_id,
                    object_id=4,
                    object_type="person",
                    bbox=_person_bbox(cx2, feet_y2, w=40, h=110),
                    confidence=0.84,
                    timestamp=timestamp,
                )
            )

    # Object 9 on a second camera — same numeric ID must stay isolated if mixed.
    return tracks


def synthetic_tracks_cam02(frame_idx: int, timestamp: float) -> List[TrackedObject]:
    t = frame_idx
    if t > 80:
        return []
    cx = 200 + t * 3.0
    return [
        TrackedObject.from_raw(
            camera_id="cam_02",
            object_id=17,
            object_type="vehicle",
            bbox=(cx - 70, 400, cx + 70, 500),
            confidence=0.77,
            timestamp=timestamp,
        )
    ]
