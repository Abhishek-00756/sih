"""Tracking interface: convert YOLOX + ByteTrack style records into TrackedObject.

This module does not run detection or tracking. It only normalizes fields
commonly produced by those pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from geofence.models import TrackedObject


def _first(data: dict, keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def dict_to_tracked_object(
    data: Dict[str, Any],
    default_camera_id: str = "cam_01",
    default_timestamp: float = 0.0,
) -> TrackedObject:
    camera_id = _first(data, ("camera_id", "cam_id", "camera"), default_camera_id)
    object_id = _first(data, ("object_id", "track_id", "id", "tid"))
    if object_id is None:
        raise ValueError("tracking record missing object_id / track_id")
    object_type = _first(data, ("object_type", "class_name", "cls_name", "label", "class"), "unknown")
    if isinstance(object_type, (int, float)):
        object_type = str(int(object_type))
    bbox = _first(data, ("bbox", "bounding_box", "tlbr", "xyxy"))
    if bbox is None and all(k in data for k in ("x1", "y1", "x2", "y2")):
        bbox = [data["x1"], data["y1"], data["x2"], data["y2"]]
    if bbox is None:
        raise ValueError("tracking record missing bounding box")
    confidence = float(_first(data, ("confidence", "score", "conf", "prob"), 0.0))
    timestamp = float(_first(data, ("timestamp", "ts", "time"), default_timestamp))
    return TrackedObject.from_raw(
        camera_id=camera_id,
        object_id=object_id,
        object_type=object_type,
        bbox=bbox,
        confidence=confidence,
        timestamp=timestamp,
    )


class TrackingInterface:
    """Normalizes one frame of tracker output into TrackedObject list."""

    def __init__(self, default_camera_id: str = "cam_01"):
        self.default_camera_id = default_camera_id

    def parse_frame(
        self,
        records: Iterable[Union[TrackedObject, Dict[str, Any]]],
        camera_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        cam = camera_id or self.default_camera_id
        ts = 0.0 if timestamp is None else float(timestamp)
        out: List[TrackedObject] = []
        for rec in records:
            if isinstance(rec, TrackedObject):
                out.append(rec)
                continue
            out.append(dict_to_tracked_object(rec, default_camera_id=cam, default_timestamp=ts))
        return out

    def from_bytetrack_tlbr(
        self,
        tlbrs: Sequence[Sequence[float]],
        track_ids: Sequence[Union[int, str]],
        scores: Sequence[float],
        class_ids: Sequence[Union[int, str]],
        class_names: Optional[Dict[int, str]] = None,
        camera_id: Optional[str] = None,
        timestamp: float = 0.0,
    ) -> List[TrackedObject]:
        """Convert typical ByteTrack online output arrays into TrackedObject rows."""
        cam = camera_id or self.default_camera_id
        names = class_names or {}
        tracks: List[TrackedObject] = []
        for tlbr, tid, score, cid in zip(tlbrs, track_ids, scores, class_ids):
            if isinstance(cid, int):
                label = names.get(cid, str(cid))
            else:
                label = str(cid)
            tracks.append(
                TrackedObject.from_raw(
                    camera_id=cam,
                    object_id=tid,
                    object_type=label,
                    bbox=tlbr,
                    confidence=float(score),
                    timestamp=timestamp,
                )
            )
        return tracks
