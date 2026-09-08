"""Zone configuration store. Zones are data, not algorithm constants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from geofence.models import GeofenceZone, ZoneType


class ZoneStore:
    """Holds multiple geofences per camera. Load/save as JSON."""

    def __init__(self, zones: Optional[Iterable[GeofenceZone]] = None):
        self._zones: Dict[str, GeofenceZone] = {}
        if zones:
            for zone in zones:
                self.add(zone)

    def add(self, zone: GeofenceZone) -> None:
        self._zones[zone.zone_id] = zone

    def remove(self, zone_id: str) -> None:
        self._zones.pop(zone_id, None)

    def get(self, zone_id: str) -> Optional[GeofenceZone]:
        return self._zones.get(zone_id)

    def update_enabled(self, zone_id: str, enabled: bool) -> None:
        zone = self._zones.get(zone_id)
        if zone is None:
            raise KeyError(f"unknown zone_id: {zone_id}")
        zone.enabled = enabled

    def all_zones(self) -> List[GeofenceZone]:
        return list(self._zones.values())

    def for_camera(self, camera_id: str, enabled_only: bool = True) -> List[GeofenceZone]:
        out = []
        for zone in self._zones.values():
            if zone.camera_id != camera_id:
                continue
            if enabled_only and not zone.enabled:
                continue
            out.append(zone)
        return out

    def cameras(self) -> List[str]:
        return sorted({z.camera_id for z in self._zones.values()})

    def to_dict(self) -> dict:
        return {"zones": [z.to_dict() for z in self._zones.values()]}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ZoneStore":
        raw_zones = data.get("zones", data if isinstance(data, list) else [])
        zones = [GeofenceZone.from_dict(item) for item in raw_zones]
        return cls(zones)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ZoneStore":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)


def default_demo_zones(camera_id: str = "cam_01", frame_size=(1280, 720)) -> ZoneStore:
    """Example zones for a typical 1280x720 frame. Not used by the algorithm itself."""
    w, h = frame_size
    store = ZoneStore()
    store.add(
        GeofenceZone(
            zone_id="zone_restricted_gate",
            camera_id=camera_id,
            name="Restricted Gate",
            zone_type=ZoneType.RESTRICTED,
            polygon=[
                (int(w * 0.55), int(h * 0.45)),
                (int(w * 0.92), int(h * 0.45)),
                (int(w * 0.95), int(h * 0.95)),
                (int(w * 0.52), int(h * 0.95)),
            ],
            enabled=True,
        )
    )
    store.add(
        GeofenceZone(
            zone_id="zone_warning_approach",
            camera_id=camera_id,
            name="Warning Approach",
            zone_type=ZoneType.WARNING,
            polygon=[
                (int(w * 0.28), int(h * 0.40)),
                (int(w * 0.54), int(h * 0.40)),
                (int(w * 0.52), int(h * 0.95)),
                (int(w * 0.18), int(h * 0.95)),
            ],
            enabled=True,
        )
    )
    store.add(
        GeofenceZone(
            zone_id="zone_safe_patrol",
            camera_id=camera_id,
            name="Safe Patrol",
            zone_type=ZoneType.SAFE,
            polygon=[
                (int(w * 0.02), int(h * 0.50)),
                (int(w * 0.26), int(h * 0.50)),
                (int(w * 0.18), int(h * 0.95)),
                (int(w * 0.02), int(h * 0.95)),
            ],
            enabled=True,
        )
    )
    return store
