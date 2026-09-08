#!/usr/bin/env python3
"""Example of wiring YOLOX + ByteTrack output into the geofence engine.

This file is a template. It does not import YOLOX or ByteTrack.
Replace `next_tracker_frame()` with your pipeline's live output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofence.adapter import TrackingInterface
from geofence.engine import EngineConfig, GeofenceEngine
from geofence.zones import ZoneStore


def next_tracker_frame():
    """Yield (frame_bgr, bytetrack_records, timestamp).

    Typical ByteTrack online output per detection:
      tlbr / xyxy, track_id, score, class_id
    """
    raise NotImplementedError("connect your YOLOX+ByteTrack iterator here")


def main() -> None:
    store = ZoneStore.load(ROOT / "configs" / "zones.json")
    engine = GeofenceEngine(store, config=EngineConfig(confirm_frames=3, stale_timeout_sec=5.0))
    interface = TrackingInterface(default_camera_id="cam_01")

    # Live loop sketch:
    # for frame, records, ts in next_tracker_frame():
    #     tracks = interface.parse_frame(records, camera_id="cam_01", timestamp=ts)
    #     events = engine.process_frame(tracks, camera_id="cam_01", timestamp=ts)
    #     for ev in events:
    #         send_to_event_engine(ev.to_dict())

    sample = [
        {
            "camera_id": "cam_01",
            "object_id": 17,
            "object_type": "person",
            "bbox": [900, 480, 960, 640],
            "confidence": 0.93,
            "timestamp": 1710000001.2,
        }
    ]
    tracks = interface.parse_frame(sample)
    events = engine.process_frame(tracks, timestamp=1710000001.2)
    print(json.dumps([e.to_dict() for e in events], indent=2))


if __name__ == "__main__":
    main()
