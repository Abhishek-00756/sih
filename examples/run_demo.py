#!/usr/bin/env python3
"""Run a simulated CCTV geofence demo (no YOLOX required)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from geofence.engine import EngineConfig, GeofenceEngine
from geofence.simulator import make_background, synthetic_tracks
from geofence.visualizer import GeofenceVisualizer
from geofence.zones import ZoneStore, default_demo_zones


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geofence CCTV demo")
    p.add_argument("--zones", default=str(ROOT / "configs" / "zones.json"))
    p.add_argument("--camera-id", default="cam_01")
    p.add_argument("--confirm-frames", type=int, default=3)
    p.add_argument("--stale-timeout", type=float, default=2.0)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--headless", action="store_true", help="no GUI; write one preview image")
    p.add_argument("--out", default=str(ROOT / "assets" / "demo_preview.jpg"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    zone_path = Path(args.zones)
    if zone_path.exists():
        store = ZoneStore.load(zone_path)
    else:
        store = default_demo_zones(args.camera_id)

    engine = GeofenceEngine(
        store,
        config=EngineConfig(
            confirm_frames=args.confirm_frames,
            stale_timeout_sec=args.stale_timeout,
        ),
    )
    vis = GeofenceVisualizer()
    bg = make_background()
    dt = 1.0 / max(args.fps, 1.0)
    t0 = time.time()
    last_canvas = bg

    print("frame | events")
    for i in range(args.frames):
        ts = t0 + i * dt
        tracks = synthetic_tracks(i, ts, camera_id=args.camera_id)
        events = engine.process_frame(tracks, camera_id=args.camera_id, timestamp=ts)
        overlays = engine.overlays_for_tracks(tracks)
        last_canvas = vis.draw(
            bg,
            store.for_camera(args.camera_id, enabled_only=False),
            overlays,
            events,
            camera_id=args.camera_id,
        )
        for ev in events:
            print(f"{i:5d} | {json.dumps(ev.to_dict())}")
        if not args.headless:
            cv2.imshow("geofence-demo", last_canvas)
            if cv2.waitKey(int(dt * 1000)) & 0xFF in (27, ord("q")):
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), last_canvas)
    print(f"wrote preview {out}")
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
