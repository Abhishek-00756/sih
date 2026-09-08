#!/usr/bin/env python3
"""Click-to-define geofence polygons on a camera still."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from geofence.editor import ZoneEditor
from geofence.simulator import make_background
from geofence.zones import ZoneStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive geofence zone editor")
    p.add_argument("--image", default="", help="camera still; blank uses simulated frame")
    p.add_argument("--camera-id", default="cam_01")
    p.add_argument("--out", default=str(ROOT / "configs" / "zones.json"))
    p.add_argument("--existing", default="", help="optional JSON to edit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(args.image)
    else:
        frame = make_background()
    store = ZoneStore.load(args.existing) if args.existing else ZoneStore()
    editor = ZoneEditor(camera_id=args.camera_id, zone_store=store)
    editor.run(frame, save_path=args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
