# Shared Perception and Multi-Camera Re-ID

Production starting blueprint for border CCTV: multiple cameras, local person tracks, OSNet body embeddings, and a centralized gallery that assigns a persistent **Global ID**.

```text
IP camera / RTSP
  -> ThreadedCamera (latest frame)  video_stream.py
  -> YOLOv8 person detect
  -> ByteTrack local IDs
  -> OSNet 512-d embedding          extractor.py
  -> Spatio-temporal gallery        gallery_manager.py
  -> Persistent Global ID
  -> Footprint geofence overlay     geofence_manager.py
  -> Optional confirmed events      geofence/
```

## Perception setup

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

If `torchreid` is missing (needed for production OSNet, not for gallery unit tests):

```bash
python3 -m pip install --break-system-packages git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Place feeds at `videos/camera1.mp4` and `videos/camera2.mp4`, or pass sources on the CLI. Webcam indices and RTSP URLs also work.

```bash
python3 main.py --headless --source Cam_1_Outpost=videos/camera1.mp4 --source Cam_2_Gate=videos/camera2.mp4
```

Live RTSP stays real-time: `ThreadedCamera` always exposes the newest decoded frame and drops OpenCV's internal backlog. Footprint intrusion uses the bbox bottom-center and rate-limits alerts per Global ID (`geofence_cooldown`).

Enable geofence events on Global IDs:

```bash
python3 main.py --headless --enable-geofence --config configs/perception.yaml
```

RTSP example:

```bash
python3 main.py --display --enable-geofence --source Sector_Alpha=rtsp://admin:pass@10.0.0.8/stream1
```

Gallery-only tests (no GPU, no OSNet weights):

```bash
python3 -m pytest tests/test_gallery_manager.py tests/test_extractor.py -q
```

Key knobs live in `configs/perception.yaml`: cosine threshold, max appearance window, camera topology (min/max transit time), and YOLO/OSNet names.

Local ByteTrack IDs stay camera-specific. Only the gallery Global ID is shared across cameras.

---

# CCTV Geofencing Module

Pixel-space geofencing for an AI CCTV border-surveillance prototype. Consumes tracked objects from an existing **YOLOX + ByteTrack** pipeline and emits confirmed `GEOFENCE_ENTER` / `GEOFENCE_EXIT` events.

This module does **not** implement detection, tracking, blockchain, encryption, JWT, risk scoring, or evidence storage.

## Important limitations

* Ground-contact is the **bottom-center of the bounding box** `((x1+x2)/2, y2)`. That is an approximation of where a person/vehicle meets the ground, not a true 3D foot position.
* All geometry is in **image/pixel coordinates**. Pixels are not meters and are not GPS.
* Homography can be added later as an optional mapping layer. It is **not** required by the MVP.

## Architecture

```text
CCTV / IP Camera
  -> YOLOX
  -> ByteTrack
  -> Tracking Interface          geofence/adapter.py
  -> Ground-Contact Point        geofence/geometry.py
  -> Optional Coordinate Mapping geofence/mapping.py   (identity in MVP)
  -> Polygon Point-in-Polygon    geofence/geometry.py
  -> Per-Track / Per-Zone State  geofence/state.py
  -> Temporal Confirmation       geofence/confirmation.py
  -> Geofence Event Generator    geofence/events.py
  -> Event / Risk Engine         (your downstream system)
```

Object IDs are **camera-specific**. Track `17` on `cam_01` is not track `17` on `cam_02`.

## Project structure

```text
geofence/
  models.py          TrackedObject, GeofenceZone, GeofenceEvent
  geometry.py        bottom-center ground point, Shapely PIP
  mapping.py         identity mapper + optional homography stub
  zones.py           JSON zone store (multi-camera, multi-zone)
  state.py           persistent (camera_id, object_id, zone_id) state
  confirmation.py    consecutive-frame ENTER/EXIT confirmation
  events.py          structured event payloads
  engine.py          wires the pipeline
  adapter.py         YOLOX/ByteTrack field normalization
  visualizer.py      OpenCV overlay
  editor.py          click-to-draw polygon zones
  simulator.py       synthetic tracks for demo/tests
examples/
  run_demo.py
  edit_zones.py
  integrate_bytetrack.py
configs/
  zones.json
  example_tracks.json
  example_event.json
tests/
  test_geofence.py
```

## Setup

Python 3.9+ recommended.

```bash
python3 -m pip install -r requirements.txt
```

On a server without GUI libraries, install `opencv-python-headless` instead of `opencv-python`.

Dependencies: OpenCV, Shapely, NumPy.

## Run the simulated demo

No YOLOX needed. A synthetic person walks across safe / warning / restricted zones.

```bash
python3 examples/run_demo.py --headless
```

With a GUI window:

```bash
python3 examples/run_demo.py
```

Preview image is written to `assets/demo_preview.jpg`.

## Define zones interactively

```bash
python3 examples/edit_zones.py --camera-id cam_01 --out configs/zones.json
```

* Left-click: add vertex
* Right-click: undo
* `1` / `2` / `3`: safe / warning / restricted
* Enter: finish polygon
* `s`: save JSON
* `q` or Esc: quit

Zones stay in config files. They are not hard-coded in the algorithm.

## Example tracking input

```json
{
  "camera_id": "cam_01",
  "object_id": 17,
  "object_type": "person",
  "bbox": [900, 500, 946, 640],
  "confidence": 0.93,
  "timestamp": 1710000001.5
}
```

Aliases accepted: `track_id`, `tlbr` / `xyxy`, `class_name`, `score`, `ts`.

## Example geofence configuration

See `configs/zones.json`. Each zone has `zone_id`, `camera_id`, `name`, `zone_type` (`safe` | `warning` | `restricted`), `polygon` pixel vertices, and `enabled`.

## Example generated event

See `configs/example_event.json`:

```json
{
  "event_type": "GEOFENCE_ENTER",
  "camera_id": "cam_01",
  "object_id": "17",
  "object_type": "person",
  "zone_id": "zone_restricted_gate",
  "zone_name": "Restricted Gate",
  "zone_type": "restricted",
  "timestamp": 1710000001.5,
  "bounding_box": [900.0, 500.0, 946.0, 640.0],
  "ground_contact_position": [923.0, 640.0],
  "confidence": 0.93,
  "confirmed_status": "INSIDE"
}
```

Transitions:

* OUTSIDE -> INSIDE = `GEOFENCE_ENTER`
* INSIDE -> OUTSIDE = `GEOFENCE_EXIT`
* OUTSIDE -> OUTSIDE = no event
* INSIDE -> INSIDE = no event

An object must stay inside/outside for `confirm_frames` consecutive frames (default 3) before the transition is confirmed. Remaining inside does **not** spam alerts.

Temporary tracker dropouts keep last-seen state until `stale_timeout_sec`.

## Connect to a real YOLOX + ByteTrack stream

1. Keep your detector/tracker unchanged.
2. Each frame, convert ByteTrack output into dicts or arrays.
3. Parse with `TrackingInterface`, then `GeofenceEngine.process_frame`.
4. Forward `event.to_dict()` to your event/risk engine.

```python
from geofence.adapter import TrackingInterface
from geofence.engine import EngineConfig, GeofenceEngine
from geofence.visualizer import GeofenceVisualizer
from geofence.zones import ZoneStore

store = ZoneStore.load("configs/zones.json")
engine = GeofenceEngine(store, EngineConfig(confirm_frames=3, stale_timeout_sec=5.0))
iface = TrackingInterface(default_camera_id="cam_01")
vis = GeofenceVisualizer()

# inside your camera loop, after ByteTrack update:
tracks = iface.from_bytetrack_tlbr(
    tlbrs=online_tlbr,          # Nx4 xyxy
    track_ids=online_ids,
    scores=online_scores,
    class_ids=online_classes,
    class_names={0: "person", 2: "vehicle"},
    camera_id="cam_01",
    timestamp=frame_time,
)
events = engine.process_frame(tracks, camera_id="cam_01", timestamp=frame_time)
overlays = engine.overlays_for_tracks(tracks)
annotated = vis.draw(frame, store.for_camera("cam_01"), overlays, events, camera_id="cam_01")

for ev in events:
    publish(ev.to_dict())   # your event engine
```

Dict-style records also work:

```python
tracks = iface.parse_frame(bytetrack_dicts, camera_id="cam_01", timestamp=ts)
```

A full sketch is in `examples/integrate_bytetrack.py`.

## Testing

```bash
python3 -m pip install pytest
python3 -m pytest tests/test_geofence.py -q
```

Coverage includes ground-contact math, PIP, ENTER/EXIT without repeat alerts, temporal confirmation, multi-camera ID isolation, stale-state timeout, zone enable/disable, JSON round-trip, and event payload fields.

## Optional homography (not used by MVP)

`geofence/mapping.py` exposes `HomographyMapper`. Pass it into `GeofenceEngine(..., mapper=...)`. Zones would then be defined in the mapped plane. Leave it unset for pixel-space operation.
