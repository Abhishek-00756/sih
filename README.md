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
  -> CLAHE night/fog enhance        enhancer.py
  -> Virtual tripwire (IN/OUT)      geofence_manager.py
  -> Context risk score (0-100)     risk_scorer.py
  -> Dwell-time loitering           activity_analyzer.py
  -> Vehicle crop OCR (ANPR)        anpr_manager.py
  -> Haar face mugshot (one/GID)    face_manager.py
  -> SQLite evidence + SHA-256 chain alert_logger.py
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

Live RTSP stays real-time: `ThreadedCamera` always exposes the newest decoded frame and drops OpenCV's internal backlog. YOLOv8 tracks COCO classes `[0, 2, 3, 5, 7]` (person, car, motorcycle, bus, truck). Persons get OSNet Global IDs and geofence checks; vehicles keep ByteTrack IDs (`V-<id>`) and skip Re-ID. Nearby vehicle crops (bbox larger than `anpr_min_width` x `anpr_min_height`, default 150px) are passed to EasyOCR once per track; recognized plates replace the overlay label and are cached in `known_plates`. Person body crops are scanned with OpenCV Haar Cascade; the first frontal face per Global ID is saved under `face_database/`. `ActivityAnalyzer` flags loitering once dwell time exceeds `dwell_threshold` (default 30s). Footprint intrusion uses the bbox bottom-center and rate-limits alerts per Global ID (`geofence_cooldown`). CLAHE (`enhancer.py`) lifts local contrast on dark/foggy frames before detection (`--no-enhance` to skip). `VirtualTripwire` labels crossings `INBOUND` or `OUTBOUND` from the line cross product. `RiskScorer` assigns 0-100 (base 10, +40 inbound, +20 night 20:00-06:00 **IST**, +30 loiter >60s) so C2 can rank alerts. Each alert writes a timestamped full-frame + crop to `alert_snapshots/`, a row in `border_alerts.db`, and a SHA-256 hash-chained ledger block. Pass `--no-anpr` to skip OCR and `--no-face-capture` to skip mugshots. Verify evidence integrity with `python3 main.py --verify-ledger`.

Enable geofence events on Global IDs:

```bash
python3 main.py --headless --enable-geofence --config configs/perception.yaml
```

RTSP example:

```bash
python3 main.py --display --enable-geofence --source Sector_Alpha=rtsp://admin:pass@10.0.0.8/stream1
```

Query recent incidents:

```bash
sqlite3 border_alerts.db "SELECT id, timestamp, camera_id, global_id, risk_score, risk_label, direction, snapshot_path FROM security_alerts ORDER BY id DESC LIMIT 10;"
```

Each intrusion creates a dated incident folder under `alert_snapshots/` with `full.jpg`, `crop.jpg`, and `meta.json`. Open the C2 review console to search by Global ID / camera:

```bash
python3 c2_console.py --port 8080
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

This standalone `geofence/` package provides geometry, zone state, confirmation, mapping, and event generation. The top-level surveillance pipeline additionally provides detection, tracking, risk scoring, evidence storage, and SHA-256 ledger functionality.

## Important limitations

* Ground-contact is the **bottom-center of the bounding box** `((x1+x2)/2, y2)`. That is an approximation of where a person/vehicle meets the ground, not a true 3D foot position.
* All geometry is in **image/pixel coordinates**. Pixels are not meters and are not GPS.
* Homography can be added later as an optional mapping layer. It is **not** required by the MVP.
