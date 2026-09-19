# Shared Perception and Multi-Camera Re-ID

Production starting blueprint for border CCTV: multiple cameras, local person tracks, OSNet body embeddings, and a centralized gallery that assigns a persistent **Global ID**.

```text
IP camera / RTSP or paired mobile camera
  -> ThreadedCamera / mobile frame receiver
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
  -> Shared face registry           face_intelligence.py
  -> Evidence + integrity ledger    alert_logger.py / cybersecurity/
```

## Operator dashboard

The dashboard is the operator-facing layer. **ANPR is camera-specific. Face recognition uses one global registry, while camera participation is configurable.** One central face registry is shared by the deployment, and each camera has its own `face_recognition` switch deciding whether that camera sends detections into the shared matcher.

```text
CAM-01 ─┐
CAM-02 ─┼──> Shared person gallery / Global IDs
CAM-03 ─┘
   │
   ├── per-camera: AI processing
   ├── per-camera: geofence
   ├── per-camera: tripwire
   ├── per-camera: ANPR
   ├── per-camera: face-recognition participation
   └── per-camera: enhancement

CAM-01 face crop ─┐
CAM-03 face crop ─┼──> GLOBAL FACE REGISTRY
CAM-02 (OFF)      ┘
```

Camera settings live in `configs/cameras.json`. Each camera has its own AI switch plus geofence/tripwire/ANPR/face-recognition/enhancement flags. The dashboard can display three feeds concurrently while AI is selectively enabled, which avoids forcing the Mac to process all streams simultaneously.

Start the dashboard with:

```bash
python3 dashboard_server.py
```

The server prints the LAN address it is listening on. Open the dashboard at the HTTPS address (for example `https://192.168.x.x:8080`). If you open `http://localhost:8081`, it redirects to the same HTTPS dashboard automatically. The dashboard provides live views, per-camera runtime controls, a shared face-enrollment registry, face-recognition backend status, local evidence-ledger verification, and QR onboarding for browser-based mobile cameras.

### QR mobile-camera pairing

Click **+ ADD CAMERA** in the dashboard and choose a camera slot. BORDER SENTINEL creates a short-lived QR pairing session and shows the QR code. The QR uses the exact host name/IP that the dashboard is using, so there is no manual IP entry. Scan it with a phone on the same LAN/Wi-Fi; the phone opens a secure local camera page, grants browser camera permission, and sends JPEG frames to the paired camera slot.

```text
Desktop dashboard
      |
      | Generate QR
      v
   QR pairing
      |
      | scan
      v
   Phone browser
      |
      | camera frames over HTTPS
      v
CAM-03 / selected slot
      |
      v
Shared perception + dashboard
```

The dashboard uses HTTPS on `8080` for both the desktop dashboard and phone camera page. A small HTTP compatibility server runs on `8081` and redirects to HTTPS. This single-port design avoids common mobile-hotspot port-isolation problems and keeps QR links simple. A development certificate warning may appear on first use; continue to the local page and allow camera access. Pairing tokens expire after five minutes and should not be treated as permanent credentials.

Install the QR dependency with:

```bash
python3 -m pip install --break-system-packages -r requirements-dashboard.txt
```

RTSP and HTTP camera URLs remain supported for direct camera integrations.

### Shared face recognition

The old `face_manager.py` remains a lightweight Haar face-capture utility. The dashboard's actual recognition registry is `face_intelligence.py`. It uses InsightFace when installed and stores enrolled face embeddings under `face_registry/`. Install its optional dependencies with:

```bash
python3 -m pip install --break-system-packages -r requirements-face.txt
```

A person is enrolled once; the registry is not tied to a camera. Camera `face_recognition` flags determine which feeds can participate in matching.

### Evidence integrity / blockchain adapter

`cybersecurity/ledger_anchor.py` provides a small anchoring interface. The current implementation is a local SHA-256 hash chain for immediate tamper-evident evidence integrity. It is intentionally separated so a permissioned blockchain implementation can be added later without changing dashboard evidence payloads.

## Perception setup

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

If `torchreid` is missing (needed for production OSNet, not for gallery unit tests):

```bash
python3 -m pip install --break-system-packages git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Place feeds at `videos/camera1.mp4` and `videos/cam2.mp4`, or pass sources on the CLI. Webcam indices and RTSP URLs also work.

```bash
python3 main.py --headless --source Cam_1_Outpost=videos/camera1.mp4 --source Cam_2_Gate=videos/cam2.mp4
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

Each intrusion creates a dated incident folder under `alert_snapshots/` with `full.jpg`, `crop.jpg`, and `meta.json`. Open the legacy C2 review console to search by Global ID / camera:

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
## Blockchain evidence anchoring

BORDER SENTINEL records every geofence/tripwire intrusion in the local evidence database and tamper-evident SHA-256 chain. It also asynchronously submits the evidence SHA-256 plus incident metadata to the deployed Hyperledger Fabric evidence chaincode on evidencechannel.

For the local Fabric development network:

```bash
export FABRIC_TEST_NETWORK=~/fabric-samples/test-network
export FABRIC_CHANNEL=evidencechannel
export FABRIC_CHAINCODE=evidence
export BORDER_SENTINEL_BLOCKCHAIN_ENABLED=true
```

The Python runtime invokes the Fabric peer CLI with Org1 and Org2 endorsement and --waitForEvent, then stores the Fabric transaction ID and anchoring status back into border_alerts.db and each incident meta.json.

The image itself stays off-chain. Its SHA-256 fingerprint is what is recorded on Fabric so the evidence can later be independently verified.