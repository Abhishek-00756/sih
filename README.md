# BORDER SENTINEL

## AI-Based Intelligent Video Analytics & Secure Evidence Platform

BORDER SENTINEL is a multi-camera border-surveillance prototype designed around an existing CCTV infrastructure. It combines computer vision, person re-identification, persistent cross-camera identity, configurable perimeter rules, incident evidence capture, cybersecurity integrity controls, and a permissioned Hyperledger Fabric evidence ledger.

The system is designed for an operator / command-and-control dashboard where up to three camera feeds can be viewed together and AI can be enabled selectively per camera to control local compute load.

> **Project status:** The core surveillance dashboard, QR mobile-camera onboarding, OSNet/Global-ID pipeline, geofence/tripwire intrusion flow, local evidence integrity controls, and Hyperledger Fabric integration are implemented. The current Fabric setup is a local development/test network; a production deployment would place peers/orderers and durable evidence storage on dedicated infrastructure.

---

## 1. What the project does

BORDER SENTINEL turns normal camera feeds into a security-perception pipeline:

```text
Camera / RTSP / Phone QR Camera
          |
          v
Frame Capture
          |
          v
YOLOv8 Detection
          |
          v
ByteTrack Local Tracking
          |
          v
OSNet Person Re-ID
          |
          v
Shared Global Gallery
          |
          v
Persistent Global ID
          |
    +-----+-----+-----+-----+
    |           |           |
    v           v           v
 Geofence    Tripwire      Face ID
    |           |           |
    +-----+-----+-----+-----+
              |
              v
        Risk / Incident
              |
      +-------+--------+
      |                |
      v                v
 Visual Evidence    SHA-256
      |                |
      v                v
 SQLite + metadata  Local hash chain
      |                |
      +-------+--------+
              |
              v
       Hyperledger Fabric
       evidencechannel
              |
              v
       Fabric Transaction ID
              |
              v
        C2 Dashboard
```

### Main capabilities

| Area | Capability |
|---|---|
| Cameras | CAM-01, CAM-02, CAM-03 |
| Inputs | RTSP, HTTP sources, webcam/file sources, browser phone camera |
| Phone onboarding | Short-lived QR pairing for every camera slot |
| Detection | YOLOv8 |
| Tracking | ByteTrack |
| Person Re-ID | OSNet through Torchreid |
| Cross-camera identity | Shared appearance gallery + persistent Global IDs |
| Geofence | Per-camera zones |
| Tripwire | Operator-drawn line + INBOUND/OUTBOUND direction |
| ANPR | Per-camera vehicle/license-plate processing |
| Face recognition | Shared registry; camera participation can be enabled independently |
| Enhancement | CLAHE-based frame enhancement |
| Risk | Context/risk scoring |
| Evidence | Full-frame image, person crop, metadata JSON |
| Integrity | SHA-256 image hash + local hash chains |
| Blockchain | Hyperledger Fabric evidence chaincode |
| Dashboard | Live feeds, controls, incidents, evidence links, ledger/Fabric status |

---

# 2. Dashboard

Start the dashboard with:

```bash
python3 dashboard_server.py
```

The server runs HTTPS on port **8080** and an HTTP compatibility server on **8081**.

Example:

```text
https://192.168.x.x:8080
```

The HTTPS certificate is generated locally for development, so a browser may show a certificate warning on first use.

### Camera cards

Each camera card contains:

- Live feed
- Camera status
- AI Processing switch
- Geofence toggle
- Tripwire toggle
- ANPR toggle
- Face ID participation toggle
- Enhancement toggle
- OSNet state
- Current Global IDs
- Tripwire drawing/editing controls
- **QR PAIR** button

AI processing is controlled per camera. This lets an operator display all three live feeds while only processing the cameras that need AI.

---

# 3. QR mobile-camera onboarding

Every camera slot supports browser-based phone-camera pairing.

```text
Desktop
  |
  | click QR PAIR
  v
Generate short-lived token
  |
  v
QR code
  |
  | scan
  v
Phone browser
  |
  | allow camera
  v
JPEG frames over HTTPS
  |
  v
Selected camera slot
```

### How to use it

1. Open the dashboard on the computer.
2. Click **QR PAIR** on CAM-01, CAM-02, or CAM-03.
3. Scan the generated QR with the phone.
4. Continue through the local HTTPS certificate warning if shown.
5. Allow browser camera access.
6. The phone becomes the selected camera's live source.

The phone and computer must be able to reach each other on the same LAN/Wi-Fi/hotspot during local development.

Pairing tokens expire after approximately five minutes and should not be treated as permanent credentials.

RTSP/HTTP inputs remain supported for real CCTV integrations.

---

# 4. Shared perception and Global IDs

Each camera maintains its own ByteTrack local track IDs.

Person appearance is processed through one shared OSNet feature extractor and one shared global gallery:

```text
CAM-01 local track 17  --CAM-02 local track 4   ----> OSNet --> Shared Gallery --> GID 12
CAM-03 local track 8  --/
```

A person can therefore retain the same **Global ID** when the appearance similarity and temporal constraints are sufficient.

### Important distinction

- **Local track ID:** camera-specific
- **Global ID:** shared across cameras
- **OSNet:** person appearance embedding
- **Gallery:** cross-camera identity association

A Global ID match is an appearance-based association and is not a guarantee of human identity.

---

# 5. OSNet

The production Re-ID path is intended to use real OSNet, not the histogram fallback.

The relevant implementation is:

```text
extractor.py
    |
    v
Torch + Torchvision + Torchreid
    |
    v
OSNet
    |
    v
512-d normalized embedding
```

Check the environment:

```bash
python3 -c "import torch, torchvision, torchreid; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('torchreid: OK')"
```

The dashboard should report:

```text
OSNET: OSNET
```

not:

```text
OSNET: FALLBACK
```

The fallback embedding exists for compatibility/testing but should not be presented as production OSNet.

---

# 6. Geofence

Geofences are polygon regions used to detect a tracked object's footprint inside a restricted zone.

The footprint is approximated using the bottom-center of the bounding box:

```text
footprint = ((x1 + x2) / 2, y2)
```

This is a pixel-space ground-contact approximation, not a true 3D position or GPS coordinate.

Restricted-zone intrusion generates:

- Camera ID
- Global ID
- Zone ID/name
- Bounding box
- Footprint
- Timestamp
- Risk score/factors
- Evidence snapshot
- SHA-256 hash
- Local ledger record
- Fabric evidence transaction

Geofence configuration is stored under:

```text
configs/zones.json
```

---

# 7. Virtual tripwire

The dashboard provides an operator-drawn virtual tripwire.

### Configuration

On a camera card:

```text
DRAW LINE
   |
   +--> click start point
   |
   +--> click end point
   |
   v
SAVE LINE
```

Coordinates are stored normalized to the image dimensions.

The backend detects when a tracked person's footprint path crosses the line and classifies:

```text
INBOUND
OUTBOUND
UNKNOWN
```

Tripwire detection uses the same persistent Global ID and evidence pipeline as the rest of the intrusion system.

---

# 8. ANPR

ANPR is **camera-specific**.

Each camera can independently enable:

```text
ANPR ON
    |
    v
vehicle detection
    |
    v
vehicle crop
    |
    v
EasyOCR
    |
    v
plate text
```

This means ANPR processing on CAM-02 does not automatically turn ANPR on for CAM-01 or CAM-03.

---

# 9. Shared face recognition

Face recognition uses one shared registry.

A person is enrolled once:

```text
Face image
   |
   v
InsightFace
   |
   v
Shared face registry
```

Any camera with its **Face ID** feature enabled may participate in matching against the same registry.

Relevant files:

```text
face_intelligence.py
face_registry/
```

The optional face dependencies are installed separately because the model/runtime is heavier:

```bash
python3 -m pip install -r requirements-face.txt
```

---

# 10. Evidence capture

Every intrusion creates a dedicated incident folder.

Typical structure:

```text
alert_snapshots/
└── 20260919/
    └── cam_CAM-01_gid_12_<timestamp>/
        ├── full.jpg
        ├── crop.jpg
        └── meta.json
```

### full.jpg

The complete camera frame with the intrusion annotation.

### crop.jpg

The detected person's cropped evidence image.

### meta.json

Structured incident metadata including:

- Event ID
- Timestamp
- Camera ID
- Global ID
- Event type
- Direction
- Bounding box
- Footprint
- Risk information
- Image SHA-256
- Local chain hashes
- Fabric status
- Fabric transaction ID

---

# 11. Cybersecurity and tamper-evident storage

BORDER SENTINEL uses multiple integrity layers.

### Layer 1 — Evidence image hashing

The saved evidence image is hashed with SHA-256.

```text
full.jpg
   |
   v
SHA-256
   |
   v
image_hash
```

Changing the file changes its SHA-256 value.

### Layer 2 — Local evidence ledger

```text
border_alerts.db
    |
    +-- security_alerts
    |
    +-- security_ledger
```

The `security_ledger` uses:

```text
previous_hash
      +
event data
      +
image_hash
      |
      v
block_hash
```

Each new record refers to the previous record, creating a tamper-evident chain.

### Layer 3 — Dashboard metadata ledger

The dashboard also maintains:

```text
border_evidence_ledger.db
```

through:

```text
cybersecurity/ledger_anchor.py
```

### Verification

The dashboard can report whether the local chains remain valid.

The application exposes:

```text
/api/security/status
/api/ledger/verify
```

---

# 12. Hyperledger Fabric blockchain

The blockchain layer is now a real Hyperledger Fabric integration.

### Network

The current development network contains:

```text
Org1MSP
Org2MSP
   |
   v
evidencechannel
   |
   v
evidence chaincode
   |
   v
Orderer
```

The evidence chaincode is located in:

```text
blockchain/
└── chaincode/
    └── evidence-contract/
        ├── evidence_contract.go
        └── go.mod
```

The Python integration is:

```text
blockchain/fabric_client.py
```

### What goes on Fabric

The actual image does **not** go on-chain.

Instead, the application records:

```text
event_id
camera_id
global_id
event_type
timestamp
direction
evidence_sha256
previous_hash
```

The image remains in off-chain evidence storage.

This makes verification possible:

```text
Stored evidence
      |
      v
Calculate SHA-256
      |
      v
Compare with Fabric
      |
   +--+--+
   |     |
 MATCH  NO MATCH
   |     |
   v     v
valid  altered/mismatch
```

### Fabric chaincode functions

```text
RecordEvidence()
GetEvidence()
EvidenceExists()
VerifyEvidence()
DeleteEvidence()   # controlled/testing operation
```

For an operational deployment, destructive ledger operations such as `DeleteEvidence` should be reviewed and restricted according to the deployment's audit policy.

### Python application flow

```text
AlertLogger.log_intrusion()
       |
       +--> save evidence
       |
       +--> calculate SHA-256
       |
       +--> local security ledger
       |
       +--> FabricEvidenceClient.record_evidence()
                    |
                    v
              RecordEvidence()
                    |
                    v
               Fabric commit
                    |
                    v
              transaction_id
                    |
                    v
          save status in SQLite/meta.json
```

Fabric anchoring is performed asynchronously so a temporary blockchain/network failure does not stop the real-time camera-perception loop.

### Important deployment limitation

The current Fabric setup is a **local development/test network on the operator machine**. It is a real Fabric ledger, but it is not yet a production multi-site network with remote peers and independently managed ordering infrastructure.

---

# 13. Current application-to-blockchain status

The following flow has been tested:

```text
Python application
       |
       v
FabricEvidenceClient
       |
       v
Hyperledger Fabric
       |
       +--> Org1
       |
       +--> Org2
       |
       +--> Orderer
       |
       v
evidencechannel
       |
       v
ANCHORED
       |
       v
Fabric transaction ID
```

A successful application test returns:

```text
status: ANCHORED
transaction_id: <64-character Fabric transaction ID>
```

---

# 14. Project structure

Important files:

```text
sih/
├── dashboard_server.py          # Flask C2 dashboard + camera/QR APIs
├── dashboard_runtime.py         # shared live AI perception runtime
├── dashboard_pairing.py         # QR pairing sessions
│
├── dashboard_static/
│   ├── index.html               # operator dashboard
│   ├── app.js                   # dashboard logic
│   ├── style.css                # dashboard styling
│   └── mobile_camera.html       # phone camera page
│
├── extractor.py                 # OSNet Re-ID extractor
├── gallery_manager.py           # shared Global ID gallery
├── geofence_manager.py          # geofence + virtual tripwire
├── alert_logger.py              # evidence + local security ledger
├── risk_scorer.py               # incident risk score
├── activity_analyzer.py         # dwell / loitering logic
├── anpr_manager.py              # ANPR/OCR
├── enhancer.py                  # image enhancement
├── face_intelligence.py         # shared face registry
│
├── cybersecurity/
│   └── ledger_anchor.py         # SHA-256 metadata chain
│
├── blockchain/
│   ├── fabric_client.py         # Python -> Fabric integration
│   └── chaincode/
│       └── evidence-contract/
│           ├── evidence_contract.go
│           └── go.mod
│
├── configs/
│   ├── cameras.json
│   ├── perception.yaml
│   └── zones.json
│
├── requirements.txt             # core AI/runtime dependencies
├── requirements-dashboard.txt   # QR/dashboard dependency
├── requirements-face.txt        # InsightFace/ONNX dependencies
└── README.md
```

Runtime-generated data is intentionally excluded from Git:

```text
*.db
alert_snapshots/
face_registry/
model weights
large sample videos
```

A fresh machine therefore creates its own local evidence history and its own local Fabric ledger.

---

# 15. COMPLETE SETUP FOR A NEW MACHINE

## Recommended platform

The easiest team setup is:

```text
macOS
Apple Silicon or Intel
Docker Desktop
Python 3
Go
Git
jq
```

The Hyperledger Fabric documentation recommends Docker Desktop on macOS and uses Homebrew for prerequisites. Go is required when developing Go chaincode. See the official Fabric prerequisites documentation for platform-specific details:

https://hyperledger-fabric.readthedocs.io/en/latest/prereqs.html
```

## Step 1 — Clone BORDER SENTINEL

```bash
git clone https://github.com/Abhishek-00756/sih.git
cd sih
```

## Step 2 — Install system prerequisites

On macOS with Homebrew:

```bash
brew install git go jq
brew install --cask docker
open -a Docker
```

Wait for Docker Desktop to finish starting.

Verify:

```bash
docker --version
go version
git --version
jq --version
```

## Step 3 — Create Python environment

```bash
cd ~/sih
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

## Step 4 — Install project dependencies

Core perception:

```bash
python3 -m pip install -r requirements.txt
```

Dashboard / QR:

```bash
python3 -m pip install -r requirements-dashboard.txt
```

Face recognition:

```bash
python3 -m pip install -r requirements-face.txt
```

## Step 5 — Install OSNet / Torchreid

```bash
python3 -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Verify:

```bash
python3 -c "import torch, torchvision, torchreid; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('torchreid: OK')"
```

## Step 6 — Install Hyperledger Fabric

From the home directory:

```bash
cd ~
curl -sSL https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh | bash -s -- samples,binaries,docker
```

Verify:

```bash
ls ~/fabric-samples
ls ~/fabric-samples/test-network
```

You should see the Fabric test network and `network.sh`.

## Step 7 — Start the Fabric network

```bash
cd ~/fabric-samples/test-network

./network.sh up createChannel -c evidencechannel -ca
```

This creates the local orderer, peers, certificate authorities and the `evidencechannel` channel.

## Step 8 — Build/check the evidence chaincode

```bash
cd ~/sih/blockchain/chaincode/evidence-contract
go mod tidy
go build
```

No output from `go build` means the chaincode compiled successfully.

## Step 9 — Deploy the BORDER SENTINEL evidence contract

```bash
cd ~/fabric-samples/test-network

./network.sh deployCC   -c evidencechannel   -ccn evidence   -ccp ~/sih/blockchain/chaincode/evidence-contract   -ccl go
```

Verify:

```bash
peer lifecycle chaincode querycommitted   --channelID evidencechannel   --name evidence
```

Expected:

```text
Committed chaincode definition for chaincode 'evidence'
on channel 'evidencechannel'
```

## Step 10 — Set the Fabric environment for BORDER SENTINEL

Each new terminal session that runs BORDER SENTINEL should have:

```bash
export FABRIC_TEST_NETWORK=~/fabric-samples/test-network
export FABRIC_CHANNEL=evidencechannel
export FABRIC_CHAINCODE=evidence
export BORDER_SENTINEL_BLOCKCHAIN_ENABLED=true
```

The Python Fabric client reads these variables automatically.

## Step 11 — Verify the Python -> Fabric connection

```bash
cd ~/sih
source .venv/bin/activate

python3 - <<'PY'
from blockchain.fabric_client import FabricEvidenceClient

client = FabricEvidenceClient()
print(client.status())
PY
```

Expected shape:

```text
enabled: True
configured: True
status: READY
channel: evidencechannel
chaincode: evidence
```

## Step 12 — Start BORDER SENTINEL

```bash
cd ~/sih
source .venv/bin/activate

export FABRIC_TEST_NETWORK=~/fabric-samples/test-network
export FABRIC_CHANNEL=evidencechannel
export FABRIC_CHAINCODE=evidence
export BORDER_SENTINEL_BLOCKCHAIN_ENABLED=true

python3 dashboard_server.py
```

Open the HTTPS LAN address printed by the server:

```text
https://<computer-lan-ip>:8080
```

For phone QR onboarding, the phone must be able to reach that LAN address.

---

# 16. First-run checklist

After starting the project, verify:

```text
[ ] Dashboard opens
[ ] CAM-01 / CAM-02 / CAM-03 cards appear
[ ] Phone QR pairing works
[ ] Camera shows ONLINE
[ ] AI can be enabled
[ ] OSNET reports OSNET, not FALLBACK
[ ] Global IDs appear
[ ] Tripwire can be drawn
[ ] Tripwire reports INBOUND / OUTBOUND
[ ] Evidence appears under alert_snapshots/
[ ] border_alerts.db receives the alert
[ ] Local ledger stays valid
[ ] Fabric status is READY/ANCHORED
[ ] New incident gets a Fabric transaction ID
```

---

# 17. Testing a real tripwire incident

1. Start Docker and the Fabric network.
2. Start the BORDER SENTINEL dashboard.
3. Pair a phone with a camera using **QR PAIR**.
4. Enable **AI PROCESSING** for that camera.
5. Enable **Tripwire**.
6. Draw and save a tripwire.
7. Walk across the line in the camera view.
8. Confirm the direction is reported.
9. Check **Live Intrusions**.
10. Open **VIEW EVIDENCE**.

The expected evidence flow is:

```text
Tripwire crossing
      |
      v
Incident row
      |
      +--> full.jpg
      +--> crop.jpg
      +--> meta.json
      |
      +--> image SHA-256
      |
      +--> local security_ledger
      |
      +--> Fabric RecordEvidence()
                  |
                  v
             ANCHORED
                  |
                  v
           transaction ID
```

---

# 18. Query incidents from SQLite

```bash
cd ~/sih

sqlite3 border_alerts.db "SELECT id, event_id, timestamp, camera_id, global_id, alert_type, direction, blockchain_status, blockchain_tx_id FROM security_alerts ORDER BY id DESC LIMIT 10;"
```

A successfully anchored incident should contain:

```text
blockchain_status = ANCHORED
blockchain_tx_id  = <Fabric transaction ID>
```

---

# 19. Verify local ledger integrity

The dashboard exposes the verification API, and the local blockchain-style ledger can also be checked from Python.

For the dashboard metadata ledger:

```bash
curl -k https://localhost:8080/api/ledger/verify
```

For combined security status:

```bash
curl -k https://localhost:8080/api/security/status
```

---

# 20. Verify a Fabric evidence record manually

Set the peer environment:

```bash
cd ~/fabric-samples/test-network

export PATH="$PWD/../bin:$PATH"
export FABRIC_CFG_PATH="$PWD/../config/"

source ./scripts/envVar.sh
setGlobals 1
```

Then query a known event:

```bash
peer chaincode query   -C evidencechannel   -n evidence   -c '{"function":"GetEvidence","Args":["TEST-001"]}'
```

To use Org2:

```bash
source ./scripts/envVar.sh
setGlobals 2
```

Then query the same event again.

---

# 21. Useful API endpoints

### Dashboard state

```text
GET /api/state
```

### Recent incidents

```text
GET /api/alerts/recent?limit=25
```

### Evidence snapshot

```text
GET /api/alerts/<alert_id>/snapshot
```

### Security status

```text
GET /api/security/status
```

### Ledger verification

```text
GET /api/ledger/verify
```

### Face registry

```text
GET /api/face/people
POST /api/face/enroll
```

### QR pairing

```text
POST /api/pairing/create
GET  /api/pairing/<token>/qr.png
GET  /api/pairing/<token>/status
POST /api/pairing/<token>/frame
POST /api/pairing/<token>/disconnect
```

### Tripwire configuration

```text
GET    /api/cameras/<camera_id>/tripwire
POST   /api/cameras/<camera_id>/tripwire
DELETE /api/cameras/<camera_id>/tripwire
```

---

# 22. Configuration

## cameras.json

Controls camera source, enable state, AI processing and per-camera feature switches.

Conceptually:

```json
{
  "cameras": {
    "CAM-01": {
      "name": "Outpost Alpha",
      "source": "mobile://...",
      "enabled": true,
      "ai_enabled": true,
      "features": {
        "geofence": true,
        "tripwire": true,
        "anpr": false,
        "face_recognition": true,
        "enhancement": true
      }
    }
  }
}
```

QR pairing modifies the selected camera's runtime source to a `mobile://...` session token.

## perception.yaml

Controls:

- OSNet model
- YOLO weights
- Re-ID thresholds
- Global gallery time window
- loitering/dwell threshold
- ANPR confidence
- enhancement mode
- geofence configuration
- tripwire configuration
- alert database
- evidence snapshot directory

---

# 23. Performance model

Running three camera feeds does not require three AI pipelines to be active simultaneously.

The architecture is:

```text
3 live camera feeds
       |
       +--> CAM-01 AI ON
       |
       +--> CAM-02 AI OFF
       |
       +--> CAM-03 AI OFF
```

This reduces local compute pressure.

YOLO detectors are maintained per camera while OSNet and the Global Gallery are shared.

---

# 24. Evidence and blockchain design principle

The design intentionally separates:

### Off-chain

```text
full.jpg
crop.jpg
meta.json
SQLite
analytics
dashboard
```

### On-chain

```text
event ID
camera
Global ID
event type
time
direction
evidence SHA-256
previous hash
```

This avoids storing large image files in the blockchain while still making the evidence cryptographically verifiable.

---

# 25. Production roadmap

The current local deployment is a strong development/prototype environment. For a production deployment, the next architectural steps are:

```text
Current local setup
       |
       v
Dedicated AI / perception server
       |
       +--> durable object evidence storage
       |
       +--> PostgreSQL / analytics database
       |
       +--> remote Fabric peers
       |
       +--> managed ordering service
       |
       v
Vercel / web dashboard
```

Production work should also include:

- Remote authentication and authorization
- TLS certificates managed by a trusted certificate authority
- Secure camera ingestion
- Durable object storage for evidence
- Centralized logging/monitoring
- Remote Fabric peers instead of a single laptop test network
- Secret management
- Backup and disaster recovery
- Access control and audit policy for evidence deletion/export
- Load testing and model optimization

---

# 26. Important limitations

### Image-space geometry

Geofences and tripwires operate on image/pixel coordinates. They do not directly represent physical distances.

### Re-ID

OSNet provides appearance embeddings. Cross-camera Global ID association can be strong but is not a proof of identity.

### Face recognition

Face recognition accuracy depends on face quality, lighting, camera angle, enrollment quality and model behavior.

### ANPR

OCR quality depends heavily on plate resolution, angle, blur, lighting and camera placement.

### Fabric

The current Fabric network is a local development/test deployment. It demonstrates real permissioned-ledger transactions but is not yet a production multi-site blockchain network.

### Runtime data

Evidence databases and images are generated locally and are ignored by Git. Cloning the repository creates the software, not a copy of historical evidence.

---

# 27. Stopping the system

Stop the dashboard with:

```text
Ctrl+C
```

To stop the local Fabric network:

```bash
cd ~/fabric-samples/test-network
./network.sh down
```

> **Warning:** bringing the Fabric test network down removes the test-network containers and local ledger data created by that test network. Do not use this as a production data-management procedure.

---

# 28. Repository workflow for teammates

When the repository changes:

```bash
cd ~/sih
git pull origin main
```

Then activate the environment:

```bash
source .venv/bin/activate
```

Run syntax checks:

```bash
python3 -m py_compile   blockchain/fabric_client.py   alert_logger.py   dashboard_runtime.py   dashboard_server.py
```

For chaincode:

```bash
cd blockchain/chaincode/evidence-contract
go mod tidy
go build
```

Return to the project:

```bash
cd ~/sih
```

---

# 29. Development notes

- Do not commit `.env`, private keys, Fabric certificates, database files, model weights, or generated evidence.
- Do not delete `border_alerts.db` or `alert_snapshots/` during an evidence test unless you intentionally want to reset local evidence.
- Keep the Fabric network running while testing automatic anchoring.
- A fresh machine needs its own Fabric network and chaincode deployment.
- Historical evidence is intentionally local to the machine where it was generated.
- For a multi-machine deployment, the current absolute local Fabric paths in `blockchain/fabric_client.py` should be replaced with a proper service/API or remote Fabric gateway architecture.

---

# 30. Official reference

Hyperledger Fabric prerequisites and development/test-network documentation:

https://hyperledger-fabric.readthedocs.io/en/latest/prereqs.html

https://hyperledger-fabric.readthedocs.io/en/latest/test_network.html

The project intentionally keeps the surveillance/evidence application separate from the Fabric network so the blockchain backend can evolve independently.

---

## Summary

BORDER SENTINEL currently provides this complete prototype path:

```text
3 Cameras
   |
   +--> QR phone onboarding
   |
   v
YOLOv8
   |
   v
ByteTrack
   |
   v
OSNet
   |
   v
Shared Global IDs
   |
   +--> Geofence
   +--> Tripwire IN/OUT
   +--> ANPR
   +--> Face recognition
   +--> Enhancement
   |
   v
Incident
   |
   +--> full.jpg
   +--> crop.jpg
   +--> meta.json
   +--> SQLite
   +--> SHA-256
   +--> local tamper-evident chain
   |
   v
Hyperledger Fabric
   |
   v
Evidence transaction + TX ID
   |
   v
BORDER SENTINEL C2 DASHBOARD
```

This repository is the implementation and development starting point for a secure AI-assisted border-surveillance system, with clear separation between computer vision, evidence integrity, and permissioned blockchain anchoring.
