#!/usr/bin/env python3
"""BORDER SENTINEL operator dashboard.

Camera capture lives here. The perception controller consumes the newest
captured frame, so live video and selected AI processing share one capture
pipeline. AI is controlled independently for every camera.

Face recognition uses one shared registry, while each camera decides whether
its detections participate in that global matcher.

The dashboard also supports short-lived QR pairing for a browser-based mobile
camera. The mobile page uses HTTPS and sends JPEG frames to the paired slot;
RTSP/HTTP camera sources remain supported for existing deployments.
"""
from __future__ import annotations

import io
import json
import logging
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import qrcode
from flask import Flask, Response, jsonify, redirect, request, send_file, send_from_directory

from cybersecurity.ledger_anchor import EvidenceLedger
from dashboard_pairing import PairingManager
from dashboard_runtime import PerceptionController
from face_intelligence import SharedFaceRegistry

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "cameras.json"
LOGGER = logging.getLogger("border.dashboard")
app = Flask(__name__, static_folder="dashboard_static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
HTTPS_PORT = int(__import__("os").environ.get("BORDER_SENTINEL_HTTPS_PORT", "8080"))
HTTP_PORT = int(__import__("os").environ.get("BORDER_SENTINEL_HTTP_PORT", "8081"))


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_config(config: dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


def _lan_ip(preferred_host: Optional[str] = None) -> str:
    """Return the address another device should use for pairing."""
    host = (preferred_host or "").strip()
    if host and host not in {"localhost", "127.0.0.1", "::1"}:
        return host

    if Path("/usr/sbin/ipconfig").exists():
        for iface in ("en0", "en1", "en2", "en3", "en4", "en5"):
            try:
                result = subprocess.run(
                    ["/usr/sbin/ipconfig", "getifaddr", iface],
                    capture_output=True, text=True, timeout=1.0, check=False,
                )
                addr = result.stdout.strip()
                if addr and not addr.startswith("127."):
                    return addr
            except (OSError, subprocess.SubprocessError):
                pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        addr = sock.getsockname()[0]
        sock.close()
        if addr and not addr.startswith("127."):
            return addr
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if addr and not addr.startswith("127."):
                return addr
    except OSError:
        pass
    return "127.0.0.1"



class CameraView:
    def __init__(self, camera_id: str, spec: dict) -> None:
        self.camera_id = camera_id
        self.name = str(spec.get("name", camera_id))
        self.source = spec.get("source", "0")
        self.enabled = bool(spec.get("enabled", True))
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._raw_jpeg: Optional[bytes] = None
        self._processed_jpeg: Optional[bytes] = None
        self._processed_sequence = -1
        self._sequence = 0
        self._status = "STARTING"
        self._last_frame_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{camera_id}", daemon=True)
        self._thread.start()

    @staticmethod
    def _resolve_source(source: Any) -> Union[int, str]:
        if isinstance(source, int):
            return source
        text = str(source).strip()
        if text.isdigit():
            return int(text)
        path = Path(text)
        if not path.is_absolute() and not text.startswith(("rtsp://", "http://", "https://", "mobile://")):
            path = ROOT / path
        return str(path)

    def _run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        last_source = None
        while not self._stop.is_set():
            if not self.enabled:
                if cap is not None:
                    cap.release()
                    cap = None
                    last_source = None
                with self._lock:
                    self._status = "DISABLED"
                time.sleep(0.5)
                continue

            source_text = str(self.source)
            if source_text.startswith("mobile://"):
                with self._lock:
                    age = time.time() - self._last_frame_at if self._last_frame_at else 999
                    self._status = "ONLINE" if age < 3.0 else "WAITING FOR PHONE"
                time.sleep(0.15)
                continue

            resolved = self._resolve_source(self.source)
            if cap is None or last_source != self.source or not cap.isOpened():
                if cap is not None:
                    cap.release()
                last_source = self.source
                cap = cv2.VideoCapture(resolved)
                if not cap.isOpened():
                    with self._lock:
                        self._status = "OFFLINE"
                    time.sleep(1.0)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                if isinstance(resolved, str) and Path(resolved).exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue
                with self._lock:
                    self._status = "NO FRAME"
                time.sleep(0.2)
                continue

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue
            with self._lock:
                self._frame = frame
                self._raw_jpeg = encoded.tobytes()
                self._sequence += 1
                self._last_frame_at = time.time()
                self._status = "ONLINE"

        if cap is not None:
            cap.release()

    def jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._processed_sequence == self._sequence and self._processed_jpeg:
                return self._processed_jpeg
            return self._raw_jpeg

    def frame(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return None, self._sequence
            return self._frame.copy(), self._sequence

    def set_processed_frame(self, frame: Optional[np.ndarray]) -> None:
        with self._lock:
            if frame is None:
                self._processed_jpeg = None
                self._processed_sequence = -1
                return
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                return
            self._processed_jpeg = encoded.tobytes()
            self._processed_sequence = self._sequence

    def set_remote_jpeg(self, jpeg_bytes: bytes) -> bool:
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return False
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return False
        with self._lock:
            self._frame = image
            self._raw_jpeg = encoded.tobytes()
            self._processed_jpeg = None
            self._processed_sequence = -1
            self._sequence += 1
            self._last_frame_at = time.time()
            self._status = "ONLINE"
        return True

    def set_source(self, source: Any) -> None:
        with self._lock:
            self.source = source
            self._status = "RECONNECTING"

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = bool(enabled)

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "name": self.name,
                "source": self.source,
                "enabled": self.enabled,
                "status": self._status,
                "last_frame_at": self._last_frame_at,
                "sequence": self._sequence,
            }


_CONFIG_AT_START = _load_config()
CAMERAS = {cid: CameraView(cid, spec) for cid, spec in (_CONFIG_AT_START.get("cameras") or {}).items()}
face_registry_config = (_CONFIG_AT_START.get("shared_face_recognition") or {})
FACE_REGISTRY = SharedFaceRegistry(
    registry_dir=ROOT / face_registry_config.get("registry_dir", "face_registry"),
    threshold=float(face_registry_config.get("match_threshold", 0.45)),
)
LEDGER = EvidenceLedger(ROOT / "border_evidence_ledger.db")
PERCEPTION = PerceptionController(CAMERAS, FACE_REGISTRY, LEDGER)
PAIRING = PairingManager(ttl_seconds=300)


@app.before_request
def _force_https_on_http():
    if not request.is_secure and request.environ.get("SERVER_PORT") == str(HTTP_PORT):
        host = request.host.split(":", 1)[0]
        return redirect(f"https://{host}:{HTTPS_PORT}{request.full_path}", code=307)

@app.get("/")
def index() -> Response:
    return send_from_directory(ROOT / "dashboard_static", "index.html")


@app.get("/pair/<token>")
def mobile_pair(token: str):
    session = PAIRING.get(token)
    if session is None:
        return "Pairing link expired or invalid.", 404
    return send_from_directory(ROOT / "dashboard_static", "mobile_camera.html")


@app.post("/api/pairing/create")
def create_pairing():
    data = request.get_json(silent=True) or {}
    camera_id = str(data.get("camera_id", "")).strip()
    if camera_id not in CAMERAS:
        return jsonify({"error": "camera not found"}), 404
    scheme = "https"
    client_host = str(data.get("client_host", "")).strip()
    host = _lan_ip(client_host)
    if host in {"127.0.0.1", "localhost", "::1"}:
        return jsonify({"error": "Open the dashboard using the LAN address shown in the terminal, then generate the QR again."}), 400
    dashboard_url = f"{scheme}://{host}:{HTTPS_PORT}"
    session = PAIRING.create(camera_id, dashboard_url)
    config = _load_config()
    spec = config.setdefault("cameras", {}).get(camera_id, {})
    spec["enabled"] = True
    spec["source"] = f"mobile://{session.token}"
    CAMERAS[camera_id].set_enabled(True)
    CAMERAS[camera_id].set_source(spec["source"])
    _save_config(config)
    return jsonify({
        "ok": True,
        "camera_id": camera_id,
        "camera_name": spec.get("name", camera_id),
        "pairing_url": f"{dashboard_url}/pair/{session.token}",
        "token": session.token,
        "expires_at": session.payload["expires_at"],
        "fingerprint": PAIRING.fingerprint(session.token),
    })


@app.get("/api/pairing/<token>/qr.png")
def pairing_qr(token: str):
    session = PAIRING.get(token)
    if session is None:
        return jsonify({"error": "pairing expired"}), 404
    image = qrcode.make(f"{session.payload['dashboard_url']}/pair/{token}")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png", max_age=0)


@app.get("/api/pairing/<token>/status")
def pairing_status(token: str):
    session = PAIRING.get(token)
    if session is None:
        return jsonify({"error": "pairing expired"}), 404
    view = CAMERAS[session.camera_id]
    return jsonify({
        "camera_id": session.camera_id,
        "camera_name": view.name,
        "status": view.status().get("status"),
        "last_frame_at": view.status().get("last_frame_at"),
        "expires_at": session.expires_at,
    })


@app.post("/api/pairing/<token>/frame")
def pairing_frame(token: str):
    session = PAIRING.get(token)
    if session is None:
        return jsonify({"error": "pairing expired"}), 404
    raw = request.get_data(cache=False)
    if not raw:
        return jsonify({"error": "empty frame"}), 400
    view = CAMERAS.get(session.camera_id)
    if view is None:
        return jsonify({"error": "camera not found"}), 404
    if not view.set_remote_jpeg(raw):
        return jsonify({"error": "invalid jpeg"}), 400
    return jsonify({"ok": True, "camera_id": session.camera_id})


@app.post("/api/pairing/<token>/disconnect")
def pairing_disconnect(token: str):
    session = PAIRING.get(token)
    if session is None:
        return jsonify({"error": "pairing expired"}), 404
    view = CAMERAS[session.camera_id]
    view.set_remote_jpeg(view.jpeg() or b"") if False else None
    with view._lock:
        view._status = "WAITING FOR PHONE"
    return jsonify({"ok": True})


@app.get("/api/state")
def api_state():
    config = _load_config()
    cameras = config.get("cameras") or {}
    payload = []
    for cid, view in CAMERAS.items():
        spec = cameras.get(cid, {})
        state = view.status()
        state["features"] = spec.get("features", {})
        state["ai_enabled"] = bool(spec.get("ai_enabled", False))
        state["ai"] = PERCEPTION.state(cid)
        state["tripwires"] = spec.get("tripwires") or []
        payload.append(state)
    return jsonify({
        "cameras": payload,
        "face_recognition": FACE_REGISTRY.status(),
        "ledger": LEDGER.verify(),
        "perception": PERCEPTION.global_state(),
    })


@app.get("/api/cameras/<camera_id>/stream")
def camera_stream(camera_id: str):
    view = CAMERAS.get(camera_id)
    if view is None:
        return jsonify({"error": "camera not found"}), 404

    def generate():
        last_sequence = -1
        while True:
            frame = view.jpeg()
            current_sequence = view.status()["sequence"]
            if frame is None or current_sequence == last_sequence:
                time.sleep(0.03)
                continue
            last_sequence = current_sequence
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\nPragma: no-cache\r\n\r\n" + frame + b"\r\n"

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/cameras/<camera_id>/tripwire")
def get_tripwire(camera_id: str):
    config = _load_config()
    spec = (config.get("cameras") or {}).get(camera_id)
    if spec is None:
        return jsonify({"error": "camera not found"}), 404
    return jsonify({"camera_id": camera_id, "tripwires": spec.get("tripwires") or []})


@app.post("/api/cameras/<camera_id>/tripwire")
def save_tripwire(camera_id: str):
    data = request.get_json(silent=True) or {}
    pt1 = data.get("pt1")
    pt2 = data.get("pt2")
    if not isinstance(pt1, (list, tuple)) or not isinstance(pt2, (list, tuple)) or len(pt1) != 2 or len(pt2) != 2:
        return jsonify({"error": "pt1 and pt2 must each contain [x, y]"}), 400
    try:
        p1 = [float(pt1[0]), float(pt1[1])]
        p2 = [float(pt2[0]), float(pt2[1])]
    except (TypeError, ValueError):
        return jsonify({"error": "tripwire points must be numeric"}), 400
    if any(v < 0.0 or v > 1.0 for v in p1 + p2):
        return jsonify({"error": "tripwire points must be normalized between 0 and 1"}), 400
    if p1 == p2:
        return jsonify({"error": "tripwire endpoints must be different"}), 400

    config = _load_config()
    spec = (config.get("cameras") or {}).get(camera_id)
    if spec is None:
        return jsonify({"error": "camera not found"}), 404
    wire = {
        "id": str(data.get("id", "primary")),
        "pt1": p1,
        "pt2": p2,
        "normalized": True,
        "inbound_positive": bool(data.get("inbound_positive", True)),
        "cooldown_seconds": float(data.get("cooldown_seconds", 60.0)),
    }
    wires = [w for w in (spec.get("tripwires") or []) if str(w.get("id", "")) != wire["id"]]
    wires.append(wire)
    spec["tripwires"] = wires
    _save_config(config)
    return jsonify({"ok": True, "camera_id": camera_id, "tripwires": wires})


@app.delete("/api/cameras/<camera_id>/tripwire")
def clear_tripwire(camera_id: str):
    config = _load_config()
    spec = (config.get("cameras") or {}).get(camera_id)
    if spec is None:
        return jsonify({"error": "camera not found"}), 404
    spec["tripwires"] = []
    _save_config(config)
    return jsonify({"ok": True, "camera_id": camera_id, "tripwires": []})


@app.post("/api/cameras/<camera_id>/settings")
def update_camera_settings(camera_id: str):
    data = request.get_json(silent=True) or {}
    config = _load_config()
    cameras = config.setdefault("cameras", {})
    if camera_id not in cameras:
        return jsonify({"error": "camera not found"}), 404
    spec = cameras[camera_id]

    if "ai_enabled" in data:
        spec["ai_enabled"] = bool(data["ai_enabled"])
    if "enabled" in data:
        spec["enabled"] = bool(data["enabled"])
        CAMERAS[camera_id].set_enabled(spec["enabled"])
    if "source" in data:
        spec["source"] = str(data["source"])
        CAMERAS[camera_id].set_source(spec["source"])
    if "feature" in data and "value" in data:
        feature = str(data["feature"])
        features = spec.setdefault("features", {})
        if feature not in {"geofence", "tripwire", "anpr", "face_recognition", "enhancement"}:
            return jsonify({"error": "unsupported feature"}), 400
        features[feature] = bool(data["value"])

    _save_config(config)
    return jsonify({"ok": True, "camera": camera_id, "settings": spec})


@app.get("/api/face/people")
def list_people():
    return jsonify({"people": FACE_REGISTRY.list_people(), "status": FACE_REGISTRY.status()})


@app.post("/api/face/enroll")
def enroll_face():
    person_id = request.form.get("person_id", "").strip()
    display_name = request.form.get("display_name", "").strip()
    upload = request.files.get("image")
    if not person_id or not display_name or upload is None:
        return jsonify({"error": "person_id, display_name and image are required"}), 400

    raw = upload.read()
    image = cv2.imdecode(np.frombuffer(raw, dtype="uint8"), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "invalid image"}), 400
    try:
        result = FACE_REGISTRY.enroll(person_id, display_name, image)
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc), "status": FACE_REGISTRY.status()}), 400
    return jsonify({"ok": True, "person": result})


@app.get("/api/ledger/verify")
def verify_ledger():
    return jsonify(LEDGER.verify())


@app.post("/api/ledger/test-anchor")
def test_anchor():
    data = request.get_json(silent=True) or {}
    data.setdefault("event", "dashboard_test")
    return jsonify(LEDGER.anchor(data))


def _run_https():
    # Local development HTTPS so phone browsers can access camera APIs.
    # Port 8080 is used because some mobile hotspots isolate uncommon ports.
    app.run(host="0.0.0.0", port=HTTPS_PORT, threaded=True, debug=False, ssl_context="adhoc")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    threading.Thread(target=_run_https, name="https-dashboard", daemon=True).start()
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True, debug=False)
