#!/usr/bin/env python3
"""BORDER SENTINEL operator dashboard.

Camera capture lives here. The perception controller consumes the newest
captured frame, so live video and selected AI processing share one capture
pipeline. AI is controlled independently for every camera.

Face recognition uses one shared registry, while each camera decides whether
its detections participate in that global matcher.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from cybersecurity.ledger_anchor import EvidenceLedger
from face_intelligence import SharedFaceRegistry
from dashboard_runtime import PerceptionController

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "cameras.json"
LOGGER = logging.getLogger("border.dashboard")
app = Flask(__name__, static_folder="dashboard_static", static_url_path="/static")


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_config(config: dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


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
        if not path.is_absolute() and not text.startswith(("rtsp://", "http://", "https://")):
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
            return self._processed_jpeg if self._processed_sequence == self._sequence and self._processed_jpeg else self._raw_jpeg

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

    def set_source(self, source: Any) -> None:
        with self._lock:
            self.source = source
            self._status = "RECONNECTING"

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = bool(enabled)

    def stop(self) -> None:
        self._stop.set()


def _build_camera_views() -> Dict[str, CameraView]:
    config = _load_config()
    return {cid: CameraView(cid, spec) for cid, spec in (config.get("cameras") or {}).items()}


_CONFIG_AT_START = _load_config()
CAMERAS = _build_camera_views()
face_registry_config = (_CONFIG_AT_START.get("shared_face_recognition") or {})
FACE_REGISTRY = SharedFaceRegistry(
    registry_dir=ROOT / face_registry_config.get("registry_dir", "face_registry"),
    threshold=float(face_registry_config.get("match_threshold", 0.45)),
)
LEDGER = EvidenceLedger(ROOT / "border_evidence_ledger.db")
PERCEPTION = PerceptionController(CAMERAS, FACE_REGISTRY, LEDGER)


@app.get("/")
def index() -> Response:
    return send_from_directory(ROOT / "dashboard_static", "index.html")


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
        last_payload = None
        while True:
            frame = view.jpeg()
            if frame is None:
                time.sleep(0.1)
                continue
            if frame == last_payload:
                time.sleep(0.03)
                continue
            last_payload = frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\nPragma: no-cache\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.run(host="0.0.0.0", port=8080, threaded=True, debug=False)
