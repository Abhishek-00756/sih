"""QR pairing primitives for BORDER SENTINEL mobile cameras.

This module creates short-lived pairing payloads. A production mobile-camera
client can scan the QR payload and use the returned pairing URL to register a
camera with the dashboard. The actual media transport remains separate so RTSP
and future WebRTC clients can share the same camera registry.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PairingSession:
    token: str
    camera_id: str
    created_at: float
    expires_at: float
    payload: dict


class PairingManager:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = int(ttl_seconds)
        self._sessions: Dict[str, PairingSession] = {}

    def create(self, camera_id: str, dashboard_url: str) -> PairingSession:
        token = secrets.token_urlsafe(18)
        now = time.time()
        payload = {
            "type": "border-sentinel-camera-pair",
            "version": 1,
            "camera_id": camera_id,
            "dashboard_url": dashboard_url.rstrip("/"),
            "token": token,
            "expires_at": int(now + self.ttl_seconds),
        }
        session = PairingSession(
            token=token,
            camera_id=camera_id,
            created_at=now,
            expires_at=now + self.ttl_seconds,
            payload=payload,
        )
        self._sessions[token] = session
        return session

    def get(self, token: str) -> Optional[PairingSession]:
        session = self._sessions.get(token)
        if session is None:
            return None
        if time.time() >= session.expires_at:
            self._sessions.pop(token, None)
            return None
        return session

    @staticmethod
    def encode_payload(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def fingerprint(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12].upper()
