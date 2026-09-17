"""Shared face-recognition registry used across all cameras.

The registry is camera-agnostic: people are enrolled once and any camera can
submit a face crop for matching. InsightFace is used when installed. The class
keeps the model optional so the dashboard can report a clear unavailable state
rather than silently falling back to a non-face-recognition implementation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("perception.face_intelligence")


class SharedFaceRegistry:
    def __init__(
        self,
        registry_dir: Path | str = "face_registry",
        threshold: float = 0.45,
        model_name: str = "buffalo_l",
    ) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        self.model_name = model_name
        self.people_path = self.registry_dir / "people.json"
        self.samples_dir = self.registry_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.people: Dict[str, dict] = self._load_people()
        self.app = None
        self.available = False
        self.error: Optional[str] = None
        self._load_model()

    def _load_people(self) -> Dict[str, dict]:
        if not self.people_path.exists():
            return {}
        try:
            payload = json.loads(self.people_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to load face registry: %s", exc)
            return {}

    def _save_people(self) -> None:
        tmp = self.people_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.people, indent=2), encoding="utf-8")
        tmp.replace(self.people_path)

    def _load_model(self) -> None:
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.available = True
            self.error = None
            LOGGER.info("Shared face recognition ready: InsightFace/%s", self.model_name)
        except Exception as exc:  # dependency/runtime/model download issues are environment-specific
            self.available = False
            self.error = str(exc)
            LOGGER.warning("Shared face recognition unavailable: %s", exc)

    def status(self) -> dict:
        return {
            "available": self.available,
            "backend": "InsightFace" if self.available else None,
            "model": self.model_name,
            "threshold": self.threshold,
            "known_people": len(self.people),
            "error": self.error,
        }

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise ValueError("invalid face embedding")
        return vector / norm

    def _embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self.available or self.app is None:
            return None
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None
        faces = self.app.get(image)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
        )
        embedding = getattr(face, "embedding", None)
        return self._normalize(embedding) if embedding is not None else None

    def enroll(self, person_id: str, display_name: str, image: np.ndarray) -> dict:
        if not self.available:
            raise RuntimeError(self.error or "face recognition backend unavailable")
        person_id = str(person_id).strip()
        display_name = str(display_name).strip()
        if not person_id or not display_name:
            raise ValueError("person_id and display_name are required")

        embedding = self._embedding_from_image(image)
        if embedding is None:
            raise ValueError("no usable face found in image")

        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in person_id)
        if not safe_id:
            raise ValueError("person_id contains no usable characters")
        sample_path = self.samples_dir / f"{safe_id}_{int(time.time())}.jpg"
        if not cv2.imwrite(str(sample_path), image):
            raise RuntimeError("failed to save enrollment sample")

        entry = self.people.setdefault(
            person_id,
            {"person_id": person_id, "display_name": display_name, "samples": [], "embeddings": []},
        )
        entry["display_name"] = display_name
        entry.setdefault("samples", []).append(str(sample_path))
        entry.setdefault("embeddings", []).append(embedding.tolist())
        self._save_people()

        return {
            "person_id": person_id,
            "display_name": display_name,
            "sample_count": len(entry["samples"]),
            "sample_path": str(sample_path),
        }

    def match(self, image: np.ndarray) -> Optional[dict]:
        embedding = self._embedding_from_image(image)
        if embedding is None:
            return None

        best: Optional[Tuple[str, float]] = None
        for person_id, entry in self.people.items():
            for raw in entry.get("embeddings", []):
                candidate = self._normalize(np.asarray(raw, dtype=np.float32))
                similarity = float(np.dot(embedding, candidate))
                if best is None or similarity > best[1]:
                    best = (person_id, similarity)

        if best is None or best[1] < self.threshold:
            return None

        person_id, similarity = best
        entry = self.people[person_id]
        return {
            "person_id": person_id,
            "display_name": entry.get("display_name", person_id),
            "similarity": round(similarity, 4),
            "threshold": self.threshold,
        }

    def list_people(self) -> List[dict]:
        return [
            {
                "person_id": pid,
                "display_name": entry.get("display_name", pid),
                "sample_count": len(entry.get("samples", [])),
            }
            for pid, entry in sorted(self.people.items())
        ]
