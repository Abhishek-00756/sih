"""CPU Haar Cascade face capture for person crops from the shared perception loop.

Saves one mugshot per Global ID so operators get a frontal snapshot without
spamming disk or stalling the live stream.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

LOGGER = logging.getLogger("perception.face")

DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"
DEFAULT_SAVE_DIR = "face_database"
DEFAULT_SCALE_FACTOR = 1.1
DEFAULT_MIN_NEIGHBORS = 4
DEFAULT_MIN_SIZE = (30, 30)
DEFAULT_HEAD_MARGIN = 0.2


def _safe_token(value: object) -> str:
    text = "".join(ch if str(ch).isalnum() or ch in "-_" else "_" for ch in str(value))
    return text.strip("_") or "unknown"


class FaceManager:
    def __init__(
        self,
        save_dir: Union[str, Path] = DEFAULT_SAVE_DIR,
        cascade_path: Optional[Union[str, Path]] = None,
        cascade: Optional[Any] = None,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
        min_size: Tuple[int, int] = DEFAULT_MIN_SIZE,
        head_margin: float = DEFAULT_HEAD_MARGIN,
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 1:
            raise ValueError("min_neighbors must be positive")
        if head_margin < 0:
            raise ValueError("head_margin must be non-negative")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = (int(min_size[0]), int(min_size[1]))
        self.head_margin = float(head_margin)
        self.captured_faces: Set[int] = set()
        self.last_saved: Optional[Path] = None
        self.cascade = cascade if cascade is not None else self._load_cascade(cascade_path)

    def _load_cascade(self, cascade_path: Optional[Union[str, Path]]) -> Optional[Any]:
        path = Path(cascade_path) if cascade_path else None
        if path is None:
            haarcascades = getattr(cv2, "data", None)
            base = getattr(haarcascades, "haarcascades", None) if haarcascades else None
            if base:
                path = Path(base) / DEFAULT_CASCADE
        if path is None or not path.exists():
            LOGGER.warning("Haar cascade not found; face capture disabled.")
            return None
        classifier = cv2.CascadeClassifier(str(path))
        if classifier.empty():
            LOGGER.warning("Failed to load Haar cascade %s; face capture disabled.", path)
            return None
        LOGGER.info("Loaded face cascade from %s", path)
        return classifier

    def already_captured(self, global_id: int) -> bool:
        return int(global_id) in self.captured_faces

    def _largest_face(self, faces: Sequence[Sequence[int]]) -> Optional[Tuple[int, int, int, int]]:
        best: Optional[Tuple[int, int, int, int]] = None
        best_area = 0
        for face in faces:
            if face is None or len(face) < 4:
                continue
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            area = w * h
            if area > best_area:
                best_area = area
                best = (x, y, w, h)
        return best

    def _expand_box(
        self,
        box: Tuple[int, int, int, int],
        height: int,
        width: int,
    ) -> Tuple[int, int, int, int]:
        x, y, w, h = box
        margin = int(h * self.head_margin)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)
        return x1, y1, x2, y2

    def detect_and_save_face(
        self,
        person_crop: Optional[np.ndarray],
        global_id: int,
        camera_id: str,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Scan a body crop for a frontal face and save one mugshot per Global ID."""
        gid = int(global_id)
        if gid in self.captured_faces:
            return False
        if self.cascade is None or person_crop is None:
            return False
        if not isinstance(person_crop, np.ndarray) or person_crop.size == 0:
            return False
        if person_crop.ndim < 2 or min(person_crop.shape[:2]) < self.min_size[1]:
            return False

        gray = (
            cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            if person_crop.ndim == 3
            else person_crop
        )
        try:
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )
        except Exception as exc:
            LOGGER.debug("Face detect failed: %s", exc)
            return False

        box = self._largest_face(faces)
        if box is None:
            return False

        h, w = person_crop.shape[:2]
        x1, y1, x2, y2 = self._expand_box(box, h, w)
        if x2 <= x1 or y2 <= y1:
            return False
        face_crop = person_crop[y1:y2, x1:x2]
        if face_crop.size == 0:
            return False

        now = (
            datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
            if timestamp is not None
            else datetime.now(timezone.utc)
        )
        filename = (
            f"face_GID_{gid}_cam_{_safe_token(camera_id)}_"
            f"{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        filepath = self.save_dir / filename
        if not cv2.imwrite(str(filepath), face_crop):
            LOGGER.warning("Failed to write face crop %s", filepath)
            return False

        self.captured_faces.add(gid)
        self.last_saved = filepath
        LOGGER.info(
            "[INTELLIGENCE] Face captured for Global ID %s at %s. Saved to %s",
            gid,
            camera_id,
            filepath,
        )
        return True
