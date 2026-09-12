"""EasyOCR-based ANPR for vehicle crops from the shared perception loop.

Runs only on large (nearby) vehicle boxes so the real-time stream stays fast.
A reader can be injected for tests; if EasyOCR is missing, OCR is a no-op.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("perception.anpr")

PlateResult = Tuple[Optional[str], float]

DEFAULT_LANGUAGES = ("en",)
MIN_PLATE_LEN = 4
MAX_PLATE_LEN = 12
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_WIDTH = 150
DEFAULT_MIN_HEIGHT = 150


class ANPRManager:
    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        languages: Optional[Sequence[str]] = None,
        gpu: Optional[bool] = None,
        min_plate_len: int = MIN_PLATE_LEN,
        max_plate_len: int = MAX_PLATE_LEN,
        reader: Optional[Any] = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if min_plate_len < 1 or max_plate_len < min_plate_len:
            raise ValueError("invalid plate length bounds")

        self.min_confidence = float(min_confidence)
        self.min_plate_len = int(min_plate_len)
        self.max_plate_len = int(max_plate_len)
        self.languages = tuple(languages) if languages else DEFAULT_LANGUAGES
        self.plate_pattern = re.compile(
            rf"^(?=.*\d)[A-Z0-9]{{{self.min_plate_len},{self.max_plate_len}}}$"
        )
        self.reader = reader
        if self.reader is not None:
            return

        self.reader = self._load_reader(gpu=gpu)

    def _load_reader(self, gpu: Optional[bool]) -> Optional[Any]:
        try:
            import easyocr
        except ImportError as exc:
            LOGGER.warning("EasyOCR unavailable (%s); ANPR disabled.", exc)
            return None

        use_gpu = bool(gpu) if gpu is not None else self._cuda_available()
        LOGGER.info("Loading OCR engine (gpu=%s)...", use_gpu)
        try:
            return easyocr.Reader(list(self.languages), gpu=use_gpu)
        except Exception as exc:
            LOGGER.warning("EasyOCR failed to initialize (%s); ANPR disabled.", exc)
            return None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    @staticmethod
    def clean_text(text: str) -> str:
        """Strip non-alphanumerics and force uppercase."""
        return re.sub(r"[^A-Z0-9]", "", str(text).upper())

    def _preprocess(self, vehicle_crop: np.ndarray) -> np.ndarray:
        if vehicle_crop.ndim == 3:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = vehicle_crop
        return cv2.bilateralFilter(gray, 11, 17, 17)

    def crop_is_readable(
        self,
        width: int,
        height: int,
        min_width: int = DEFAULT_MIN_WIDTH,
        min_height: int = DEFAULT_MIN_HEIGHT,
    ) -> bool:
        return int(width) >= int(min_width) and int(height) >= int(min_height)

    def read_license_plate(self, vehicle_crop: Optional[np.ndarray]) -> PlateResult:
        """Return (plate_text, confidence) from a BGR vehicle crop, or (None, 0.0)."""
        if self.reader is None or vehicle_crop is None:
            return None, 0.0
        if not isinstance(vehicle_crop, np.ndarray) or vehicle_crop.size == 0:
            return None, 0.0
        if vehicle_crop.ndim < 2 or min(vehicle_crop.shape[:2]) < 8:
            return None, 0.0

        filtered = self._preprocess(vehicle_crop)
        try:
            results = self.reader.readtext(filtered)
        except Exception as exc:
            LOGGER.debug("OCR read failed: %s", exc)
            return None, 0.0

        best_text: Optional[str] = None
        highest_conf = 0.0
        for item in results or []:
            if not item or len(item) < 3:
                continue
            _bbox, text, conf = item[0], item[1], float(item[2])
            cleaned = self.clean_text(text)
            if not self.plate_pattern.match(cleaned):
                continue
            if conf > self.min_confidence and conf > highest_conf:
                highest_conf = conf
                best_text = cleaned
        return best_text, highest_conf
