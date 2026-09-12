"""CLAHE pre-processing for night, fog, and low-contrast CCTV frames.

Equalizes only the Lab lightness channel so color stays stable while
local contrast is lifted before YOLO / OCR / Haar run.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

DEFAULT_CLIP_LIMIT = 3.0
DEFAULT_TILE_GRID = (8, 8)
AUTO_MEAN_L_THRESHOLD = 90.0


class VideoEnhancer:
    def __init__(
        self,
        clip_limit: float = DEFAULT_CLIP_LIMIT,
        tile_grid_size: Tuple[int, int] = DEFAULT_TILE_GRID,
        auto_mean_l: float = AUTO_MEAN_L_THRESHOLD,
    ) -> None:
        if clip_limit <= 0:
            raise ValueError("clip_limit must be positive")
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))
        self.auto_mean_l = float(auto_mean_l)
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def enhance_frame(self, frame: Optional[np.ndarray], mode: str = "auto") -> Optional[np.ndarray]:
        """Enhance visibility. mode is auto (skip bright frames), always, or off."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
        if frame.ndim != 3 or frame.shape[2] != 3:
            return frame
        if mode == "off":
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        if mode == "auto" and float(np.mean(l_channel)) >= self.auto_mean_l:
            return frame

        equalized = self.clahe.apply(l_channel)
        merged = cv2.merge((equalized, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
