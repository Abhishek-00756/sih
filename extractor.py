"""OSNet feature extractor for person Re-ID.

Crops a detected body, resizes to 256x128, and returns an L2-normalized
512-d embedding. Falls back to a deterministic color histogram embedding
when torchreid/torch are unavailable (tests / dry-run).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger("perception.extractor")

OSNET_INPUT_SIZE = (256, 128)
MIN_CROP_H = 20
MIN_CROP_W = 20
FEATURE_DIM = 512


def _histogram_embedding(bgr_crop: np.ndarray, dim: int = FEATURE_DIM) -> np.ndarray:
    """Lightweight fallback embedding used when OSNet weights are not loaded."""
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [128], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [128], [0, 256]).flatten()
    vec = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    if vec.size < dim:
        vec = np.pad(vec, (0, dim - vec.size))
    else:
        vec = vec[:dim]
    scale = float(np.linalg.norm(vec))
    return vec / (scale + 1e-6)


class PersonFeatureExtractor:
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: Optional[str] = None,
        use_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.use_fallback = use_fallback
        self.model = None
        self.transform = None
        self.device = device or "cpu"

        if use_fallback:
            LOGGER.warning("Using histogram fallback embeddings (not production OSNet).")
            return

        try:
            import torch
            import torchvision.transforms as T
            import torchreid
        except ImportError as exc:
            LOGGER.warning("torchreid unavailable (%s); using histogram fallback.", exc)
            self.use_fallback = True
            return

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1041,
            pretrained=True,
        )
        self.model.eval()
        self.model.to(self.device)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(OSNET_INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._torch = torch
        LOGGER.info("Loaded %s on %s", model_name, self.device)

    def extract(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """Return a 1D L2-normalized embedding, or None if the crop is invalid."""
        if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
            return None
        if bgr_crop.ndim != 3 or bgr_crop.shape[0] < MIN_CROP_H or bgr_crop.shape[1] < MIN_CROP_W:
            return None

        if self.use_fallback or self.model is None:
            return _histogram_embedding(bgr_crop)

        rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            features = self.model(tensor)
            features = features.cpu().numpy().flatten().astype(np.float32)
        scale = float(np.linalg.norm(features))
        return features / (scale + 1e-6)

    def extract_batch(self, crops: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        return [self.extract(crop) for crop in crops]
