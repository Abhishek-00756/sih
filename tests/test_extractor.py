"""Extractor tests using the histogram fallback (no torchreid required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor import FEATURE_DIM, PersonFeatureExtractor


def test_rejects_tiny_or_empty_crops():
    ext = PersonFeatureExtractor(use_fallback=True)
    assert ext.extract(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert ext.extract(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_fallback_returns_normalized_512d():
    ext = PersonFeatureExtractor(use_fallback=True)
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    crop[:, :, 2] = 200
    vec = ext.extract(crop)
    assert vec is not None
    assert vec.shape == (FEATURE_DIM,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


def test_similar_crops_have_high_cosine():
    ext = PersonFeatureExtractor(use_fallback=True)
    red = np.zeros((80, 40, 3), dtype=np.uint8)
    red[:, :, 2] = 220
    red2 = red.copy()
    red2[:5] = 210
    blue = np.zeros((80, 40, 3), dtype=np.uint8)
    blue[:, :, 0] = 220
    a = ext.extract(red)
    b = ext.extract(red2)
    c = ext.extract(blue)
    assert float(np.dot(a, b)) > float(np.dot(a, c))
