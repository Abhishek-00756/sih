"""CLAHE night/fog enhancer tests (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enhancer import VideoEnhancer


def test_always_mode_changes_dark_frame():
    enhancer = VideoEnhancer()
    dark = np.full((64, 64, 3), 20, dtype=np.uint8)
    out = enhancer.enhance_frame(dark, mode="always")
    assert out is not None
    assert out.shape == dark.shape
    assert int(out.mean()) > int(dark.mean())


def test_auto_skips_bright_frames():
    enhancer = VideoEnhancer(auto_mean_l=90.0)
    bright = np.full((48, 48, 3), 220, dtype=np.uint8)
    out = enhancer.enhance_frame(bright, mode="auto")
    assert np.array_equal(out, bright)


def test_off_and_empty_are_noop():
    enhancer = VideoEnhancer()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    assert enhancer.enhance_frame(frame, mode="off") is frame
    assert enhancer.enhance_frame(None) is None
