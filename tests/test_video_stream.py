"""ThreadedCamera tests against a short synthetic file (no RTSP required)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_stream import ThreadedCamera, _looks_like_live_source


def _write_clip(path: Path, frames: int = 12, size: int = 32) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (size, size))
    assert writer.isOpened()
    for i in range(frames):
        img = np.full((size, size, 3), i * 10, dtype=np.uint8)
        writer.write(img)
    writer.release()


def test_live_source_detection():
    assert _looks_like_live_source("rtsp://admin:pass@10.0.0.8/stream")
    assert _looks_like_live_source(0)
    assert not _looks_like_live_source("videos/camera1.mp4")


def test_threaded_file_yields_frames(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    cam = ThreadedCamera(str(clip), name="unit", freeze_ttl=1.0)
    try:
        deadline = time.time() + 3.0
        got = False
        while time.time() < deadline:
            ret, frame = cam.read()
            if ret and frame is not None:
                got = True
                assert frame.shape[2] == 3
                break
            time.sleep(0.05)
        assert got
    finally:
        cam.release()
