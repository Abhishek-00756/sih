"""Face capture tests using an injected fake Haar cascade (no XML required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_manager import FaceManager


class FakeCascade:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0
        self.last_image = None

    def detectMultiScale(self, image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)):
        self.calls += 1
        self.last_image = image
        return self.faces


def _crop(h=120, w=80):
    return np.full((h, w, 3), 40, dtype=np.uint8)


def test_saves_one_face_per_global_id(tmp_path: Path):
    cascade = FakeCascade([(10, 10, 40, 40)])
    mgr = FaceManager(save_dir=tmp_path, cascade=cascade)
    crop = _crop()
    assert mgr.detect_and_save_face(crop, 7, "Cam_1_Outpost", timestamp=1_710_000_000) is True
    assert mgr.already_captured(7)
    assert mgr.last_saved is not None
    assert mgr.last_saved.exists()
    assert cascade.calls == 1
    assert cascade.last_image.ndim == 2
    assert mgr.detect_and_save_face(crop, 7, "Cam_1_Outpost", timestamp=1_710_000_001) is False
    assert cascade.calls == 1
    saved = list(tmp_path.glob("face_GID_7_cam_Cam_1_Outpost_*.jpg"))
    assert len(saved) == 1


def test_picks_largest_face(tmp_path: Path):
    cascade = FakeCascade([(5, 5, 20, 20), (15, 10, 50, 50)])
    mgr = FaceManager(save_dir=tmp_path, cascade=cascade, head_margin=0.0)
    crop = _crop(h=80, w=80)
    assert mgr.detect_and_save_face(crop, 3, "Gate") is True
    assert mgr.last_saved is not None


def test_skips_empty_crop_and_no_faces(tmp_path: Path):
    cascade = FakeCascade([])
    mgr = FaceManager(save_dir=tmp_path, cascade=cascade)
    assert mgr.detect_and_save_face(None, 1, "cam") is False
    assert mgr.detect_and_save_face(np.zeros((0, 0, 3), dtype=np.uint8), 1, "cam") is False
    assert mgr.detect_and_save_face(_crop(), 1, "cam") is False
    assert not mgr.already_captured(1)


def test_disabled_cascade_is_noop(tmp_path: Path):
    mgr = FaceManager(save_dir=tmp_path, cascade=FakeCascade([(0, 0, 40, 40)]))
    mgr.cascade = None
    assert mgr.detect_and_save_face(_crop(), 9, "cam") is False


def test_rejects_invalid_constructor_args(tmp_path: Path):
    try:
        FaceManager(save_dir=tmp_path, cascade=FakeCascade([]), scale_factor=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        FaceManager(save_dir=tmp_path, cascade=FakeCascade([]), min_neighbors=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
