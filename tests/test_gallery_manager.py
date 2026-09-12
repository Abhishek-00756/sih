"""Unit tests for the cross-camera gallery manager (no GPU / OSNet required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gallery_manager import GlobalGalleryManager, TransitConstraint


def _emb(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-6)


def test_same_local_track_keeps_global_id():
    gallery = GlobalGalleryManager(similarity_threshold=0.72, max_time_diff=60.0)
    a = _emb(1)
    gid1 = gallery.match_or_register("cam_a", 7, a, timestamp=1.0)
    gid2 = gallery.match_or_register("cam_a", 7, a, timestamp=1.2)
    assert gid1 == gid2 == 1


def test_cross_camera_match_reuses_global_id():
    gallery = GlobalGalleryManager(similarity_threshold=0.72, max_time_diff=60.0)
    person = _emb(42)
    noise = _emb(99)
    gid_a = gallery.match_or_register("cam_a", 1, person, timestamp=10.0)
    gid_b = gallery.match_or_register("cam_b", 9, person, timestamp=12.0)
    gid_c = gallery.match_or_register("cam_b", 10, noise, timestamp=12.1)
    assert gid_a == gid_b
    assert gid_c != gid_a


def test_stale_identity_is_not_matched():
    gallery = GlobalGalleryManager(similarity_threshold=0.72, max_time_diff=5.0)
    person = _emb(3)
    first = gallery.match_or_register("cam_a", 1, person, timestamp=1.0)
    later = gallery.match_or_register("cam_b", 2, person, timestamp=20.0)
    assert first == 1
    assert later == 2


def test_transit_constraint_blocks_impossible_jump():
    gallery = GlobalGalleryManager(similarity_threshold=0.5, max_time_diff=60.0)
    gallery.set_topology(
        [("cam_a", "cam_b", 8.0, 40.0, False)]
    )
    person = _emb(11)
    gid_a = gallery.match_or_register("cam_a", 1, person, timestamp=0.0)
    too_fast = gallery.match_or_register("cam_b", 2, person, timestamp=1.0)
    ok = gallery.match_or_register("cam_b", 3, person, timestamp=10.0)
    assert gid_a != too_fast
    assert ok == gid_a


def test_prune_drops_expired_tracks():
    gallery = GlobalGalleryManager(similarity_threshold=0.72, max_time_diff=5.0)
    gallery.match_or_register("cam_a", 1, _emb(5), timestamp=1.0)
    dropped = gallery.prune(timestamp=20.0)
    assert dropped == 1
    assert gallery.active_count(timestamp=20.0) == 0


def test_snapshot_contains_last_camera():
    gallery = GlobalGalleryManager()
    gallery.match_or_register("outpost", 4, _emb(8), timestamp=3.0)
    snap = gallery.snapshot()
    assert snap[1]["last_cam"] == "outpost"
    assert snap[1]["hit_count"] == 1
