"""Centralized spatio-temporal gallery for cross-camera person Re-ID.

Maps per-camera ByteTrack local IDs onto persistent Global IDs using
cosine similarity on OSNet embeddings plus camera topology / transit-time
constraints to suppress physically impossible matches.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from numpy.linalg import norm


CameraPair = Tuple[str, str]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    scale = float(norm(arr))
    if scale < 1e-12:
        return arr
    return arr / scale


@dataclass(frozen=True)
class TransitConstraint:
    """Minimum / maximum seconds to travel between two cameras."""

    min_transit_sec: float = 0.0
    max_transit_sec: float = 60.0
    overlapping: bool = False


@dataclass
class GlobalTrack:
    embedding: np.ndarray
    last_cam: str
    last_seen: float
    hit_count: int = 1
    last_local_id: int = -1
    last_score: float = 1.0


@dataclass(frozen=True)
class MatchResult:
    global_id: int
    is_new: bool
    score: float
    matched_existing: bool


class GlobalGalleryManager:
    """Thread-safe in-memory gallery of active global identities."""

    def __init__(
        self,
        similarity_threshold: float = 0.72,
        max_time_diff: float = 60.0,
        local_track_ttl: float = 8.0,
        ema_existing: float = 0.8,
        ema_matched: float = 0.7,
        topology: Optional[Mapping[CameraPair, TransitConstraint]] = None,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if max_time_diff <= 0.0:
            raise ValueError("max_time_diff must be positive")

        self.similarity_threshold = float(similarity_threshold)
        self.max_time_diff = float(max_time_diff)
        self.local_track_ttl = float(local_track_ttl)
        self.ema_existing = float(ema_existing)
        self.ema_matched = float(ema_matched)
        self.topology: Dict[CameraPair, TransitConstraint] = dict(topology or {})

        self.next_global_id = 1
        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        self._local_last_seen: Dict[Tuple[str, int], float] = {}
        self._lock = threading.RLock()

    def set_topology(
        self,
        edges: Iterable[Tuple[str, str, float, float, bool]],
    ) -> None:
        """Register directed transit constraints: (src, dst, min_s, max_s, overlapping)."""
        with self._lock:
            for src, dst, min_s, max_s, overlapping in edges:
                self.topology[(str(src), str(dst))] = TransitConstraint(
                    min_transit_sec=float(min_s),
                    max_transit_sec=float(max_s),
                    overlapping=bool(overlapping),
                )

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b) + 1e-6))

    def _transit_ok(self, last_cam: str, camera_id: str, time_diff: float) -> bool:
        if last_cam == camera_id:
            return True
        constraint = self.topology.get((last_cam, camera_id))
        if constraint is None:
            return time_diff <= self.max_time_diff
        if constraint.overlapping:
            return time_diff <= constraint.max_transit_sec
        if time_diff < constraint.min_transit_sec:
            return False
        return time_diff <= constraint.max_transit_sec

    def _gid_busy_on_other_camera(
        self,
        gid: int,
        camera_id: str,
        timestamp: float,
        occupy_window: float = 0.4,
    ) -> bool:
        track = self.global_tracks.get(gid)
        if track is None:
            return False
        constraint = self.topology.get((track.last_cam, camera_id))
        overlapping = bool(constraint and constraint.overlapping)
        if overlapping:
            return False
        if track.last_cam != camera_id and (timestamp - track.last_seen) < occupy_window:
            return True
        return False

    def _smooth_embedding(self, old: np.ndarray, new: np.ndarray, keep: float) -> np.ndarray:
        blended = keep * old + (1.0 - keep) * new
        return _l2_normalize(blended)

    def _expire_local_maps(self, timestamp: float) -> None:
        stale = [
            key
            for key, seen in self._local_last_seen.items()
            if timestamp - seen > self.local_track_ttl
        ]
        for key in stale:
            self.local_to_global.pop(key, None)
            self._local_last_seen.pop(key, None)

    def prune(self, timestamp: Optional[float] = None) -> int:
        """Drop gallery identities that have not been seen within max_time_diff."""
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            expired = [
                gid
                for gid, track in self.global_tracks.items()
                if now - track.last_seen > self.max_time_diff
            ]
            for gid in expired:
                self.global_tracks.pop(gid, None)
            dead_keys = [key for key, gid in self.local_to_global.items() if gid not in self.global_tracks]
            for key in dead_keys:
                self.local_to_global.pop(key, None)
                self._local_last_seen.pop(key, None)
            return len(expired)

    def match_or_register(
        self,
        camera_id: str,
        local_id: int,
        embedding: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> int:
        """Map a body embedding to a persistent Global ID."""
        return self.match_or_register_detailed(
            camera_id, local_id, embedding, timestamp=timestamp
        ).global_id

    def match_or_register_detailed(
        self,
        camera_id: str,
        local_id: int,
        embedding: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> MatchResult:
        if timestamp is None:
            timestamp = time.time()
        camera_id = str(camera_id)
        local_id = int(local_id)
        embedding = _l2_normalize(embedding)
        lookup_key = (camera_id, local_id)

        with self._lock:
            self._expire_local_maps(timestamp)
            self.prune(timestamp)

            if lookup_key in self.local_to_global:
                gid = self.local_to_global[lookup_key]
                track = self.global_tracks.get(gid)
                if track is not None:
                    track.embedding = self._smooth_embedding(
                        track.embedding, embedding, self.ema_existing
                    )
                    track.last_seen = timestamp
                    track.last_cam = camera_id
                    track.last_local_id = local_id
                    track.hit_count += 1
                    self._local_last_seen[lookup_key] = timestamp
                    return MatchResult(
                        global_id=gid,
                        is_new=False,
                        score=1.0,
                        matched_existing=True,
                    )
                self.local_to_global.pop(lookup_key, None)

            best_match_id: Optional[int] = None
            highest_score = -1.0

            for gid, track in self.global_tracks.items():
                time_diff = abs(timestamp - track.last_seen)
                if time_diff > self.max_time_diff:
                    continue
                if not self._transit_ok(track.last_cam, camera_id, time_diff):
                    continue
                if self._gid_busy_on_other_camera(gid, camera_id, timestamp):
                    continue
                score = self._cosine_similarity(embedding, track.embedding)
                if score > self.similarity_threshold and score > highest_score:
                    highest_score = score
                    best_match_id = gid

            if best_match_id is not None:
                gid = best_match_id
                track = self.global_tracks[gid]
                track.embedding = self._smooth_embedding(
                    track.embedding, embedding, self.ema_matched
                )
                track.last_seen = timestamp
                track.last_cam = camera_id
                track.last_local_id = local_id
                track.hit_count += 1
                track.last_score = highest_score
                is_new = False
                score = highest_score
            else:
                gid = self.next_global_id
                self.next_global_id += 1
                self.global_tracks[gid] = GlobalTrack(
                    embedding=embedding,
                    last_cam=camera_id,
                    last_seen=timestamp,
                    last_local_id=local_id,
                    last_score=1.0,
                )
                is_new = True
                score = 1.0

            self.local_to_global[lookup_key] = gid
            self._local_last_seen[lookup_key] = timestamp
            return MatchResult(
                global_id=gid,
                is_new=is_new,
                score=score,
                matched_existing=not is_new,
            )

    def active_count(self, timestamp: Optional[float] = None) -> int:
        now = time.time() if timestamp is None else float(timestamp)
        with self._lock:
            return sum(1 for t in self.global_tracks.values() if now - t.last_seen <= self.max_time_diff)

    def snapshot(self) -> Dict[int, Dict[str, object]]:
        with self._lock:
            return {
                gid: {
                    "last_cam": track.last_cam,
                    "last_seen": track.last_seen,
                    "hit_count": track.hit_count,
                    "last_local_id": track.last_local_id,
                    "last_score": track.last_score,
                }
                for gid, track in self.global_tracks.items()
            }
