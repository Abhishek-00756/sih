#!/usr/bin/env python3
"""Multi-camera Shared Perception loop: YOLOv8 + ByteTrack + OSNet + Global IDs.

Optionally forwards tracks into the existing geofence engine so Global IDs
(not per-camera local IDs) drive ENTER/EXIT events.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor import PersonFeatureExtractor
from gallery_manager import GlobalGalleryManager
from geofence_manager import GeofenceManager
from video_stream import ThreadedCamera

LOGGER = logging.getLogger("perception")

SourceMap = Mapping[str, Union[str, int]]


def _load_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("perception config must be a mapping")
    return data


def _parse_sources(raw: Iterable[str]) -> Dict[str, Union[str, int]]:
    sources: Dict[str, Union[str, int]] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"camera source must be name=path, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if value.isdigit():
            sources[name] = int(value)
        else:
            sources[name] = value
    return sources


def _sources_from_config(cfg: dict) -> Dict[str, Union[str, int]]:
    cameras = cfg.get("cameras") or {}
    sources: Dict[str, Union[str, int]] = {}
    for cam_id, spec in cameras.items():
        if isinstance(spec, (str, int)):
            sources[str(cam_id)] = spec
            continue
        if not isinstance(spec, dict) or "source" not in spec:
            raise ValueError(f"camera {cam_id} is missing a source")
        src = spec["source"]
        sources[str(cam_id)] = int(src) if isinstance(src, str) and src.isdigit() else src
    return sources


def _topology_from_config(cfg: dict) -> List[Tuple[str, str, float, float, bool]]:
    edges: List[Tuple[str, str, float, float, bool]] = []
    cameras = cfg.get("cameras") or {}
    for src, spec in cameras.items():
        if not isinstance(spec, dict):
            continue
        neighbors = spec.get("neighbors") or {}
        for dst, rule in neighbors.items():
            rule = rule or {}
            edges.append(
                (
                    str(src),
                    str(dst),
                    float(rule.get("min_transit_sec", 0.0)),
                    float(rule.get("max_transit_sec", 60.0)),
                    bool(rule.get("overlapping", False)),
                )
            )
    return edges


def _color_for_id(gid: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(gid * 9973)
    return tuple(int(x) for x in rng.integers(40, 255, size=3))


def _open_captures(camera_sources: SourceMap) -> Dict[str, ThreadedCamera]:
    caps: Dict[str, ThreadedCamera] = {}
    for cam_id, src in camera_sources.items():
        cap = ThreadedCamera(src, name=str(cam_id), buffer_size=1)
        deadline = time.time() + 5.0
        opened = cap.isOpened()
        while not opened and time.time() < deadline:
            time.sleep(0.05)
            opened = cap.isOpened()
        if not opened:
            cap.release()
            raise RuntimeError(f"failed to open camera {cam_id}: {src}")
        caps[str(cam_id)] = cap
    return caps


def _overlay_geofence(cfg: dict, zones_path: Optional[Path], cooldown: float) -> Optional[GeofenceManager]:
    inline = cfg.get("restricted_zone")
    if inline:
        return GeofenceManager(zone_points=inline, cooldown_seconds=cooldown)
    if zones_path is None or not zones_path.exists():
        return GeofenceManager(
            zone_points=[(100, 400), (500, 400), (600, 600), (50, 600)],
            cooldown_seconds=cooldown,
        )
    try:
        return GeofenceManager.from_zone_store(zones_path, cooldown_seconds=cooldown, restricted_only=True)
    except ValueError:
        LOGGER.warning("No restricted polygons in %s; using demo zone", zones_path)
        return GeofenceManager(
            zone_points=[(100, 400), (500, 400), (600, 600), (50, 600)],
            cooldown_seconds=cooldown,
        )


def _maybe_geofence(zones_path: Optional[Path]):
    if zones_path is None or not zones_path.exists():
        return None, None, None
    from geofence.adapter import TrackingInterface
    from geofence.engine import EngineConfig, GeofenceEngine
    from geofence.zones import ZoneStore

    store = ZoneStore.load(zones_path)
    engine = GeofenceEngine(store, EngineConfig(confirm_frames=3, stale_timeout_sec=5.0))
    return store, engine, TrackingInterface()


def run_multi_camera_tracking(
    camera_sources: SourceMap,
    similarity_threshold: float = 0.75,
    max_time_diff: float = 120.0,
    detector_weights: str = "yolov8n.pt",
    osnet_model: str = "osnet_x1_0",
    person_class: int = 0,
    headless: bool = False,
    display: bool = False,
    max_frames: int = 0,
    topology: Optional[Iterable[Tuple[str, str, float, float, bool]]] = None,
    geofence_zones: Optional[Path] = None,
    enable_geofence: bool = False,
    use_fallback_extractor: bool = False,
    output_dir: Optional[Path] = None,
    overlay_geofence: Optional[GeofenceManager] = None,
    geofence_cfg: Optional[dict] = None,
    cooldown_seconds: float = 60.0,
) -> Dict[str, object]:
    from ultralytics import YOLO

    # One detector/tracker per camera so ByteTrack IDs never mix across streams.
    detectors = {cam_id: YOLO(detector_weights) for cam_id in camera_sources}
    extractor = PersonFeatureExtractor(model_name=osnet_model, use_fallback=use_fallback_extractor)
    gallery = GlobalGalleryManager(
        similarity_threshold=similarity_threshold,
        max_time_diff=max_time_diff,
    )
    if topology:
        gallery.set_topology(topology)

    zone_store, geo_engine, geo_iface = (None, None, None)
    if enable_geofence:
        zone_store, geo_engine, geo_iface = _maybe_geofence(geofence_zones)
        if overlay_geofence is None:
            overlay_geofence = _overlay_geofence(geofence_cfg or {}, geofence_zones, cooldown_seconds)

    caps = _open_captures(camera_sources)
    writers: Dict[str, cv2.VideoWriter] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting Multi-Camera Shared Perception Engine...")
    frame_idx = 0
    events_emitted = 0
    assignments = 0

    try:
        while True:
            any_ok = False
            current_timestamp = time.time()
            for cam_id, cap in caps.items():
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                any_ok = True
                geo_cam = (geofence_cfg or {}).get("geofence_camera_map", {}).get(cam_id, cam_id)
                if overlay_geofence is not None:
                    overlay_geofence.draw_zones(frame, camera_id=geo_cam)

                results = detectors[cam_id].track(
                    frame,
                    persist=True,
                    classes=[person_class],
                    tracker="bytetrack.yaml",
                    verbose=False,
                )

                geo_records = []
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    scores = results[0].boxes.conf.cpu().numpy()

                    for box, local_id, score in zip(boxes, track_ids, scores):
                        x1, y1, x2, y2 = box
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        person_crop = frame[y1:y2, x1:x2]

                        embedding = extractor.extract(person_crop)
                        if embedding is None:
                            continue

                        result = gallery.match_or_register_detailed(
                            camera_id=cam_id,
                            local_id=int(local_id),
                            embedding=embedding,
                            timestamp=current_timestamp,
                        )
                        assignments += 1
                        color = _color_for_id(result.global_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"GID {result.global_id} L{local_id}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                        if overlay_geofence is not None:
                            overlay_geofence.check_intrusion(
                                frame,
                                (x1, y1, x2, y2),
                                result.global_id,
                                camera_id=geo_cam,
                                timestamp=current_timestamp,
                            )
                        geo_records.append(
                            {
                                "camera_id": cam_id,
                                "object_id": result.global_id,
                                "object_type": "person",
                                "bbox": [x1, y1, x2, y2],
                                "confidence": float(score),
                                "timestamp": current_timestamp,
                            }
                        )

                if enable_geofence and geo_engine is not None and geo_iface is not None:
                    tracks = geo_iface.parse_frame(
                        geo_records, camera_id=cam_id, timestamp=current_timestamp
                    )
                    events = geo_engine.process_frame(
                        tracks, camera_id=cam_id, timestamp=current_timestamp
                    )
                    events_emitted += len(events)
                    for ev in events:
                        LOGGER.info("GEOFENCE %s", json.dumps(ev.to_dict()))
                if overlay_geofence is not None:
                    events_emitted += len(overlay_geofence.pop_alerts())

                if display and not headless:
                    cv2.imshow(f"Feed: {cam_id}", frame)

                if output_dir is not None:
                    writer = writers.get(cam_id)
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(output_dir / f"{cam_id}.mp4"), fourcc, 20.0, (w, h)
                        )
                        writers[cam_id] = writer
                    writer.write(frame)

            if display and not headless:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
            if not any_ok:
                LOGGER.info("All camera streams ended.")
                break
    finally:
        for cap in caps.values():
            cap.release()
        for writer in writers.values():
            writer.release()
        if display and not headless:
            cv2.destroyAllWindows()

    summary = {
        "frames": frame_idx,
        "assignments": assignments,
        "active_global_ids": gallery.active_count(),
        "geofence_events": events_emitted,
        "gallery": gallery.snapshot(),
    }
    LOGGER.info("Done: %s", json.dumps({k: v for k, v in summary.items() if k != "gallery"}))
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared Perception multi-camera Re-ID")
    parser.add_argument("--config", default=str(ROOT / "configs" / "perception.yaml"))
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="camera source as name=path (repeatable). Overrides config cameras.",
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-time-diff", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fallback-extractor", action="store_true")
    parser.add_argument("--enable-geofence", action="store_true")
    parser.add_argument("--cooldown", type=float, default=None)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = _load_config(Path(args.config)) if args.config else {}
    sources = _parse_sources(args.source) if args.source else _sources_from_config(cfg)
    if not sources:
        LOGGER.error("No camera sources configured. Pass --source Cam=path or edit configs/perception.yaml")
        return 2

    topology = _topology_from_config(cfg)
    zones = Path(cfg.get("geofence_zones", ROOT / "configs" / "zones.json"))
    if not zones.is_absolute():
        zones = ROOT / zones

    run_multi_camera_tracking(
        camera_sources=sources,
        similarity_threshold=float(args.threshold or cfg.get("similarity_threshold", 0.75)),
        max_time_diff=float(args.max_time_diff or cfg.get("max_time_diff", 120.0)),
        detector_weights=args.weights or cfg.get("detector_weights", "yolov8n.pt"),
        osnet_model=cfg.get("osnet_model", "osnet_x1_0"),
        person_class=int(cfg.get("person_class", 0)),
        headless=bool(args.headless or cfg.get("headless", False)),
        display=bool(args.display or cfg.get("display", False)),
        max_frames=args.max_frames,
        topology=topology,
        geofence_zones=zones,
        enable_geofence=bool(args.enable_geofence or cfg.get("enable_geofence", False)),
        use_fallback_extractor=args.fallback_extractor,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        geofence_cfg=cfg,
        cooldown_seconds=float(args.cooldown or cfg.get("geofence_cooldown", 60.0)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
