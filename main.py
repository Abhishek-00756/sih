#!/usr/bin/env python3
"""Multi-camera Shared Perception loop: YOLOv8 + ByteTrack + OSNet + Global IDs.

Per-camera runtime switches come from configs/cameras.json when available.
ANPR, geofence, tripwire, enhancement and face-recognition participation can
therefore be enabled independently for each camera. Face recognition itself
uses one shared registry across the process.
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

from activity_analyzer import (
    COCO_CLASSES,
    PERSON_CLASS,
    TRACK_CLASSES,
    ActivityAnalyzer,
)
from alert_logger import AlertLogger
from anpr_manager import ANPRManager
from enhancer import VideoEnhancer
from extractor import PersonFeatureExtractor
from face_intelligence import SharedFaceRegistry
from face_manager import FaceManager
from gallery_manager import GlobalGalleryManager
from geofence_manager import GeofenceManager, VirtualTripwire
from video_stream import ThreadedCamera

LOGGER = logging.getLogger("perception")

SourceMap = Mapping[str, Union[str, int]]
FeatureMap = Mapping[str, Mapping[str, bool]]


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


def _load_dashboard_feature_config(path: Optional[Path] = None) -> Dict[str, Dict[str, bool]]:
    """Load dashboard camera feature switches keyed by both camera ID and source."""
    config_path = path or ROOT / "configs" / "cameras.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not load dashboard camera config %s: %s", config_path, exc)
        return {}

    result: Dict[str, Dict[str, bool]] = {}
    for camera_id, spec in (payload.get("cameras") or {}).items():
        if not isinstance(spec, dict):
            continue
        features = spec.get("features") or {}
        merged = {
            "ai_enabled": bool(spec.get("ai_enabled", True)),
            "geofence": bool(features.get("geofence", True)),
            "tripwire": bool(features.get("tripwire", True)),
            "anpr": bool(features.get("anpr", True)),
            "face_recognition": bool(features.get("face_recognition", True)),
            "enhancement": bool(features.get("enhancement", True)),
        }
        result[str(camera_id)] = merged
        source = spec.get("source")
        if source is not None:
            result[f"source::{source}"] = merged
    return result


def _features_for_camera(
    camera_id: str,
    source: Union[str, int],
    dashboard_features: Optional[FeatureMap],
) -> Dict[str, bool]:
    """Resolve dashboard settings by camera ID, then by configured source."""
    if not dashboard_features:
        return {
            "ai_enabled": True,
            "geofence": True,
            "tripwire": True,
            "anpr": True,
            "face_recognition": True,
            "enhancement": True,
        }
    if camera_id in dashboard_features:
        return dict(dashboard_features[camera_id])
    return dict(dashboard_features.get(f"source::{source}") or {
        "ai_enabled": True,
        "geofence": True,
        "tripwire": True,
        "anpr": True,
        "face_recognition": True,
        "enhancement": True,
    })


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


def _tripwires_from_config(cfg: dict, cooldown: float) -> List[VirtualTripwire]:
    raw = cfg.get("tripwires") or []
    wires: List[VirtualTripwire] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pt1 = item.get("pt1") or item.get("line_pt1")
        pt2 = item.get("pt2") or item.get("line_pt2")
        if not pt1 or not pt2:
            continue
        wires.append(
            VirtualTripwire(
                pt1,
                pt2,
                wire_id=str(item.get("id") or item.get("wire_id") or "border"),
                camera_id=item.get("camera_id"),
                cooldown_seconds=float(item.get("cooldown_seconds", cooldown)),
                inbound_positive=bool(item.get("inbound_positive", True)),
            )
        )
    return wires


def _overlay_geofence(
    cfg: dict,
    zones_path: Optional[Path],
    cooldown: float,
    logger: Optional[AlertLogger] = None,
) -> Optional[GeofenceManager]:
    wires = _tripwires_from_config(cfg, cooldown)
    inline = cfg.get("restricted_zone")
    if inline:
        return GeofenceManager(
            zone_points=inline,
            cooldown_seconds=cooldown,
            tripwires=wires,
            logger=logger,
        )
    if zones_path is None or not zones_path.exists():
        return GeofenceManager(
            zone_points=[(100, 400), (500, 400), (600, 600), (50, 600)],
            cooldown_seconds=cooldown,
            tripwires=wires,
            logger=logger,
        )
    try:
        mgr = GeofenceManager.from_zone_store(zones_path, cooldown_seconds=cooldown, restricted_only=True)
        mgr.logger = logger
        mgr.tripwires = wires
        return mgr
    except ValueError:
        LOGGER.warning("No restricted polygons in %s; using demo zone", zones_path)
        return GeofenceManager(
            zone_points=[(100, 400), (500, 400), (600, 600), (50, 600)],
            cooldown_seconds=cooldown,
            tripwires=wires,
            logger=logger,
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
    person_class: int = PERSON_CLASS,
    track_classes: Optional[Iterable[int]] = None,
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
    alert_db: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
    dwell_threshold: float = 30.0,
    enable_loitering: bool = True,
    enable_anpr: bool = True,
    anpr_min_confidence: float = 0.5,
    anpr_min_width: int = 150,
    anpr_min_height: int = 150,
    anpr_gpu: Optional[bool] = None,
    enable_face_capture: bool = True,
    face_save_dir: Optional[Path] = None,
    enable_enhance: bool = True,
    enhance_mode: str = "auto",
    dashboard_features: Optional[FeatureMap] = None,
    face_registry_dir: Optional[Path] = None,
    face_registry_threshold: float = 0.45,
    face_registry_model: str = "buffalo_l",
) -> Dict[str, object]:
    from ultralytics import YOLO

    detect_classes = list(track_classes) if track_classes is not None else list(TRACK_CLASSES)
    resolved_features = {
        cam_id: _features_for_camera(cam_id, camera_sources[cam_id], dashboard_features)
        for cam_id in camera_sources
    }

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
    alert_logger: Optional[AlertLogger] = None
    analyzer: Optional[ActivityAnalyzer] = None
    if enable_geofence or enable_loitering:
        db_path = alert_db or ROOT / "border_alerts.db"
        snap_path = snapshot_dir or ROOT / "alert_snapshots"
        alert_logger = AlertLogger(db_path=db_path, snapshot_dir=snap_path)
    if enable_geofence:
        zone_store, geo_engine, geo_iface = _maybe_geofence(geofence_zones)
        if overlay_geofence is None:
            overlay_geofence = _overlay_geofence(
                geofence_cfg or {},
                geofence_zones,
                cooldown_seconds,
                logger=alert_logger,
            )
        elif overlay_geofence.logger is None:
            overlay_geofence.logger = alert_logger
    if enable_loitering:
        analyzer = ActivityAnalyzer(dwell_threshold=dwell_threshold, logger=alert_logger)

    anpr: Optional[ANPRManager] = None
    known_plates: Dict[str, str] = {}
    plates_recognized = 0
    if enable_anpr and any(spec.get("anpr", True) for spec in resolved_features.values()):
        anpr = ANPRManager(min_confidence=anpr_min_confidence, gpu=anpr_gpu)

    face_mgr: Optional[FaceManager] = None
    faces_captured = 0
    if enable_face_capture and any(spec.get("face_recognition", True) for spec in resolved_features.values()):
        face_mgr = FaceManager(save_dir=face_save_dir or ROOT / "face_database")

    shared_face_registry: Optional[SharedFaceRegistry] = None
    if any(spec.get("face_recognition", True) for spec in resolved_features.values()):
        shared_face_registry = SharedFaceRegistry(
            registry_dir=face_registry_dir or ROOT / "face_registry",
            threshold=face_registry_threshold,
            model_name=face_registry_model,
        )
        if not shared_face_registry.available:
            LOGGER.warning("Shared face recognition is unavailable; face participation is idle.")

    enhancer: Optional[VideoEnhancer] = VideoEnhancer() if enable_enhance else None

    caps = _open_captures(camera_sources)
    writers: Dict[str, cv2.VideoWriter] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting Multi-Camera Shared Perception Engine...")
    frame_idx = 0
    events_emitted = 0
    assignments = 0
    loitering_alerts = 0

    try:
        while True:
            any_ok = False
            current_timestamp = time.time()
            for cam_id, cap in caps.items():
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                any_ok = True
                features = resolved_features.get(cam_id, {})
                ai_enabled = features.get("ai_enabled", True)
                camera_geofence = enable_geofence and features.get("geofence", True)
                camera_tripwire = camera_geofence and features.get("tripwire", True)
                camera_anpr = enable_anpr and features.get("anpr", True)
                camera_face = enable_face_capture and features.get("face_recognition", True)
                camera_enhance = enable_enhance and features.get("enhancement", True)

                if enhancer is not None and camera_enhance:
                    frame = enhancer.enhance_frame(frame, mode=enhance_mode)
                geo_cam = (geofence_cfg or {}).get("geofence_camera_map", {}).get(cam_id, cam_id)
                if overlay_geofence is not None and camera_geofence:
                    overlay_geofence.draw_zones(frame, camera_id=geo_cam)

                results = []
                if ai_enabled:
                    results = detectors[cam_id].track(
                        frame,
                        persist=True,
                        classes=detect_classes,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )

                geo_records = []
                current_active_ids: List[Union[int, str]] = []
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    scores = results[0].boxes.conf.cpu().numpy()
                    class_ids = (
                        results[0].boxes.cls.cpu().numpy().astype(int)
                        if results[0].boxes.cls is not None
                        else np.full(len(track_ids), person_class, dtype=int)
                    )

                    for box, local_id, score, cls_id in zip(boxes, track_ids, scores, class_ids):
                        x1, y1, x2, y2 = box
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        entity_type = COCO_CLASSES.get(int(cls_id), "Unknown")
                        bbox = (x1, y1, x2, y2)

                        if int(cls_id) == person_class:
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
                            display_id: Union[int, str] = result.global_id
                            entity_id: Union[int, str] = result.global_id
                            color = _color_for_id(result.global_id)
                            label = f"{entity_type} GID {result.global_id} L{local_id}"

                            if camera_face:
                                if face_mgr is not None:
                                    saved = face_mgr.detect_and_save_face(
                                        person_crop,
                                        result.global_id,
                                        cam_id,
                                        timestamp=current_timestamp,
                                    )
                                    if saved:
                                        faces_captured += 1
                                if shared_face_registry is not None and shared_face_registry.available:
                                    match = shared_face_registry.match(person_crop)
                                    if match:
                                        label += f" | {match['display_name']} {match['similarity']:.0%}"

                            if overlay_geofence is not None and camera_geofence:
                                overlay_geofence.check_intrusion(
                                    frame,
                                    bbox,
                                    result.global_id,
                                    camera_id=geo_cam,
                                    timestamp=current_timestamp,
                                    source_camera_id=cam_id,
                                )
                                if camera_tripwire:
                                    overlay_geofence.check_tripwire(
                                        frame,
                                        bbox,
                                        result.global_id,
                                        camera_id=geo_cam,
                                        timestamp=current_timestamp,
                                        source_camera_id=cam_id,
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
                        else:
                            entity_id = f"V-{int(local_id)}"
                            display_id = entity_id
                            color = (255, 165, 0)
                            plate_key = f"{cam_id}:{entity_id}"
                            plate_text = known_plates.get(plate_key)
                            if (
                                camera_anpr
                                and anpr is not None
                                and plate_text is None
                                and anpr.crop_is_readable(
                                    x2 - x1,
                                    y2 - y1,
                                    min_width=anpr_min_width,
                                    min_height=anpr_min_height,
                                )
                            ):
                                vehicle_crop = frame[y1:y2, x1:x2]
                                text, conf = anpr.read_license_plate(vehicle_crop)
                                if text:
                                    known_plates[plate_key] = text
                                    plate_text = text
                                    plates_recognized += 1
                                    LOGGER.info(
                                        "[ANPR] Camera %s recognized plate %s (conf=%.2f) on %s",
                                        cam_id,
                                        plate_text,
                                        conf,
                                        entity_id,
                                    )
                            if plate_text:
                                label = f"Plate: {plate_text}"
                            else:
                                label = f"{entity_type} {display_id}"
                            geo_records.append(
                                {
                                    "camera_id": cam_id,
                                    "object_id": entity_id,
                                    "object_type": entity_type.lower(),
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": float(score),
                                    "timestamp": current_timestamp,
                                    "plate": plate_text,
                                }
                            )

                        current_active_ids.append(entity_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                        if analyzer is not None:
                            analyzer.analyze_behavior(
                                frame,
                                bbox,
                                entity_id,
                                entity_type=entity_type,
                                camera_id=cam_id,
                                timestamp=current_timestamp,
                                draw=True,
                            )

                if analyzer is not None:
                    analyzer.cleanup_stale_tracks(current_active_ids, camera_id=cam_id)
                    loitering_alerts += len(analyzer.pop_alerts())

                if camera_geofence and geo_engine is not None and geo_iface is not None:
                    tracks = geo_iface.parse_frame(
                        geo_records, camera_id=cam_id, timestamp=current_timestamp
                    )
                    events = geo_engine.process_frame(
                        tracks, camera_id=cam_id, timestamp=current_timestamp
                    )
                    events_emitted += len(events)
                    for ev in events:
                        LOGGER.info("GEOFENCE %s", json.dumps(ev.to_dict()))
                        if alert_logger is None:
                            continue
                        ev_type = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                        if ev_type != "GEOFENCE_ENTER":
                            continue
                        try:
                            gid = int(ev.object_id)
                        except (TypeError, ValueError):
                            gid = 0
                        alert_logger.log_intrusion(
                            camera_id=cam_id,
                            global_id=gid,
                            frame=frame,
                            bbox=ev.bounding_box,
                            timestamp=ev.timestamp,
                            alert_type=ev_type,
                            zone_id=ev.zone_id,
                            zone_name=ev.zone_name,
                            footprint=ev.ground_contact_position,
                        )
                if overlay_geofence is not None and camera_geofence:
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
        "loitering_alerts": loitering_alerts,
        "plates_recognized": plates_recognized,
        "known_plates": dict(known_plates),
        "faces_captured": faces_captured,
        "face_registry": shared_face_registry.status() if shared_face_registry is not None else None,
        "camera_features": resolved_features,
        "gallery": gallery.snapshot(),
    }
    LOGGER.info(
        "Done: %s",
        json.dumps({k: v for k, v in summary.items() if k not in ("gallery", "known_plates")}),
    )
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
    parser.add_argument("--verify-ledger", action="store_true")
    parser.add_argument("--dwell-threshold", type=float, default=None)
    parser.add_argument("--no-loitering", action="store_true")
    parser.add_argument("--enable-anpr", action="store_true")
    parser.add_argument("--no-anpr", action="store_true")
    parser.add_argument("--anpr-min-confidence", type=float, default=None)
    parser.add_argument("--enable-face-capture", action="store_true")
    parser.add_argument("--no-face-capture", action="store_true")
    parser.add_argument("--face-dir", default="")
    parser.add_argument("--no-enhance", action="store_true")
    parser.add_argument("--enhance-mode", default="")
    parser.add_argument("--cooldown", type=float, default=None)
    parser.add_argument("--alert-db", default="")
    parser.add_argument("--snapshot-dir", default="")
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
    if args.verify_ledger:
        db = Path(args.alert_db) if args.alert_db else Path(cfg.get("alert_db", ROOT / "border_alerts.db"))
        snaps = Path(args.snapshot_dir) if args.snapshot_dir else Path(cfg.get("snapshot_dir", ROOT / "alert_snapshots"))
        result = AlertLogger(db_path=db, snapshot_dir=snaps).verify_ledger()
        LOGGER.info("%s", result["message"])
        return 0 if result.get("ok") else 1
    sources = _parse_sources(args.source) if args.source else _sources_from_config(cfg)
    if not sources:
        LOGGER.error("No camera sources configured. Pass --source Cam=path or edit configs/perception.yaml")
        return 2

    topology = _topology_from_config(cfg)
    zones = Path(cfg.get("geofence_zones", ROOT / "configs" / "zones.json"))
    if not zones.is_absolute():
        zones = ROOT / zones

    dashboard_cfg = _load_dashboard_feature_config()
    face_cfg_path = ROOT / "configs" / "cameras.json"
    face_cfg = {}
    if face_cfg_path.exists():
        try:
            raw_face_cfg = json.loads(face_cfg_path.read_text(encoding="utf-8")).get("shared_face_recognition") or {}
            face_cfg = raw_face_cfg if isinstance(raw_face_cfg, dict) else {}
        except (OSError, json.JSONDecodeError):
            face_cfg = {}

    run_multi_camera_tracking(
        camera_sources=sources,
        similarity_threshold=float(args.threshold or cfg.get("similarity_threshold", 0.75)),
        max_time_diff=float(args.max_time_diff or cfg.get("max_time_diff", 120.0)),
        detector_weights=args.weights or cfg.get("detector_weights", "yolov8n.pt"),
        osnet_model=cfg.get("osnet_model", "osnet_x1_0"),
        person_class=int(cfg.get("person_class", PERSON_CLASS)),
        track_classes=cfg.get("track_classes", list(TRACK_CLASSES)),
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
        alert_db=Path(args.alert_db) if args.alert_db else Path(cfg.get("alert_db", ROOT / "border_alerts.db")),
        snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else Path(cfg.get("snapshot_dir", ROOT / "alert_snapshots")),
        dwell_threshold=float(args.dwell_threshold if args.dwell_threshold is not None else cfg.get("dwell_threshold", 30.0)),
        enable_loitering=bool(cfg.get("enable_loitering", True)) and not args.no_loitering,
        enable_anpr=bool(args.enable_anpr or cfg.get("enable_anpr", True)) and not args.no_anpr,
        anpr_min_confidence=float(
            args.anpr_min_confidence
            if args.anpr_min_confidence is not None
            else cfg.get("anpr_min_confidence", 0.5)
        ),
        anpr_min_width=int(cfg.get("anpr_min_width", 150)),
        anpr_min_height=int(cfg.get("anpr_min_height", 150)),
        anpr_gpu=cfg.get("anpr_gpu"),
        enable_face_capture=(
            bool(args.enable_face_capture or cfg.get("enable_face_capture", True))
            and not args.no_face_capture
        ),
        face_save_dir=Path(args.face_dir)
        if args.face_dir
        else Path(cfg.get("face_save_dir", ROOT / "face_database")),
        enable_enhance=bool(cfg.get("enable_enhance", True)) and not args.no_enhance,
        enhance_mode=str(args.enhance_mode or cfg.get("enhance_mode", "auto")),
        dashboard_features=dashboard_cfg,
        face_registry_dir=Path(face_cfg.get("registry_dir", ROOT / "face_registry")),
        face_registry_threshold=float(face_cfg.get("match_threshold", 0.45)),
        face_registry_model=str(face_cfg.get("model", "buffalo_l")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
