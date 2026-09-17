"""Live AI runtime shared by the BORDER SENTINEL dashboard."""
from __future__ import annotations
import json, logging, threading, time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
import yaml
from activity_analyzer import COCO_CLASSES, PERSON_CLASS, TRACK_CLASSES
from alert_logger import AlertLogger
from anpr_manager import ANPRManager
from enhancer import VideoEnhancer
from extractor import PersonFeatureExtractor
from gallery_manager import GlobalGalleryManager
from geofence_manager import GeofenceManager, VirtualTripwire

LOGGER = logging.getLogger("border.runtime")
ROOT = Path(__file__).resolve().parent

class PerceptionController:
    """Consumes the newest frame from each dashboard camera and runs selected AI."""
    def __init__(self, cameras: Dict[str, Any], face_registry: Any, ledger: Any) -> None:
        self.cameras = cameras
        self.face_registry = face_registry
        self.ledger = ledger
        self.config_path = ROOT / "configs" / "cameras.json"
        self.perception_path = ROOT / "configs" / "perception.yaml"
        self.zones_path = ROOT / "configs" / "zones.json"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.detectors: Dict[str, Any] = {}
        self.gallery = GlobalGalleryManager(similarity_threshold=0.75, max_time_diff=120.0)
        self.extractor: Optional[PersonFeatureExtractor] = None
        self.enhancer: Optional[VideoEnhancer] = None
        self.anpr: Dict[str, ANPRManager] = {}
        self.geofence_managers: Dict[str, GeofenceManager] = {}
        self.last_sequences: Dict[str, int] = {}
        self.last_alerts: Dict[Tuple[str, int, str], float] = {}
        self.last_error: Optional[str] = None
        self.started_at = time.time()
        self.metrics = {cid: {"processed_frames": 0, "persons": 0, "vehicles": 0,
                              "global_ids": [], "face_matches": 0, "plates": [],
                              "last_processed_at": 0.0, "fps": 0.0, "osnet": "NOT LOADED",
                              "last_event": None} for cid in cameras}
        self._zones_mtime = -1.0
        self._zones: Dict[str, list] = {}
        self._tripwires_mtime = -1.0
        self._tripwires: Dict[str, list] = {}
        self.thread = threading.Thread(target=self._run, name="perception-controller", daemon=True)
        self.thread.start()

    def _camera_specs(self) -> dict:
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8")).get("cameras", {})
        except Exception as exc:
            self.last_error = f"camera config: {exc}"
            return {}

    def _perception_cfg(self) -> dict:
        try:
            return yaml.safe_load(self.perception_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            self.last_error = f"perception config: {exc}"
            return {}

    def _camera_enabled(self, camera_id: str) -> bool:
        spec = self._camera_specs().get(camera_id, {})
        return bool(spec.get("enabled", True)) and bool(spec.get("ai_enabled", False))

    def _feature_enabled(self, camera_id: str, key: str, default: bool = False) -> bool:
        spec = self._camera_specs().get(camera_id, {})
        return bool((spec.get("features") or {}).get(key, default))

    def _load_zones(self) -> None:
        try:
            mtime = self.zones_path.stat().st_mtime
        except FileNotFoundError:
            self._zones = {}
            return
        if mtime == self._zones_mtime:
            return
        try:
            raw = json.loads(self.zones_path.read_text(encoding="utf-8"))
            grouped: Dict[str, list] = {}
            for zone in raw.get("zones", []):
                if zone.get("enabled", True):
                    grouped.setdefault(str(zone.get("camera_id", "")), []).append(zone)
            self._zones = grouped
            self._zones_mtime = mtime
            self.geofence_managers.clear()
        except Exception as exc:
            LOGGER.warning("Failed to load zones: %s", exc)

    def _load_tripwires(self) -> None:
        try:
            mtime = self.perception_path.stat().st_mtime
        except FileNotFoundError:
            self._tripwires = {}
            return
        if mtime == self._tripwires_mtime:
            return
        cfg = self._perception_cfg()
        grouped: Dict[str, list] = {}
        for item in cfg.get("tripwires") or []:
            if isinstance(item, dict) and item.get("camera_id") and item.get("pt1") and item.get("pt2"):
                grouped.setdefault(str(item["camera_id"]), []).append(item)
        self._tripwires = grouped
        self._tripwires_mtime = mtime
        self.geofence_managers.clear()

    def _ensure_detector(self, camera_id: str) -> Any:
        detector = self.detectors.get(camera_id)
        if detector is None:
            from ultralytics import YOLO
            weights = str(self._perception_cfg().get("detector_weights", "yolov8n.pt"))
            detector = YOLO(weights)
            self.detectors[camera_id] = detector
            LOGGER.info("Loaded YOLO detector for %s", camera_id)
        return detector

    def _ensure_extractor(self) -> PersonFeatureExtractor:
        if self.extractor is None:
            cfg = self._perception_cfg()
            self.extractor = PersonFeatureExtractor(model_name=str(cfg.get("osnet_model", "osnet_x1_0")), use_fallback=False)
        return self.extractor

    def _ensure_anpr(self, camera_id: str) -> ANPRManager:
        if camera_id not in self.anpr:
            cfg = self._perception_cfg()
            self.anpr[camera_id] = ANPRManager(min_confidence=float(cfg.get("anpr_min_confidence", 0.5)), gpu=cfg.get("anpr_gpu"))
        return self.anpr[camera_id]

    def _ensure_enhancer(self) -> VideoEnhancer:
        if self.enhancer is None:
            self.enhancer = VideoEnhancer()
        return self.enhancer

    def _ensure_tripwire_manager(self, camera_id: str) -> Optional[GeofenceManager]:
        if camera_id in self.geofence_managers:
            return self.geofence_managers[camera_id]
        self._load_tripwires()
        cfg = self._perception_cfg()
        camera_key = str(cfg.get("geofence_camera_map", {}).get(camera_id, camera_id))
        items = self._tripwires.get(camera_key) or self._tripwires.get(camera_id) or []
        wires = [VirtualTripwire(item["pt1"], item["pt2"], wire_id=str(item.get("id", "border")),
                                 camera_id=camera_key, cooldown_seconds=float(item.get("cooldown_seconds", cfg.get("geofence_cooldown", 60.0))),
                                 inbound_positive=bool(item.get("inbound_positive", True))) for item in items]
        if not wires:
            return None
        mgr = GeofenceManager(zone_points=[(0,0),(1,0),(1,1),(0,1)], cooldown_seconds=float(cfg.get("geofence_cooldown",60.0)), tripwires=wires, logger=None)
        self.geofence_managers[camera_id] = mgr
        return mgr

    def _match_face(self, crop: np.ndarray) -> Optional[dict]:
        if self.face_registry is None:
            return None
        try:
            status = self.face_registry.status() or {}
            if status.get("available") is False:
                return None
        except Exception:
            pass
        for name in ("match", "recognize", "recognize_face", "match_face"):
            fn = getattr(self.face_registry, name, None)
            if not callable(fn):
                continue
            try:
                result = fn(crop)
            except Exception as exc:
                LOGGER.debug("Face match failed: %s", exc)
                return None
            if isinstance(result, dict):
                who = result.get("display_name") or result.get("name") or result.get("person_name")
                if not who:
                    return None
                try:
                    conf = float(result.get("confidence", result.get("similarity", 0.0)))
                except (TypeError, ValueError):
                    conf = 0.0
                return {"display_name": str(who), "confidence": conf}
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                try:
                    return {"display_name": str(result[0]), "confidence": float(result[1])}
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _inside(point: Tuple[int, int], polygon: Any) -> bool:
        pts = np.asarray(polygon, dtype=np.int32)
        return len(pts) >= 3 and cv2.pointPolygonTest(pts, point, False) >= 0

    def _draw_zones(self, frame: np.ndarray, camera_id: str) -> None:
        self._load_zones()
        cfg = self._perception_cfg()
        key = str(cfg.get("geofence_camera_map", {}).get(camera_id, camera_id))
        for zone in self._zones.get(key) or self._zones.get(camera_id) or []:
            polygon = np.asarray(zone.get("polygon", []), dtype=np.int32)
            if len(polygon) < 3:
                continue
            restricted = str(zone.get("zone_type", "")).lower() == "restricted"
            cv2.polylines(frame, [polygon], True, (60,70,220) if restricted else (50,180,120), 3 if restricted else 2)
            x, y = [int(v) for v in polygon[0]]
            cv2.putText(frame, str(zone.get("name", zone.get("zone_id", "zone"))), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1, cv2.LINE_AA)

    def _emit_intrusion(self, camera_id: str, gid: int, frame: np.ndarray, bbox: Tuple[int,int,int,int], now: float, zone: dict) -> None:
        key = (camera_id, gid, str(zone.get("zone_id", "zone")))
        if now - self.last_alerts.get(key, 0.0) < 60.0:
            return
        self.last_alerts[key] = now
        x1,y1,x2,y2 = bbox
        details = {"event":"GEOFENCE_ENTER", "camera_id":camera_id, "global_id":gid,
                   "zone_id":zone.get("zone_id"), "zone_name":zone.get("name"), "zone_type":zone.get("zone_type"),
                   "timestamp":now, "bbox":[x1,y1,x2,y2], "footprint":[(x1+x2)//2,y2]}
        self.metrics[camera_id]["last_event"] = details
        try:
            cfg = self._perception_cfg()
            logger = AlertLogger(db_path=ROOT / str(cfg.get("alert_db","border_alerts.db")), snapshot_dir=ROOT / str(cfg.get("snapshot_dir","alert_snapshots")))
            logger.log_intrusion(camera_id=camera_id, global_id=gid, frame=frame, bbox=bbox, timestamp=now,
                                 alert_type="GEOFENCE_ENTER", zone_id=str(zone.get("zone_id","")), zone_name=str(zone.get("name","")),
                                 footprint=((x1+x2)//2,y2))
        except Exception as exc:
            LOGGER.exception("Evidence log failed: %s", exc)
        try:
            self.ledger.anchor(details)
        except Exception as exc:
            LOGGER.debug("Ledger anchor failed: %s", exc)
        LOGGER.warning("INTRUSION %s", json.dumps(details))

    def _check_zones(self, frame: np.ndarray, camera_id: str, gid: int, bbox: Tuple[int,int,int,int], now: float) -> None:
        if not self._feature_enabled(camera_id, "geofence", False):
            return
        self._load_zones()
        cfg = self._perception_cfg()
        key = str(cfg.get("geofence_camera_map", {}).get(camera_id, camera_id))
        point = ((bbox[0]+bbox[2])//2, bbox[3])
        for zone in self._zones.get(key) or self._zones.get(camera_id) or []:
            if str(zone.get("zone_type","")).lower() == "restricted" and self._inside(point, zone.get("polygon",[])):
                self._emit_intrusion(camera_id, gid, frame, bbox, now, zone)

    def _process_camera(self, camera_id: str, view: Any) -> None:
        frame, sequence = view.frame()
        if frame is None or sequence == self.last_sequences.get(camera_id, -1):
            return
        self.last_sequences[camera_id] = sequence
        cfg = self._camera_specs().get(camera_id, {})
        features = cfg.get("features") or {}
        work = frame.copy()
        if features.get("enhancement", False):
            work = self._ensure_enhancer().enhance_frame(work, mode="auto")
        if features.get("geofence", False):
            self._draw_zones(work, camera_id)

        detector = self._ensure_detector(camera_id)
        results = detector.track(work, persist=True, classes=list(TRACK_CLASSES), tracker="bytetrack.yaml", verbose=False)
        persons = vehicles = face_matches = 0
        gids, plates = set(), []
        tripwire = self._ensure_tripwire_manager(camera_id) if features.get("tripwire", False) else None
        extractor = self._ensure_extractor()
        now = time.time()
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            tids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes.cls is not None else np.full(len(tids), PERSON_CLASS)
            for box, local_id, score, cls_id in zip(boxes, tids, confs, clss):
                x1,y1,x2,y2 = [int(v) for v in box]
                h,w = work.shape[:2]
                x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
                if x2<=x1 or y2<=y1: continue
                bbox=(x1,y1,x2,y2)
                if int(cls_id)==PERSON_CLASS:
                    crop=work[y1:y2,x1:x2]
                    emb=extractor.extract(crop)
                    if emb is None: continue
                    match=self.gallery.match_or_register_detailed(camera_id=camera_id, local_id=int(local_id), embedding=emb, timestamp=now)
                    gid=int(match.global_id); gids.add(gid); persons += 1
                    label=f"PERSON GID {gid}"
                    if features.get("face_recognition", False):
                        face=self._match_face(crop)
                        if face:
                            face_matches += 1
                            label += f" | {face['display_name']} {face['confidence']:.0%}"
                    self._check_zones(frame, camera_id, gid, bbox, now)
                    if tripwire is not None:
                        try: tripwire.check_tripwire(work, bbox, gid, camera_id=camera_id, timestamp=now, source_camera_id=camera_id)
                        except Exception as exc: LOGGER.debug("Tripwire failed: %s", exc)
                    color=(60,220,130)
                else:
                    vehicles += 1; label=f"{COCO_CLASSES.get(int(cls_id),'Vehicle')} V-{int(local_id)}"; color=(40,180,240)
                    if features.get("anpr", False):
                        anpr=self._ensure_anpr(camera_id)
                        if anpr.crop_is_readable(x2-x1,y2-y1,min_width=150,min_height=150):
                            text,_=anpr.read_license_plate(work[y1:y2,x1:x2])
                            if text: label += f" | PLATE {text}"; plates.append(text)
                cv2.rectangle(work,(x1,y1),(x2,y2),color,2)
                cv2.putText(work,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.52,color,2,cv2.LINE_AA)

        metric=self.metrics[camera_id]
        metric["processed_frames"] += 1; metric["persons"]=persons; metric["vehicles"]=vehicles
        metric["global_ids"]=sorted(gids); metric["face_matches"]=face_matches; metric["plates"]=plates[-10:]
        prev=metric.get("last_processed_at",0.0); metric["last_processed_at"]=now
        if prev: metric["fps"]=0.8*float(metric.get("fps",0.0))+0.2*(1.0/max(now-prev,1e-6))
        metric["osnet"]="FALLBACK" if extractor.use_fallback else "OSNET"
        view.set_processed_frame(work)

    def _run(self) -> None:
        LOGGER.info("Dashboard perception controller started")
        while not self.stop_event.is_set():
            for camera_id, view in self.cameras.items():
                if self._camera_enabled(camera_id):
                    try: self._process_camera(camera_id, view)
                    except Exception as exc:
                        self.last_error=f"{camera_id}: {exc}"; LOGGER.exception("AI processing failed for %s", camera_id)
                else:
                    view.set_processed_frame(None)
            time.sleep(0.005)

    def state(self, camera_id: str) -> dict:
        with self.lock: metric=dict(self.metrics.get(camera_id, {}))
        metric["ai_enabled"]=self._camera_enabled(camera_id)
        metric["runtime"]="RUNNING" if metric["ai_enabled"] else "STANDBY"
        return metric

    def global_state(self) -> dict:
        return {"running":self.thread.is_alive(), "uptime":max(0.0,time.time()-self.started_at),
                "last_error":self.last_error, "osnet":None if self.extractor is None else (not self.extractor.use_fallback),
                "gallery":self.gallery.active_count()}

    def stop(self) -> None:
        self.stop_event.set(); self.thread.join(timeout=2.0)
