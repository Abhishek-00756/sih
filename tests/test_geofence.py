"""Unit tests for the pixel-space geofencing MVP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofence.adapter import TrackingInterface, dict_to_tracked_object
from geofence.confirmation import TemporalConfirmer
from geofence.engine import EngineConfig, GeofenceEngine
from geofence.geometry import ground_contact_from_bbox, point_in_polygon
from geofence.mapping import HomographyMapper, IdentityMapper
from geofence.models import EventType, GeofenceZone, TrackedObject, TrackZoneState, ZoneType
from geofence.zones import ZoneStore


SQUARE = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]


def _zone(zone_id="z1", camera_id="cam_01", zone_type=ZoneType.RESTRICTED, poly=None):
    return GeofenceZone(
        zone_id=zone_id,
        camera_id=camera_id,
        name=zone_id,
        zone_type=zone_type,
        polygon=list(poly or SQUARE),
        enabled=True,
    )


def _track(oid=17, bbox=(180, 160, 220, 250), cam="cam_01", ts=1.0, cls="person", conf=0.9):
    return TrackedObject.from_raw(cam, oid, cls, bbox, conf, ts)


def test_ground_contact_is_bbox_bottom_center():
    assert ground_contact_from_bbox((10.0, 20.0, 50.0, 80.0)) == (30.0, 80.0)


def test_point_in_polygon_inside_outside_boundary():
    assert point_in_polygon((200.0, 200.0), SQUARE) is True
    assert point_in_polygon((10.0, 10.0), SQUARE) is False
    assert point_in_polygon((100.0, 200.0), SQUARE) is True


def test_temporal_confirmation_requires_consecutive_frames():
    conf = TemporalConfirmer(confirm_frames=3)
    state = TrackZoneState(camera_id="c", object_id="1", zone_id="z")
    assert conf.update(state, True).event_type is None
    assert conf.update(state, True).event_type is None
    result = conf.update(state, True)
    assert result.event_type == EventType.GEOFENCE_ENTER
    assert conf.update(state, True).event_type is None
    assert conf.update(state, False).event_type is None
    assert conf.update(state, False).event_type is None
    result = conf.update(state, False)
    assert result.event_type == EventType.GEOFENCE_EXIT
    assert conf.update(state, False).event_type is None


def test_confirmation_resets_when_flickering():
    conf = TemporalConfirmer(confirm_frames=3)
    state = TrackZoneState(camera_id="c", object_id="1", zone_id="z")
    conf.update(state, True)
    conf.update(state, True)
    conf.update(state, False)
    r = conf.update(state, True)
    assert r.event_type is None
    assert state.confirmed_inside is False


def test_engine_enter_exit_no_repeat_while_inside():
    store = ZoneStore([_zone()])
    engine = GeofenceEngine(store, EngineConfig(confirm_frames=2, stale_timeout_sec=10.0))
    inside = _track(bbox=(180, 160, 220, 250), ts=1.0)
    outside = _track(bbox=(500, 160, 540, 250), ts=2.0)

    assert engine.process_frame([inside], timestamp=1.0) == []
    events = engine.process_frame([inside], timestamp=1.1)
    assert len(events) == 1
    assert events[0].event_type == EventType.GEOFENCE_ENTER
    assert events[0].zone_id == "z1"
    assert events[0].ground_contact_position == (200.0, 250.0)

    assert engine.process_frame([inside], timestamp=1.2) == []
    assert engine.process_frame([inside], timestamp=1.3) == []

    outside = _track(bbox=(500, 160, 540, 250), ts=2.0)
    assert engine.process_frame([outside], timestamp=2.0) == []
    events = engine.process_frame([_track(bbox=(500, 160, 540, 250), ts=2.1)], timestamp=2.1)
    assert len(events) == 1
    assert events[0].event_type == EventType.GEOFENCE_EXIT


def test_disabled_zone_is_ignored():
    zone = _zone()
    zone.enabled = False
    engine = GeofenceEngine(ZoneStore([zone]), EngineConfig(confirm_frames=1))
    events = engine.process_frame([_track()], timestamp=1.0)
    assert events == []


def test_object_ids_are_camera_specific():
    z1 = _zone("z_cam1", "cam_01")
    z2 = _zone("z_cam2", "cam_02", poly=[(150, 350), (350, 350), (350, 550), (150, 550)])
    engine = GeofenceEngine(ZoneStore([z1, z2]), EngineConfig(confirm_frames=1, stale_timeout_sec=30))
    t1 = _track(oid=17, cam="cam_01", bbox=(180, 160, 220, 250), ts=1.0)
    t2 = TrackedObject.from_raw("cam_02", 17, "vehicle", (180, 400, 260, 500), 0.8, 1.0)
    events = engine.process_frame([t1, t2], timestamp=1.0)
    cams = {(e.camera_id, e.object_id, e.zone_id) for e in events}
    assert ("cam_01", "17", "z_cam1") in cams
    assert ("cam_02", "17", "z_cam2") in cams
    assert engine.confirmed_status("cam_01", "17", "z_cam1").value == "INSIDE"
    assert engine.confirmed_status("cam_02", "17", "z_cam1").value == "OUTSIDE"


def test_stale_state_is_pruned_without_immediate_delete_on_gap():
    store = ZoneStore([_zone()])
    engine = GeofenceEngine(store, EngineConfig(confirm_frames=1, stale_timeout_sec=5.0))
    engine.process_frame([_track(ts=10.0)], timestamp=10.0)
    assert engine.state.get("cam_01", "17", "z1") is not None
    engine.process_frame([], timestamp=12.0)
    assert engine.state.get("cam_01", "17", "z1") is not None
    engine.process_frame([], timestamp=16.0)
    assert engine.state.get("cam_01", "17", "z1") is None


def test_multiple_zones_per_camera():
    warning = _zone("warn", zone_type=ZoneType.WARNING, poly=[(50, 50), (400, 50), (400, 400), (50, 400)])
    restricted = _zone("rest", zone_type=ZoneType.RESTRICTED)
    engine = GeofenceEngine(ZoneStore([warning, restricted]), EngineConfig(confirm_frames=1))
    events = engine.process_frame([_track()], timestamp=1.0)
    zone_ids = {e.zone_id for e in events}
    assert zone_ids == {"warn", "rest"}


def test_adapter_accepts_bytetrack_field_aliases():
    rec = {
        "cam_id": "gate",
        "track_id": 9,
        "class_name": "person",
        "tlbr": [1, 2, 3, 40],
        "score": 0.66,
        "ts": 99.5,
    }
    obj = dict_to_tracked_object(rec)
    assert obj.camera_id == "gate"
    assert obj.object_id == "9"
    assert obj.object_type == "person"
    assert obj.bbox[3] == 40.0
    assert obj.confidence == 0.66

    iface = TrackingInterface()
    tracks = iface.from_bytetrack_tlbr(
        tlbrs=[[10, 20, 30, 80]],
        track_ids=[3],
        scores=[0.5],
        class_ids=[0],
        class_names={0: "person"},
        camera_id="cam_01",
        timestamp=1.0,
    )
    assert tracks[0].object_type == "person"
    assert tracks[0].object_id == "3"


def test_zone_store_roundtrip(tmp_path):
    store = ZoneStore([_zone(), _zone("z2", "cam_02")])
    path = tmp_path / "zones.json"
    store.save(path)
    loaded = ZoneStore.load(path)
    assert loaded.get("z1").zone_type == ZoneType.RESTRICTED
    assert loaded.get("z2").camera_id == "cam_02"


def test_identity_mapper_does_not_change_points():
    p = (12.0, 34.0)
    assert IdentityMapper().map_point(p, "cam_01") == p


def test_homography_mapper_is_optional_and_not_required():
    import numpy as np

    h = np.eye(3)
    mapper = HomographyMapper(h)
    assert mapper.map_point((10.0, 20.0), "cam_01") == pytest.approx((10.0, 20.0))


def test_event_payload_fields():
    store = ZoneStore([_zone()])
    engine = GeofenceEngine(store, EngineConfig(confirm_frames=1))
    ev = engine.process_frame([_track()], timestamp=1.0)[0]
    data = ev.to_dict()
    for key in (
        "event_type",
        "camera_id",
        "object_id",
        "object_type",
        "zone_id",
        "zone_type",
        "timestamp",
        "bounding_box",
        "ground_contact_position",
        "confidence",
    ):
        assert key in data
    assert data["event_type"] == "GEOFENCE_ENTER"


def test_sample_config_loads():
    path = ROOT / "configs" / "zones.json"
    store = ZoneStore.load(path)
    assert len(store.for_camera("cam_01")) == 3
    types = {z.zone_type for z in store.for_camera("cam_01")}
    assert types == {ZoneType.SAFE, ZoneType.WARNING, ZoneType.RESTRICTED}
