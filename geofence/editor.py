"""Interactive polygon zone editor.

Click points on a camera frame to define a zone. Press:
  left-click  add vertex
  right-click undo last vertex
  ENTER       finish polygon (needs >= 3 points)
  n           finish and immediately start next zone
  s           save current store to JSON
  q / ESC     quit without adding the in-progress polygon
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from geofence.models import GeofenceZone, Point2D, ZoneType
from geofence.visualizer import ZONE_COLORS, GeofenceVisualizer
from geofence.zones import ZoneStore

WINDOW = "geofence-zone-editor"


class ZoneEditor:
    def __init__(self, camera_id: str, zone_store: Optional[ZoneStore] = None):
        self.camera_id = camera_id
        self.store = zone_store or ZoneStore()
        self._pending: List[Point2D] = []
        self._frame: Optional[np.ndarray] = None
        self._done = False
        self._save_path: Optional[Path] = None
        self._visualizer = GeofenceVisualizer()
        self._next_index = len(self.store.all_zones()) + 1
        self._zone_type = ZoneType.RESTRICTED
        self._last_message = "L-click add  R-click undo  ENTER finish  s save  q quit"

    def _mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._pending.append((float(x), float(y)))
            self._last_message = f"vertices={len(self._pending)}"
        elif event == cv2.EVENT_RBUTTONDOWN and self._pending:
            self._pending.pop()
            self._last_message = f"undo, vertices={len(self._pending)}"

    def _compose(self) -> np.ndarray:
        assert self._frame is not None
        canvas = self._visualizer.draw(self._frame, self.store.for_camera(self.camera_id, enabled_only=False), [])
        if len(self._pending) >= 1:
            pts = np.array(self._pending, dtype=np.int32)
            color = ZONE_COLORS[self._zone_type]
            for p in pts:
                cv2.circle(canvas, (int(p[0]), int(p[1])), 4, color, -1)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        hint = (
            f"cam={self.camera_id} type={self._zone_type.value}  "
            f"[1]safe [2]warning [3]restricted   {self._last_message}"
        )
        cv2.putText(canvas, hint, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _commit_pending(self) -> Optional[GeofenceZone]:
        if len(self._pending) < 3:
            self._last_message = "need at least 3 points"
            return None
        zone_id = f"zone_{self.camera_id}_{self._next_index:02d}"
        name = f"{self._zone_type.value}_{self._next_index:02d}"
        zone = GeofenceZone(
            zone_id=zone_id,
            camera_id=self.camera_id,
            name=name,
            zone_type=self._zone_type,
            polygon=list(self._pending),
            enabled=True,
        )
        self.store.add(zone)
        self._next_index += 1
        self._pending = []
        self._last_message = f"saved {zone.zone_id}"
        return zone

    def run(
        self,
        frame: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        window_name: str = WINDOW,
    ) -> ZoneStore:
        """Blocking UI. Returns the updated ZoneStore."""
        self._frame = frame.copy()
        self._save_path = Path(save_path) if save_path else None
        self._done = False
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse)
        while not self._done:
            canvas = self._compose()
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                self._done = True
            elif key in (13, 10):
                self._commit_pending()
            elif key == ord("n"):
                self._commit_pending()
            elif key == ord("s"):
                self._commit_pending()
                if self._save_path is not None:
                    self.store.save(self._save_path)
                    self._last_message = f"wrote {self._save_path}"
            elif key == ord("1"):
                self._zone_type = ZoneType.SAFE
            elif key == ord("2"):
                self._zone_type = ZoneType.WARNING
            elif key == ord("3"):
                self._zone_type = ZoneType.RESTRICTED
        cv2.destroyWindow(window_name)
        if self._save_path is not None:
            self.store.save(self._save_path)
        return self.store


def edit_zones_on_image(
    image_path: Union[str, Path],
    camera_id: str,
    output_json: Union[str, Path],
    existing: Optional[Union[str, Path]] = None,
) -> ZoneStore:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    store = ZoneStore.load(existing) if existing else ZoneStore()
    editor = ZoneEditor(camera_id=camera_id, zone_store=store)
    return editor.run(frame, save_path=output_json)
