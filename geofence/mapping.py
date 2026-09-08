"""Optional coordinate-mapping layer.

MVP uses image/pixel coordinates only. Homography (or any camera-to-world
transform) can later wrap this interface without changing the geofence
engine. Homography is NOT a dependency of the MVP.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from geofence.models import Point2D


class CoordinateMapper:
    """Maps a ground-contact point before the polygon test.

    The default implementation is identity: pixel in, pixel out.
    """

    def map_point(self, point: Point2D, camera_id: str) -> Point2D:
        return point

    def is_identity(self) -> bool:
        return True


class IdentityMapper(CoordinateMapper):
    pass


class HomographyMapper(CoordinateMapper):
    """Optional future layer: pixel -> planar world via a 3x3 homography.

    Not used by the MVP. Zones would then be defined in the same world plane.
    Homography still does not produce GPS unless the plane is georeferenced.
    """

    def __init__(self, matrix: np.ndarray, camera_id: Optional[str] = None):
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError("homography matrix must be 3x3")
        self.matrix = mat
        self.camera_id = camera_id

    def map_point(self, point: Point2D, camera_id: str) -> Point2D:
        if self.camera_id is not None and camera_id != self.camera_id:
            return point
        vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
        mapped = self.matrix @ vec
        if abs(mapped[2]) < 1e-12:
            return point
        return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))

    def is_identity(self) -> bool:
        return False


def build_mapper(homography: Optional[np.ndarray] = None, camera_id: Optional[str] = None) -> CoordinateMapper:
    if homography is None:
        return IdentityMapper()
    return HomographyMapper(homography, camera_id=camera_id)
