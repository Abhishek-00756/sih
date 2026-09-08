"""Pixel-space geometry helpers.

Ground-contact approximation
----------------------------
The object's ground-contact point is estimated as the bottom-center of its
axis-aligned bounding box:

    ( (x1 + x2) / 2,  y2 )

This is a common CCTV approximation: people and vehicles usually touch the
ground near the bottom of the box. It is NOT a true 3D foot/wheel position.
Occlusion, box jitter, kneeling people, and aerial cameras will all bias it.

Pixel coordinates
-----------------
Points and polygons live in image/pixel space of a single camera frame.
They do not represent real-world distance or geographic coordinates.
"""

from __future__ import annotations

from typing import List

from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

from geofence.models import BoundingBox, Point2D


def ground_contact_from_bbox(bbox: BoundingBox) -> Point2D:
    """Return the bottom-center of an axis-aligned bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def bbox_center(bbox: BoundingBox) -> Point2D:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _as_polygon(vertices: List[Point2D]) -> Polygon:
    if len(vertices) < 3:
        raise ValueError("polygon requires at least 3 vertices")
    poly = Polygon(vertices)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        elif poly.geom_type != "Polygon":
            poly = Polygon(vertices).buffer(0)
    if poly.is_empty:
        raise ValueError("polygon is empty after validation")
    return poly


def point_in_polygon(point: Point2D, vertices: List[Point2D]) -> bool:
    """True if `point` lies inside or on the boundary of the polygon."""
    poly = _as_polygon(vertices)
    shapely_point = Point(point[0], point[1])
    return bool(poly.covers(shapely_point))


def polygon_centroid(vertices: List[Point2D]) -> Point2D:
    poly = _as_polygon(vertices)
    c = poly.centroid
    return (float(c.x), float(c.y))


def clamp_point_to_frame(point: Point2D, width: int, height: int) -> Point2D:
    x = min(max(point[0], 0.0), float(max(width - 1, 0)))
    y = min(max(point[1], 0.0), float(max(height - 1, 0)))
    return (x, y)
