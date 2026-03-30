"""2D polygon geometry helpers for balance cost computation.

Provides point-in-polygon tests and point-to-polygon distance
calculations used by the trajectory optimizer's balance cost function.
"""

from __future__ import annotations

import numpy as np


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting test for point-in-polygon on the 2D plane.

    Args:
        point: 2D point, shape (2,).
        polygon: Convex polygon vertices, shape (n, 2).

    Returns:
        True if the point is inside the polygon.

    Raises:
        ValueError: If point is not shape (2,) or polygon has fewer than 3 vertices.
    """
    if point.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {point.shape}")
    if len(polygon) < 3:
        raise ValueError(f"polygon must have at least 3 vertices, got {len(polygon)}")
    n = len(polygon)
    inside = False
    px, py = point[0], point[1]
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        j = i
    return inside


def squared_distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
    """Minimum squared distance from a point to a polygon boundary.

    Args:
        point: 2D point, shape (2,).
        polygon: Polygon vertices, shape (n, 2).

    Returns:
        Minimum squared distance to any polygon edge.
    """
    min_dist_sq = float("inf")
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        dist_sq = _point_to_segment_sq(point, polygon[i], polygon[j])
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    return min_dist_sq


def _point_to_segment_sq(
    point: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> float:
    """Squared distance from a point to a line segment."""
    ab = seg_b - seg_a
    ap = point - seg_a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return float(np.dot(ap, ap))
    t = float(np.dot(ap, ab)) / ab_sq
    t = max(0.0, min(1.0, t))
    closest = seg_a + t * ab
    diff = point - closest
    return float(np.dot(diff, diff))
