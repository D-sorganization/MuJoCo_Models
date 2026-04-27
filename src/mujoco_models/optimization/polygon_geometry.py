# SPDX-License-Identifier: MIT
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
    px, py = float(point[0]), float(point[1])

    xj, yj = float(polygon[-1, 0]), float(polygon[-1, 1])

    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])

        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside

        xj, yj = xi, yi

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
    px, py = float(point[0]), float(point[1])

    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        dist_sq = _point_to_segment_sq(
            px,
            py,
            float(polygon[i, 0]),
            float(polygon[i, 1]),
            float(polygon[j, 0]),
            float(polygon[j, 1]),
        )
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    return min_dist_sq


# OPTIMIZATION: Replaced numpy array operations with scalar arithmetic
# to avoid array creation overhead for 2D vectors.
def _point_to_segment_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Squared distance from a point to a line segment."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-12:
        return apx * apx + apy * apy

    t = (apx * abx + apy * aby) / ab_sq
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    dx = px - closest_x
    dy = py - closest_y

    return dx * dx + dy * dy
