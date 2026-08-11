# SPDX-License-Identifier: MIT
"""2D polygon geometry helpers for balance cost computation.

Provides point-in-polygon tests and point-to-polygon distance
calculations used by the trajectory optimizer's balance cost function.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import numpy as np

from mujoco_models.exceptions import ValidationError


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting test for point-in-polygon on the 2D plane.

    Args:
        point: 2D point, shape (2,).
        polygon: Convex polygon vertices, shape (n, 2).

    Returns:
        True if the point is inside the polygon.

    Raises:
        ValidationError: If point is not shape (2,) or polygon has
            fewer than 3 vertices.
    """
    if point.shape != (2,):
        raise ValidationError(f"point must have shape (2,), got {point.shape}")
    if len(polygon) < 3:
        n_vert = len(polygon)
        raise ValidationError(f"polygon must have at least 3 vertices, got {n_vert}")

    inside = False
    px, py = float(point[0]), float(point[1])

    # OPTIMIZATION: Convert 2D NumPy array to nested list to avoid
    # expensive array indexing and C-API dispatch in tight Python loop.
    poly_list = polygon.tolist()

    xj, yj = poly_list[-1]

    for xi, yi in poly_list:
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
    px, py = float(point[0]), float(point[1])

    # OPTIMIZATION: Convert 2D NumPy array to nested list to avoid
    # expensive array indexing and C-API dispatch in tight Python loop.
    poly_list = polygon.tolist()

    xj, yj = poly_list[-1]

    for xi, yi in poly_list:
        abx = xi - xj
        aby = yi - yj

        apx = px - xj
        apy = py - yj

        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-12:
            dist_sq = apx * apx + apy * apy
        else:
            t = (apx * abx + apy * aby) / ab_sq
            if t < 0.0:
                dx = apx
                dy = apy
            elif t > 1.0:
                dx = px - xi
                dy = py - yi
            else:
                dx = apx - t * abx
                dy = apy - t * aby
            dist_sq = dx * dx + dy * dy

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
        xj, yj = xi, yi
    return min_dist_sq
