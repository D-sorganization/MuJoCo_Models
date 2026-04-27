# SPDX-License-Identifier: MIT
"""Tests for 2D polygon geometry helpers.

Covers ``point_in_polygon`` and ``squared_distance_to_polygon`` with
happy-path geometries, boundary cases, and DbC failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_models.optimization.polygon_geometry import (
    point_in_polygon,
    squared_distance_to_polygon,
)


def _unit_square() -> np.ndarray:
    """Axis-aligned unit square with vertices in CCW order."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )


def _triangle() -> np.ndarray:
    """Right triangle with legs of length 2."""
    return np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )


class TestPointInPolygon:
    """Ray-casting point-in-polygon behavior."""

    def test_center_of_square_is_inside(self) -> None:
        assert point_in_polygon(np.array([0.5, 0.5]), _unit_square()) is True

    def test_point_far_outside_square(self) -> None:
        assert point_in_polygon(np.array([5.0, 5.0]), _unit_square()) is False

    def test_negative_coordinate_outside(self) -> None:
        assert point_in_polygon(np.array([-0.1, 0.5]), _unit_square()) is False

    def test_triangle_interior(self) -> None:
        # Centroid of the (0,0),(2,0),(0,2) triangle is (2/3, 2/3).
        assert point_in_polygon(np.array([2.0 / 3.0, 2.0 / 3.0]), _triangle()) is True

    def test_triangle_outside_hypotenuse(self) -> None:
        # (1.5, 1.5) lies on the far side of the hypotenuse x + y = 2.
        assert point_in_polygon(np.array([1.5, 1.5]), _triangle()) is False


class TestPointInPolygonDbC:
    """Contract violations raise ValueError."""

    def test_wrong_point_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"point must have shape"):
            point_in_polygon(np.array([0.0, 0.0, 0.0]), _unit_square())

    def test_polygon_with_two_vertices_raises(self) -> None:
        bad = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match=r"at least 3 vertices"):
            point_in_polygon(np.array([0.5, 0.5]), bad)

    def test_empty_polygon_raises(self) -> None:
        with pytest.raises(ValueError, match=r"at least 3 vertices"):
            point_in_polygon(np.array([0.0, 0.0]), np.empty((0, 2)))


class TestSquaredDistanceToPolygon:
    """Nearest-edge squared distance computations."""

    def test_distance_to_edge_outside(self) -> None:
        # Point (2, 0.5) is 1.0 away from the right edge of the unit square.
        dist_sq = squared_distance_to_polygon(np.array([2.0, 0.5]), _unit_square())
        assert dist_sq == pytest.approx(1.0)

    def test_distance_to_corner(self) -> None:
        # Point (-1, -1) is sqrt(2) away from corner (0, 0): squared = 2.
        dist_sq = squared_distance_to_polygon(np.array([-1.0, -1.0]), _unit_square())
        assert dist_sq == pytest.approx(2.0)

    def test_zero_distance_on_edge(self) -> None:
        # Midpoint of the bottom edge lies exactly on the boundary.
        dist_sq = squared_distance_to_polygon(np.array([0.5, 0.0]), _unit_square())
        assert dist_sq == pytest.approx(0.0, abs=1e-12)

    def test_interior_point_gives_small_distance(self) -> None:
        # Center of unit square is 0.5 from each edge: squared = 0.25.
        dist_sq = squared_distance_to_polygon(np.array([0.5, 0.5]), _unit_square())
        assert dist_sq == pytest.approx(0.25)

    def test_distance_is_nonnegative(self) -> None:
        for px, py in [(0.1, 0.1), (-2.0, 3.0), (0.5, 0.5), (1.0, 1.0)]:
            dist_sq = squared_distance_to_polygon(np.array([px, py]), _unit_square())
            assert dist_sq >= 0.0

    def test_symmetric_across_square(self) -> None:
        """By symmetry, points reflected over center have equal distance."""
        sq = _unit_square()
        d1 = squared_distance_to_polygon(np.array([2.0, 0.5]), sq)
        d2 = squared_distance_to_polygon(np.array([-1.0, 0.5]), sq)
        assert d1 == pytest.approx(d2)
