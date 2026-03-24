"""Hypothesis property-based tests for geometry and body model (#1).

Uses hypothesis strategies to verify invariants hold across a wide range
of input parameters, catching edge cases that example-based tests miss.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mujoco_models.shared.body import BodyModelSpec, create_full_body
from mujoco_models.shared.body.segment_data import SEGMENT_TABLE, segment_properties
from mujoco_models.shared.utils.geometry import (
    cylinder_inertia,
    parallel_axis_shift,
    rectangular_prism_inertia,
    sphere_inertia,
)

# Strategies for physically plausible ranges
positive_float = st.floats(min_value=0.01, max_value=1e4, allow_nan=False)
body_mass = st.floats(min_value=20.0, max_value=300.0, allow_nan=False)
body_height = st.floats(min_value=1.0, max_value=2.5, allow_nan=False)


class TestCylinderInertiaProperties:
    @given(mass=positive_float, radius=positive_float, length=positive_float)
    @settings(max_examples=50)
    def test_all_inertias_positive(
        self, mass: float, radius: float, length: float
    ) -> None:
        ixx, iyy, izz = cylinder_inertia(mass, radius, length)
        assert ixx > 0
        assert iyy > 0
        assert izz > 0

    @given(mass=positive_float, radius=positive_float, length=positive_float)
    @settings(max_examples=50)
    def test_transverse_inertias_equal(
        self, mass: float, radius: float, length: float
    ) -> None:
        ixx, iyy, _izz = cylinder_inertia(mass, radius, length)
        assert ixx == pytest.approx(iyy)

    @given(mass=positive_float, radius=positive_float, length=positive_float)
    @settings(max_examples=50)
    def test_triangle_inequality(
        self, mass: float, radius: float, length: float
    ) -> None:
        ixx, iyy, izz = cylinder_inertia(mass, radius, length)
        assert ixx + iyy >= izz
        assert ixx + izz >= iyy
        assert iyy + izz >= ixx


class TestRectangularPrismInertiaProperties:
    @given(
        mass=positive_float,
        w=positive_float,
        h=positive_float,
        d=positive_float,
    )
    @settings(max_examples=50)
    def test_all_positive(self, mass: float, w: float, h: float, d: float) -> None:
        ixx, iyy, izz = rectangular_prism_inertia(mass, w, h, d)
        assert ixx > 0
        assert iyy > 0
        assert izz > 0

    @given(mass=positive_float, side=positive_float)
    @settings(max_examples=30)
    def test_cube_symmetry(self, mass: float, side: float) -> None:
        ixx, iyy, izz = rectangular_prism_inertia(mass, side, side, side)
        assert ixx == pytest.approx(iyy)
        assert iyy == pytest.approx(izz)


class TestSphereInertiaProperties:
    @given(mass=positive_float, radius=positive_float)
    @settings(max_examples=50)
    def test_all_equal(self, mass: float, radius: float) -> None:
        ixx, iyy, izz = sphere_inertia(mass, radius)
        assert ixx == pytest.approx(iyy)
        assert iyy == pytest.approx(izz)


class TestParallelAxisProperties:
    @given(mass=positive_float)
    @settings(max_examples=30)
    def test_zero_shift_unchanged(self, mass: float) -> None:
        inertia = (1.0, 2.0, 3.0)
        result = parallel_axis_shift(mass, inertia, np.zeros(3))
        assert result[0] == pytest.approx(inertia[0])
        assert result[1] == pytest.approx(inertia[1])
        assert result[2] == pytest.approx(inertia[2])

    @given(
        mass=positive_float,
        dx=st.floats(min_value=-10, max_value=10, allow_nan=False),
        dy=st.floats(min_value=-10, max_value=10, allow_nan=False),
        dz=st.floats(min_value=-10, max_value=10, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_shifted_inertia_not_smaller(
        self, mass: float, dx: float, dy: float, dz: float
    ) -> None:
        inertia = (0.1, 0.2, 0.3)
        result = parallel_axis_shift(mass, inertia, np.array([dx, dy, dz]))
        assert result[0] >= inertia[0] - 1e-10
        assert result[1] >= inertia[1] - 1e-10
        assert result[2] >= inertia[2] - 1e-10


class TestSegmentPropertiesHypothesis:
    @given(mass=body_mass, height=body_height)
    @settings(max_examples=30)
    def test_all_segments_positive(self, mass: float, height: float) -> None:
        for name in SEGMENT_TABLE:
            seg_mass, seg_len, seg_rad = segment_properties(mass, height, name)
            assert seg_mass > 0
            assert seg_len > 0
            assert seg_rad > 0


class TestFullBodyModelProperties:
    @given(mass=body_mass, height=body_height)
    @settings(max_examples=10)
    def test_always_produces_15_bodies(self, mass: float, height: float) -> None:
        worldbody = ET.Element("worldbody")
        spec = BodyModelSpec(total_mass=mass, height=height)
        bodies = create_full_body(worldbody, spec)
        assert len(bodies) == 15

    @given(mass=body_mass, height=body_height)
    @settings(max_examples=10)
    def test_all_masses_positive(self, mass: float, height: float) -> None:
        worldbody = ET.Element("worldbody")
        spec = BodyModelSpec(total_mass=mass, height=height)
        bodies = create_full_body(worldbody, spec)
        for name, body in bodies.items():
            inertial = body.find("inertial")
            assert inertial is not None, f"{name} missing inertial"
            m = float(inertial.get("mass"))  # type: ignore
            assert m > 0, f"{name} mass={m}"
