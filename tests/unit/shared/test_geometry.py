# SPDX-License-Identifier: MIT
"""Tests for geometry and inertia utilities."""

import math

import numpy as np
import pytest

from mujoco_models.shared.utils.geometry import (
    capsule_inertia,
    cylinder_inertia,
    parallel_axis_shift,
    rectangular_prism_inertia,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    sphere_inertia,
)


class TestCylinderInertia:
    def test_unit_cylinder(self) -> None:
        ixx, iyy, izz = cylinder_inertia(mass=1.0, radius=1.0, length=1.0)
        assert izz == pytest.approx(0.5)  # axial (Z-axis in MuJoCo)
        assert ixx == pytest.approx((1.0 / 12.0) * (3.0 + 1.0))
        assert iyy == pytest.approx(ixx)

    def test_transverse_equals(self) -> None:
        ixx, iyy, izz = cylinder_inertia(mass=5.0, radius=0.1, length=2.0)
        assert ixx == pytest.approx(iyy)

    def test_rejects_zero_mass(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            cylinder_inertia(0.0, 0.1, 1.0)

    def test_rejects_zero_radius(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            cylinder_inertia(1.0, 0.0, 1.0)

    def test_rejects_zero_length(self) -> None:
        with pytest.raises(ValueError, match="length"):
            cylinder_inertia(1.0, 0.1, 0.0)

    def test_barbell_shaft_like(self) -> None:
        ixx, iyy, izz = cylinder_inertia(mass=11.909, radius=0.014, length=1.31)
        assert izz > 0
        assert ixx > izz  # transverse > axial for long thin rod


class TestCapsuleInertia:
    def test_known_values(self) -> None:
        """Capsule with known mass, radius, length produces positive inertias."""
        ixx, iyy, izz = capsule_inertia(mass=2.0, radius=0.05, length=0.3)
        assert ixx > 0
        assert iyy > 0
        assert izz > 0

    def test_symmetry_transverse(self) -> None:
        """Transverse inertias (Ixx, Iyy) must be equal for axially symmetric shape."""
        ixx, iyy, izz = capsule_inertia(mass=5.0, radius=0.1, length=0.5)
        assert ixx == pytest.approx(iyy)

    def test_transverse_exceeds_axial_for_long_capsule(self) -> None:
        """For a long, thin capsule the transverse inertia exceeds the axial."""
        ixx, iyy, izz = capsule_inertia(mass=3.0, radius=0.02, length=1.0)
        assert ixx > izz

    def test_triangle_inequality(self) -> None:
        """Principal inertias must satisfy the triangle inequality."""
        ixx, iyy, izz = capsule_inertia(mass=1.0, radius=0.1, length=0.5)
        assert ixx + iyy >= izz
        assert ixx + izz >= iyy
        assert iyy + izz >= ixx

    def test_rejects_zero_mass(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            capsule_inertia(0.0, 0.1, 0.5)

    def test_rejects_zero_radius(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            capsule_inertia(1.0, 0.0, 0.5)

    def test_rejects_zero_length(self) -> None:
        with pytest.raises(ValueError, match="length"):
            capsule_inertia(1.0, 0.1, 0.0)

    def test_approaches_sphere_for_zero_length_limit(self) -> None:
        """As length shrinks, capsule inertia should approach sphere inertia."""
        ixx_c, _, izz_c = capsule_inertia(mass=1.0, radius=0.5, length=0.001)
        ixx_s, _, izz_s = sphere_inertia(mass=1.0, radius=0.5)
        assert ixx_c == pytest.approx(ixx_s, rel=0.05)
        assert izz_c == pytest.approx(izz_s, rel=0.05)


class TestRectangularPrismInertia:
    def test_cube(self) -> None:
        ixx, iyy, izz = rectangular_prism_inertia(1.0, 1.0, 1.0, 1.0)
        assert ixx == pytest.approx(iyy)
        assert iyy == pytest.approx(izz)

    def test_rejects_zero_dimension(self) -> None:
        with pytest.raises(ValueError, match="width"):
            rectangular_prism_inertia(1.0, 0.0, 1.0, 1.0)

    def test_asymmetric_box(self) -> None:
        ixx, iyy, izz = rectangular_prism_inertia(10.0, 0.3, 0.5, 0.1)
        assert ixx != iyy
        assert iyy != izz


class TestSphereInertia:
    def test_unit_sphere(self) -> None:
        ixx, iyy, izz = sphere_inertia(1.0, 1.0)
        assert ixx == pytest.approx(0.4)
        assert iyy == pytest.approx(0.4)
        assert izz == pytest.approx(0.4)

    def test_all_equal(self) -> None:
        ixx, iyy, izz = sphere_inertia(5.0, 0.3)
        assert ixx == pytest.approx(iyy)
        assert iyy == pytest.approx(izz)

    def test_rejects_zero_mass(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            sphere_inertia(0.0, 1.0)

    def test_rejects_zero_radius(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            sphere_inertia(1.0, 0.0)


class TestParallelAxisShift:
    def test_zero_displacement_unchanged(self) -> None:
        inertia = (1.0, 2.0, 3.0)
        result = parallel_axis_shift(10.0, inertia, np.array([0, 0, 0]))
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)

    def test_shift_along_x(self) -> None:
        inertia = (1.0, 1.0, 1.0)
        d = np.array([1.0, 0.0, 0.0])
        ixx, iyy, izz = parallel_axis_shift(1.0, inertia, d)
        assert ixx == pytest.approx(1.0)  # no change (shift along X)
        assert iyy == pytest.approx(2.0)
        assert izz == pytest.approx(2.0)

    def test_rejects_zero_mass(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            parallel_axis_shift(0.0, (1, 1, 1), np.array([1, 0, 0]))


class TestRotationMatrices:
    def test_rotation_x_identity(self) -> None:
        r = rotation_matrix_x(0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-12)

    def test_rotation_y_identity(self) -> None:
        r = rotation_matrix_y(0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-12)

    def test_rotation_z_identity(self) -> None:
        r = rotation_matrix_z(0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-12)

    def test_rotation_x_90(self) -> None:
        r = rotation_matrix_x(math.pi / 2)
        v = r @ np.array([0, 1, 0])
        np.testing.assert_allclose(v, [0, 0, 1], atol=1e-12)

    def test_rotation_y_90(self) -> None:
        r = rotation_matrix_y(math.pi / 2)
        v = r @ np.array([0, 0, 1])
        np.testing.assert_allclose(v, [1, 0, 0], atol=1e-12)

    def test_rotation_z_90(self) -> None:
        r = rotation_matrix_z(math.pi / 2)
        v = r @ np.array([1, 0, 0])
        np.testing.assert_allclose(v, [0, 1, 0], atol=1e-12)

    def test_rotation_is_orthogonal(self) -> None:
        r = rotation_matrix_x(0.7)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)

    def test_rotation_det_is_one(self) -> None:
        r = rotation_matrix_z(1.23)
        assert np.linalg.det(r) == pytest.approx(1.0)
