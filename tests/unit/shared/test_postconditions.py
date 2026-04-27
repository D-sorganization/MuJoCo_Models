# SPDX-License-Identifier: MIT
"""Tests for postcondition contract checks."""

import pytest

from mujoco_models.shared.contracts.postconditions import (
    ensure_mjcf_root,
    ensure_positive_definite_inertia,
    ensure_positive_mass,
    ensure_valid_xml,
)


class TestEnsureValidXml:
    def test_valid_xml(self) -> None:
        root = ensure_valid_xml("<mujoco/>")
        assert root.tag == "mujoco"

    def test_invalid_xml_raises(self) -> None:
        with pytest.raises(ValueError, match="not well-formed"):
            ensure_valid_xml("<mujoco><broken")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not well-formed"):
            ensure_valid_xml("")


class TestEnsureMjcfRoot:
    def test_valid_mjcf(self) -> None:
        root = ensure_mjcf_root('<mujoco model="test"/>')
        assert root.tag == "mujoco"

    def test_wrong_root_tag(self) -> None:
        with pytest.raises(ValueError, match="must be <mujoco>"):
            ensure_mjcf_root("<opensim/>")

    def test_malformed_raises(self) -> None:
        with pytest.raises(ValueError, match="not well-formed"):
            ensure_mjcf_root("not xml at all")


class TestEnsurePositiveMass:
    def test_positive_mass_passes(self) -> None:
        ensure_positive_mass(1.0, "test_body")

    def test_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="not positive"):
            ensure_positive_mass(0.0, "test_body")

    def test_negative_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="not positive"):
            ensure_positive_mass(-1.0, "test_body")


class TestEnsurePositiveDefiniteInertia:
    def test_valid_inertia(self) -> None:
        ensure_positive_definite_inertia(1.0, 1.0, 1.0, "body")

    def test_zero_component_raises(self) -> None:
        with pytest.raises(ValueError, match="not positive"):
            ensure_positive_definite_inertia(0.0, 1.0, 1.0, "body")

    def test_negative_component_raises(self) -> None:
        with pytest.raises(ValueError, match="not positive"):
            ensure_positive_definite_inertia(-1.0, 1.0, 1.0, "body")

    def test_triangle_inequality_violation(self) -> None:
        with pytest.raises(ValueError, match="triangle inequality"):
            ensure_positive_definite_inertia(0.1, 0.1, 10.0, "body")

    def test_cylinder_like_inertia(self) -> None:
        ensure_positive_definite_inertia(0.5, 0.5, 0.3, "cyl")

    def test_all_components_checked(self) -> None:
        with pytest.raises(ValueError, match="Iyy"):
            ensure_positive_definite_inertia(1.0, -0.5, 1.0, "body")

        with pytest.raises(ValueError, match="Izz"):
            ensure_positive_definite_inertia(1.0, 1.0, -0.5, "body")
