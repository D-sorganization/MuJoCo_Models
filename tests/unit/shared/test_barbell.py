# SPDX-License-Identifier: MIT
"""Tests for barbell model generation."""

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.shared.barbell import BarbellSpec, create_barbell_bodies


class TestBarbellSpec:
    def test_mens_defaults(self) -> None:
        spec = BarbellSpec.mens_olympic()
        assert spec.total_length == 2.20
        assert spec.bar_mass == 20.0  # type: ignore
        assert spec.shaft_diameter == 0.028
        assert spec.plate_mass_per_side == 0.0

    def test_womens_defaults(self) -> None:
        spec = BarbellSpec.womens_olympic()
        assert spec.total_length == 2.01
        assert spec.bar_mass == 15.0  # type: ignore
        assert spec.shaft_diameter == 0.025

    def test_sleeve_length(self) -> None:
        spec = BarbellSpec.mens_olympic()
        expected = (2.20 - 1.31) / 2.0
        assert spec.sleeve_length == pytest.approx(expected)

    def test_total_mass_with_plates(self) -> None:
        spec = BarbellSpec.mens_olympic(plate_mass_per_side=60.0)
        assert spec.total_mass == pytest.approx(140.0)  # type: ignore

    def test_shaft_mass_proportional(self) -> None:
        spec = BarbellSpec.mens_olympic()
        assert spec.shaft_mass == pytest.approx(20.0 * 1.31 / 2.20)

    def test_sleeve_mass(self) -> None:
        spec = BarbellSpec.mens_olympic()
        expected = 20.0 * spec.sleeve_length / 2.20
        assert spec.sleeve_mass == pytest.approx(expected)

    def test_shaft_radius(self) -> None:
        spec = BarbellSpec.mens_olympic()
        assert spec.shaft_radius == pytest.approx(0.014)

    def test_sleeve_radius(self) -> None:
        spec = BarbellSpec.mens_olympic()
        assert spec.sleeve_radius == pytest.approx(0.025)

    def test_rejects_negative_plate_mass(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BarbellSpec(plate_mass_per_side=-1.0)

    def test_rejects_shaft_longer_than_total(self) -> None:
        with pytest.raises(ValueError, match="shaft_length"):
            BarbellSpec(total_length=1.0, shaft_length=1.5)

    def test_rejects_zero_diameter(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BarbellSpec(shaft_diameter=0.0)

    def test_rejects_zero_bar_mass(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BarbellSpec(bar_mass=0.0)

    def test_rejects_zero_total_length(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BarbellSpec(total_length=0.0)

    def test_rejects_shaft_equal_to_total(self) -> None:
        with pytest.raises(ValueError, match="shaft_length"):
            BarbellSpec(total_length=2.0, shaft_length=2.0)

    def test_frozen(self) -> None:
        spec = BarbellSpec.mens_olympic()
        with pytest.raises(AttributeError):
            spec.bar_mass = 25.0  # type: ignore

    def test_womens_plate_mass(self) -> None:
        spec = BarbellSpec.womens_olympic(plate_mass_per_side=30.0)
        assert spec.total_mass == pytest.approx(75.0)  # type: ignore


class TestCreateBarbellBodies:
    @pytest.fixture()
    def elements(self) -> tuple[ET.Element, ET.Element]:
        worldbody = ET.Element("worldbody")
        equality = ET.Element("equality")
        return worldbody, equality

    def test_creates_three_bodies(
        self, elements: tuple[ET.Element, ET.Element]
    ) -> None:
        worldbody, equality = elements
        spec = BarbellSpec.mens_olympic()
        bodies = create_barbell_bodies(worldbody, equality, spec)
        assert len(bodies) == 3
        assert "barbell_shaft" in bodies
        assert "barbell_left_sleeve" in bodies
        assert "barbell_right_sleeve" in bodies

    def test_creates_two_weld_constraints(
        self, elements: tuple[ET.Element, ET.Element]
    ) -> None:
        worldbody, equality = elements
        spec = BarbellSpec.mens_olympic()
        create_barbell_bodies(worldbody, equality, spec)
        welds = equality.findall("weld")
        assert len(welds) == 2

    def test_custom_prefix(self, elements: tuple[ET.Element, ET.Element]) -> None:
        worldbody, equality = elements
        spec = BarbellSpec.mens_olympic()
        bodies = create_barbell_bodies(worldbody, equality, spec, prefix="bar")
        assert "bar_shaft" in bodies

    def test_mass_conservation(self, elements: tuple[ET.Element, ET.Element]) -> None:
        worldbody, equality = elements
        spec = BarbellSpec.mens_olympic(plate_mass_per_side=50.0)
        create_barbell_bodies(worldbody, equality, spec)
        total = sum(
            float(b.find("inertial").get("mass"))
            for b in worldbody.findall("body")  # type: ignore
        )
        assert total == pytest.approx(spec.total_mass, rel=1e-4)

    def test_bodies_have_inertial(
        self, elements: tuple[ET.Element, ET.Element]
    ) -> None:
        worldbody, equality = elements
        spec = BarbellSpec.mens_olympic()
        create_barbell_bodies(worldbody, equality, spec)
        for body in worldbody.findall("body"):
            assert body.find("inertial") is not None  # type: ignore

    def test_bodies_have_geom(self, elements: tuple[ET.Element, ET.Element]) -> None:
        worldbody, equality = elements
        spec = BarbellSpec.mens_olympic()
        create_barbell_bodies(worldbody, equality, spec)
        for body in worldbody.findall("body"):
            assert body.find("geom") is not None  # type: ignore

    def test_shaft_cylinder_geom(self, elements: tuple[ET.Element, ET.Element]) -> None:
        worldbody, equality = elements
        spec = BarbellSpec.mens_olympic()
        bodies = create_barbell_bodies(worldbody, equality, spec)
        shaft = bodies["barbell_shaft"]
        geom = shaft.find("geom")
        assert geom.get("type") == "cylinder"  # type: ignore
