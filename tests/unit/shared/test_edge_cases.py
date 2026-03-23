"""Edge-case anthropometric and model-building tests.

Tests extreme but valid anthropometric parameters (very small/large people),
ensuring the model generation pipeline handles boundary conditions correctly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.exercises.base import ExerciseConfig
from mujoco_models.shared.barbell import BarbellSpec
from mujoco_models.shared.body import BodyModelSpec, create_full_body
from mujoco_models.shared.utils.geometry import cylinder_inertia


class TestExtremeAnthropometrics:
    """Test with boundary anthropometric values."""

    def test_very_light_person(self) -> None:
        """30 kg person (child or very small adult)."""
        wb = ET.Element("worldbody")
        bodies = create_full_body(wb, BodyModelSpec(total_mass=30.0, height=1.20))
        assert len(bodies) == 15
        for name, body in bodies.items():
            mass = float(body.find("inertial").get("mass"))
            assert mass > 0, f"{name} mass not positive for light person"

    def test_very_heavy_person(self) -> None:
        """200 kg person (super heavyweight)."""
        wb = ET.Element("worldbody")
        bodies = create_full_body(wb, BodyModelSpec(total_mass=200.0, height=1.90))
        assert len(bodies) == 15
        for name, body in bodies.items():
            mass = float(body.find("inertial").get("mass"))
            assert mass > 0, f"{name} mass not positive for heavy person"

    def test_very_short_person(self) -> None:
        """1.0 m person (extreme short stature)."""
        wb = ET.Element("worldbody")
        bodies = create_full_body(wb, BodyModelSpec(total_mass=40.0, height=1.0))
        assert len(bodies) == 15

    def test_very_tall_person(self) -> None:
        """2.3 m person (extreme tall stature)."""
        wb = ET.Element("worldbody")
        bodies = create_full_body(wb, BodyModelSpec(total_mass=120.0, height=2.3))
        assert len(bodies) == 15

    def test_minimum_viable_params(self) -> None:
        """Smallest plausible values just above zero."""
        wb = ET.Element("worldbody")
        bodies = create_full_body(wb, BodyModelSpec(total_mass=0.1, height=0.1))
        assert len(bodies) == 15


class TestExtremeBarbell:
    """Test barbell at extreme but valid configurations."""

    def test_empty_bar(self) -> None:
        config = ExerciseConfig(
            barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=0.0),
        )
        assert config.barbell_spec.total_mass == pytest.approx(20.0)

    def test_very_heavy_plates(self) -> None:
        config = ExerciseConfig(
            barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=250.0),
        )
        assert config.barbell_spec.total_mass == pytest.approx(520.0)

    def test_womens_bar_empty(self) -> None:
        spec = BarbellSpec.womens_olympic()
        assert spec.bar_mass == 15.0
        assert spec.total_mass == pytest.approx(15.0)


class TestExtremeModelBuild:
    """Test complete model builds with extreme parameters."""

    @pytest.mark.parametrize(
        "mass,height",
        [
            (30.0, 1.0),
            (200.0, 2.3),
            (50.0, 1.50),
            (120.0, 2.0),
        ],
    )
    def test_all_exercises_build_at_extremes(self, mass: float, height: float) -> None:
        config = ExerciseConfig(
            body_spec=BodyModelSpec(total_mass=mass, height=height),
            barbell_spec=BarbellSpec.mens_olympic(),
        )
        seen: set[str] = set()
        for name, builder_cls in EXERCISE_REGISTRY.items():
            cls_name = builder_cls.__name__
            if cls_name in seen:
                continue
            seen.add(cls_name)
            xml_str = builder_cls(config).build()
            root = ET.fromstring(xml_str)
            assert root.tag == "mujoco", f"{name} failed at mass={mass}, h={height}"


class TestInertiaEdgeCases:
    """Verify inertia calculations at extreme dimensions."""

    def test_very_thin_cylinder(self) -> None:
        ixx, iyy, izz = cylinder_inertia(mass=1.0, radius=0.001, length=2.0)
        assert ixx > 0
        assert iyy > 0
        assert izz > 0

    def test_very_thick_cylinder(self) -> None:
        ixx, iyy, izz = cylinder_inertia(mass=100.0, radius=5.0, length=0.01)
        assert ixx > 0
        assert izz > ixx  # disc-like: axial > transverse

    def test_very_light_mass(self) -> None:
        ixx, iyy, izz = cylinder_inertia(mass=0.001, radius=0.1, length=1.0)
        assert ixx > 0
        assert iyy > 0
        assert izz > 0
