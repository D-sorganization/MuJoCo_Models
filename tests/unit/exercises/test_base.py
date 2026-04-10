"""Regression tests for the shared exercise builder base class."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.barbell import BarbellSpec
from mujoco_models.shared.body import BodyModelSpec, segment_properties


class ForwardingBuilder(ExerciseModelBuilder):
    @property
    def exercise_name(self) -> str:
        return "forwarding_builder"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        self._attach_barbell_to_hands(equality, grip_offset=self.grip_offset)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        return None


class OverriddenAccessorBuilder(ForwardingBuilder):
    def __init__(self) -> None:
        super().__init__(
            ExerciseConfig(
                body_spec=BodyModelSpec(total_mass=90.0, height=1.82),
                barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=25.0),
                gravity=(1.0, 2.0, -3.0),
            )
        )
        self._body_spec = BodyModelSpec(total_mass=95.0, height=1.90)
        self._barbell_spec = BarbellSpec.womens_olympic(plate_mass_per_side=10.0)
        self._gravity = (0.5, -0.25, -4.5)
        self._grip_offset = (0.010, 0.020, 0.030, 1.0, 0.0, 0.0, 0.0)

    @property
    def body_spec(self) -> BodyModelSpec:
        return self._body_spec

    @property
    def barbell_spec(self) -> BarbellSpec:
        return self._barbell_spec

    @property
    def gravity(self) -> tuple[float, float, float]:
        return self._gravity

    @property
    def grip_offset(self) -> tuple[float, ...] | None:
        return self._grip_offset


class TestExerciseModelBuilderAccessors:
    def test_accessors_forward_config(self) -> None:
        builder = ForwardingBuilder(ExerciseConfig())
        assert builder.body_spec == builder.config.body_spec
        assert builder.barbell_spec == builder.config.barbell_spec
        assert builder.gravity == builder.config.gravity
        assert builder.grip_offset is None

    def test_build_uses_accessor_overrides(self) -> None:
        builder = OverriddenAccessorBuilder()
        xml_str = builder.build()
        root = ET.fromstring(xml_str)

        option = root.find("option")
        assert option is not None
        assert option.get("gravity") == "0.500000 -0.250000 -4.500000"

        pelvis = root.find(".//body[@name='pelvis']/inertial")
        assert pelvis is not None
        expected_pelvis_mass, _, _ = segment_properties(
            builder.body_spec.total_mass, builder.body_spec.height, "pelvis"
        )
        assert float(pelvis.get("mass")) == pytest.approx(expected_pelvis_mass)  # type: ignore[arg-type]

        barbell_shaft = root.find(".//body[@name='barbell_shaft']/inertial")
        assert barbell_shaft is not None
        assert float(barbell_shaft.get("mass")) == pytest.approx(
            builder.barbell_spec.shaft_mass
        )  # type: ignore[arg-type]

        weld = root.find(".//weld[@name='barbell_to_hand_l']")
        assert weld is not None
        assert (
            weld.get("relpose")
            == "0.010000 0.020000 0.030000 1.000000 0.000000 0.000000 0.000000"
        )
