# SPDX-License-Identifier: MIT
"""Tests for squat model builder."""

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises.squat.squat_model import (
    SquatModelBuilder,
    build_squat_model,
)


class TestSquatModelBuilder:
    def test_exercise_name(self) -> None:
        builder = SquatModelBuilder()
        assert builder.exercise_name == "back_squat"

    def test_build_returns_xml(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_model_name(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        assert root.get("model") == "back_squat"  # type: ignore

    def test_has_barbell_weld(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}  # type: ignore
        assert "barbell_to_torso" in weld_names

    def test_has_gravity(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        assert "-9.806650" in option.get("gravity")  # type: ignore

    def test_custom_body_mass(self) -> None:
        xml_str = build_squat_model(body_mass=100.0)
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_custom_plate_mass(self) -> None:
        xml_str = build_squat_model(plate_mass_per_side=0.0)
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_default_config(self) -> None:
        builder = SquatModelBuilder()
        assert builder.gravity == (0.0, 0.0, -9.80665)

    def test_attach_barbell_adds_weld(self) -> None:
        builder = SquatModelBuilder()
        equality = ET.Element("equality")
        builder.attach_barbell(equality, {}, {})
        assert len(equality.findall("weld")) == 1

    def test_set_initial_pose_no_error(self) -> None:
        builder = SquatModelBuilder()
        worldbody = ET.Element("worldbody")
        builder.set_initial_pose(worldbody)

    def test_set_initial_pose_modifies_hip_joints(self) -> None:
        from mujoco_models.exercises.squat.squat_model import _INITIAL_HIP_FLEX

        builder = SquatModelBuilder()
        worldbody = ET.Element("worldbody")
        body = ET.SubElement(worldbody, "body")
        joint = ET.SubElement(body, "joint", name="hip_l_flex", type="hinge")
        builder.set_initial_pose(worldbody)
        assert joint.get("ref") is not None  # type: ignore
        # ref is stored in radians (matches <compiler angle='radian'>)
        assert float(joint.get("ref")) == pytest.approx(_INITIAL_HIP_FLEX, rel=1e-4)  # type: ignore

    def test_set_initial_pose_modifies_knee_joints(self) -> None:
        from mujoco_models.exercises.squat.squat_model import _INITIAL_KNEE_FLEX

        builder = SquatModelBuilder()
        worldbody = ET.Element("worldbody")
        body = ET.SubElement(worldbody, "body")
        joint = ET.SubElement(body, "joint", name="knee_r_flex", type="hinge")
        builder.set_initial_pose(worldbody)
        assert joint.get("ref") is not None  # type: ignore
        # ref is stored in radians (matches <compiler angle='radian'>)
        assert float(joint.get("ref")) == pytest.approx(_INITIAL_KNEE_FLEX, rel=1e-4)  # type: ignore

    def test_build_has_actuators(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        actuator = root.find("actuator")
        assert actuator is not None
        assert len(actuator.findall("position")) > 0

    def test_build_has_sensors(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        sensor = root.find("sensor")
        assert sensor is not None
        assert len(sensor.findall("jointpos")) > 0
