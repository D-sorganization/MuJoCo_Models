"""Tests for gait model builder."""

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises.gait.gait_model import (
    _GAIT_INITIAL_REFS,
    GaitModelBuilder,
    build_gait_model,
)


class TestGaitModelBuilder:
    def test_exercise_name(self) -> None:
        builder = GaitModelBuilder()
        assert builder.exercise_name == "gait"

    def test_build_returns_xml(self) -> None:
        xml_str = build_gait_model()
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_model_name(self) -> None:
        xml_str = build_gait_model()
        root = ET.fromstring(xml_str)
        assert root.get("model") == "gait"

    def test_no_barbell_weld_to_body(self) -> None:
        """Gait has no barbell attachment to the body."""
        builder = GaitModelBuilder()
        equality = ET.Element("equality")
        builder.attach_barbell(equality, {}, {})
        # attach_barbell is a no-op, so no welds added by it
        assert len(equality.findall("weld")) == 0

    def test_has_gravity(self) -> None:
        xml_str = build_gait_model()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        assert "-9.806650" in option.get("gravity")

    def test_custom_body_mass(self) -> None:
        xml_str = build_gait_model(body_mass=65.0)
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_default_config(self) -> None:
        builder = GaitModelBuilder()
        assert builder.config.gravity == (0.0, 0.0, -9.80665)

    def test_set_initial_pose_no_error(self) -> None:
        builder = GaitModelBuilder()
        worldbody = ET.Element("worldbody")
        builder.set_initial_pose(worldbody)

    def test_set_initial_pose_modifies_hip_joints(self) -> None:
        builder = GaitModelBuilder()
        worldbody = ET.Element("worldbody")
        body = ET.SubElement(worldbody, "body")
        joint = ET.SubElement(body, "joint", name="hip_l_flex", type="hinge")
        builder.set_initial_pose(worldbody)
        assert joint.get("ref") is not None
        assert float(joint.get("ref")) == pytest.approx(
            _GAIT_INITIAL_REFS["hip_l_flex"], rel=1e-4
        )

    def test_set_initial_pose_asymmetric(self) -> None:
        """Gait initial pose should have different left/right hip angles."""
        assert _GAIT_INITIAL_REFS["hip_l_flex"] != _GAIT_INITIAL_REFS["hip_r_flex"]

    def test_build_has_actuators(self) -> None:
        xml_str = build_gait_model()
        root = ET.fromstring(xml_str)
        actuator = root.find("actuator")
        assert actuator is not None
        assert len(actuator.findall("position")) > 0

    def test_build_has_sensors(self) -> None:
        xml_str = build_gait_model()
        root = ET.fromstring(xml_str)
        sensor = root.find("sensor")
        assert sensor is not None
        assert len(sensor.findall("jointpos")) > 0

    def test_uses_barbell_flag_is_false(self) -> None:
        """GaitModelBuilder should opt out of barbell construction."""
        assert GaitModelBuilder().uses_barbell is False

    def test_build_omits_barbell_bodies(self) -> None:
        """With uses_barbell=False, no barbell bodies should appear."""
        xml_str = build_gait_model()
        root = ET.fromstring(xml_str)
        body_names = {b.get("name") for b in root.findall(".//body")}
        assert "barbell_shaft" not in body_names
        assert "barbell_left_sleeve" not in body_names
        assert "barbell_right_sleeve" not in body_names

    def test_build_omits_barbell_welds(self) -> None:
        """With uses_barbell=False, no barbell welds should be emitted."""
        xml_str = build_gait_model()
        root = ET.fromstring(xml_str)
        weld_names = {w.get("name") for w in root.findall(".//weld")}
        assert "barbell_left_weld" not in weld_names
        assert "barbell_right_weld" not in weld_names
