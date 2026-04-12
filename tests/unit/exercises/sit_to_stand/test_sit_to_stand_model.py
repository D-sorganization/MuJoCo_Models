"""Tests for sit-to-stand model builder."""

import math
import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises.sit_to_stand.sit_to_stand_model import (
    _CHAIR_SEAT_HEIGHT,
    _STS_INITIAL_REFS,
    SitToStandModelBuilder,
    build_sit_to_stand_model,
)


class TestSitToStandModelBuilder:
    def test_exercise_name(self) -> None:
        builder = SitToStandModelBuilder()
        assert builder.exercise_name == "sit_to_stand"

    def test_build_returns_xml(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_model_name(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        assert root.get("model") == "sit_to_stand"

    def test_no_barbell_weld_to_body(self) -> None:
        """Sit-to-stand has no barbell attachment."""
        builder = SitToStandModelBuilder()
        equality = ET.Element("equality")
        builder.attach_barbell(equality, {}, {})
        assert len(equality.findall("weld")) == 0

    def test_has_gravity(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        assert "-9.806650" in option.get("gravity")

    def test_has_chair_body(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        body_names = {b.get("name") for b in root.findall(".//body")}
        assert "chair" in body_names

    def test_chair_has_seat_geom(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        geom_names = {g.get("name") for g in root.findall(".//geom")}
        assert "chair_seat" in geom_names

    def test_chair_has_back_geom(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        geom_names = {g.get("name") for g in root.findall(".//geom")}
        assert "chair_back" in geom_names

    def test_chair_welded_to_world(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}
        assert "chair_to_world" in weld_names

    def test_chair_seat_height(self) -> None:
        assert 0.40 <= _CHAIR_SEAT_HEIGHT <= 0.50

    def test_custom_body_mass(self) -> None:
        xml_str = build_sit_to_stand_model(body_mass=60.0)
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_default_config(self) -> None:
        builder = SitToStandModelBuilder()
        assert builder.config.gravity == (0.0, 0.0, -9.80665)

    def test_set_initial_pose_no_error(self) -> None:
        builder = SitToStandModelBuilder()
        worldbody = ET.Element("worldbody")
        builder.set_initial_pose(worldbody)

    def test_set_initial_pose_hip_90deg(self) -> None:
        builder = SitToStandModelBuilder()
        worldbody = ET.Element("worldbody")
        body = ET.SubElement(worldbody, "body")
        joint = ET.SubElement(body, "joint", name="hip_l_flex", type="hinge")
        builder.set_initial_pose(worldbody)
        assert joint.get("ref") is not None
        assert float(joint.get("ref")) == pytest.approx(math.radians(90), rel=1e-3)

    def test_set_initial_pose_symmetric(self) -> None:
        """Sit-to-stand initial pose should be symmetric (both hips same)."""
        assert _STS_INITIAL_REFS["hip_l_flex"] == _STS_INITIAL_REFS["hip_r_flex"]

    def test_build_has_actuators(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        actuator = root.find("actuator")
        assert actuator is not None
        assert len(actuator.findall("position")) > 0

    def test_build_has_sensors(self) -> None:
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        sensor = root.find("sensor")
        assert sensor is not None
        assert len(sensor.findall("jointpos")) > 0

    def test_uses_barbell_flag_is_false(self) -> None:
        """SitToStandModelBuilder should opt out of barbell construction."""
        assert SitToStandModelBuilder().uses_barbell is False

    def test_build_omits_barbell_bodies(self) -> None:
        """With uses_barbell=False, no barbell bodies should appear."""
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        body_names = {b.get("name") for b in root.findall(".//body")}
        assert "barbell_shaft" not in body_names
        assert "barbell_left_sleeve" not in body_names
        assert "barbell_right_sleeve" not in body_names

    def test_build_omits_barbell_welds(self) -> None:
        """With uses_barbell=False, no barbell welds should be emitted."""
        xml_str = build_sit_to_stand_model()
        root = ET.fromstring(xml_str)
        weld_names = {w.get("name") for w in root.findall(".//weld")}
        assert "barbell_left_weld" not in weld_names
        assert "barbell_right_weld" not in weld_names
