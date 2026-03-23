"""Tests for squat model builder."""

import xml.etree.ElementTree as ET

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
        assert root.get("model") == "back_squat"

    def test_has_barbell_weld(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}
        assert "barbell_to_torso" in weld_names

    def test_has_gravity(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        assert "-9.806650" in option.get("gravity")

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
        assert builder.config.gravity == (0.0, 0.0, -9.80665)

    def test_attach_barbell_adds_weld(self) -> None:
        builder = SquatModelBuilder()
        equality = ET.Element("equality")
        builder.attach_barbell(equality, {}, {})
        assert len(equality.findall("weld")) == 1

    def test_set_initial_pose_no_error(self) -> None:
        builder = SquatModelBuilder()
        worldbody = ET.Element("worldbody")
        builder.set_initial_pose(worldbody)
