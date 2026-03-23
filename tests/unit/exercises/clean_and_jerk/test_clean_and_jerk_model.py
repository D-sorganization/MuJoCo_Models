"""Tests for clean and jerk model builder."""

import xml.etree.ElementTree as ET

from mujoco_models.exercises.clean_and_jerk.clean_and_jerk_model import (
    CleanAndJerkModelBuilder,
    build_clean_and_jerk_model,
)


class TestCleanAndJerkModelBuilder:
    def test_exercise_name(self) -> None:
        builder = CleanAndJerkModelBuilder()
        assert builder.exercise_name == "clean_and_jerk"

    def test_build_returns_xml(self) -> None:
        xml_str = build_clean_and_jerk_model()
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_model_name(self) -> None:
        xml_str = build_clean_and_jerk_model()
        root = ET.fromstring(xml_str)
        assert root.get("model") == "clean_and_jerk"

    def test_has_barbell_weld(self) -> None:
        xml_str = build_clean_and_jerk_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}
        assert "barbell_to_left_hand" in weld_names

    def test_custom_params(self) -> None:
        xml_str = build_clean_and_jerk_model(
            body_mass=85.0, height=1.78, plate_mass_per_side=55.0
        )
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_attach_barbell_adds_weld(self) -> None:
        builder = CleanAndJerkModelBuilder()
        equality = ET.Element("equality")
        builder.attach_barbell(equality, {}, {})
        assert len(equality.findall("weld")) == 1

    def test_set_initial_pose_no_error(self) -> None:
        builder = CleanAndJerkModelBuilder()
        worldbody = ET.Element("worldbody")
        builder.set_initial_pose(worldbody)

    def test_default_gravity_z_up(self) -> None:
        builder = CleanAndJerkModelBuilder()
        assert builder.config.gravity == (0.0, 0.0, -9.80665)
