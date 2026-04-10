"""Tests for snatch model builder."""

import xml.etree.ElementTree as ET

from mujoco_models.exercises.snatch.snatch_model import (
    SnatchModelBuilder,
    build_snatch_model,
)


class TestSnatchModelBuilder:
    def test_exercise_name(self) -> None:
        builder = SnatchModelBuilder()
        assert builder.exercise_name == "snatch"

    def test_build_returns_xml(self) -> None:
        xml_str = build_snatch_model()
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_model_name(self) -> None:
        xml_str = build_snatch_model()
        root = ET.fromstring(xml_str)
        assert root.get("model") == "snatch"  # type: ignore

    def test_has_barbell_weld(self) -> None:
        xml_str = build_snatch_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}  # type: ignore
        assert "barbell_to_hand_l" in weld_names
        assert "barbell_to_hand_r" in weld_names

    def test_custom_params(self) -> None:
        xml_str = build_snatch_model(
            body_mass=70.0, height=1.70, plate_mass_per_side=35.0
        )
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_attach_barbell_adds_bilateral_welds(self) -> None:
        builder = SnatchModelBuilder()
        equality = ET.Element("equality")
        builder.attach_barbell(equality, {}, {})
        welds = equality.findall("weld")
        assert len(welds) == 2
        weld_names = {w.get("name") for w in welds}  # type: ignore
        assert "barbell_to_hand_l" in weld_names
        assert "barbell_to_hand_r" in weld_names

    def test_set_initial_pose_no_error(self) -> None:
        builder = SnatchModelBuilder()
        worldbody = ET.Element("worldbody")
        builder.set_initial_pose(worldbody)

    def test_default_config_z_up_gravity(self) -> None:
        builder = SnatchModelBuilder()
        gx, gy, gz = builder.gravity
        assert gx == 0.0
        assert gy == 0.0
        assert gz < 0
