"""Tests for deadlift model builder."""

import xml.etree.ElementTree as ET

from mujoco_models.exercises.deadlift.deadlift_model import (
    PLATE_RADIUS,
    DeadliftModelBuilder,
    build_deadlift_model,
)


class TestDeadliftModelBuilder:
    def test_exercise_name(self) -> None:
        builder = DeadliftModelBuilder()
        assert builder.exercise_name == "deadlift"

    def test_build_returns_xml(self) -> None:
        xml_str = build_deadlift_model()
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_model_name(self) -> None:
        xml_str = build_deadlift_model()
        root = ET.fromstring(xml_str)
        assert root.get("model") == "deadlift"

    def test_has_barbell_weld(self) -> None:
        xml_str = build_deadlift_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}
        assert "barbell_to_hand_l" in weld_names
        assert "barbell_to_hand_r" in weld_names

    def test_plate_radius_constant(self) -> None:
        assert PLATE_RADIUS == 0.225

    def test_custom_params(self) -> None:
        xml_str = build_deadlift_model(
            body_mass=90.0, height=1.85, plate_mass_per_side=100.0
        )
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_attach_barbell_adds_bilateral_welds(self) -> None:
        builder = DeadliftModelBuilder()
        equality = ET.Element("equality")
        builder.attach_barbell(equality, {}, {})
        welds = equality.findall("weld")
        assert len(welds) == 2
        weld_names = {w.get("name") for w in welds}
        assert "barbell_to_hand_l" in weld_names
        assert "barbell_to_hand_r" in weld_names

    def test_set_initial_pose_no_error(self) -> None:
        builder = DeadliftModelBuilder()
        worldbody = ET.Element("worldbody")
        builder.set_initial_pose(worldbody)

    def test_default_plate_mass(self) -> None:
        builder = DeadliftModelBuilder()
        assert builder.config.barbell_spec.plate_mass_per_side == 0.0
