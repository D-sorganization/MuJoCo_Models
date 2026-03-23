"""Tests for bench press model builder."""

import xml.etree.ElementTree as ET

from mujoco_models.exercises.bench_press.bench_press_model import (
    BENCH_HEIGHT,
    BenchPressModelBuilder,
    build_bench_press_model,
)


class TestBenchPressModelBuilder:
    def test_exercise_name(self) -> None:
        builder = BenchPressModelBuilder()
        assert builder.exercise_name == "bench_press"

    def test_build_returns_xml(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_model_name(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        assert root.get("model") == "bench_press"

    def test_has_barbell_weld(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}
        assert "barbell_to_hand_l" in weld_names
        assert "barbell_to_hand_r" in weld_names

    def test_bench_height_constant(self) -> None:
        assert BENCH_HEIGHT == 0.43

    def test_custom_params(self) -> None:
        xml_str = build_bench_press_model(
            body_mass=90.0, height=1.80, plate_mass_per_side=40.0
        )
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    def test_attach_barbell_adds_bilateral_welds(self) -> None:
        builder = BenchPressModelBuilder()
        equality = ET.Element("equality")
        builder.attach_barbell(equality, {}, {})
        welds = equality.findall("weld")
        assert len(welds) == 2
        weld_names = {w.get("name") for w in welds}
        assert "barbell_to_hand_l" in weld_names
        assert "barbell_to_hand_r" in weld_names

    def test_set_initial_pose_no_error(self) -> None:
        builder = BenchPressModelBuilder()
        worldbody = ET.Element("worldbody")
        builder.set_initial_pose(worldbody)

    def test_default_config_gravity(self) -> None:
        builder = BenchPressModelBuilder()
        assert builder.config.gravity[2] < 0
