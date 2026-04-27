# SPDX-License-Identifier: MIT
"""Tests for bench press model builder."""

import xml.etree.ElementTree as ET

import pytest

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
        assert root.get("model") == "bench_press"  # type: ignore

    def test_has_barbell_weld(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}  # type: ignore
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
        weld_names = {w.get("name") for w in welds}  # type: ignore
        assert "barbell_to_hand_l" in weld_names
        assert "barbell_to_hand_r" in weld_names

    def test_set_initial_pose_modifies_joints(self) -> None:
        builder = BenchPressModelBuilder()
        worldbody = ET.Element("worldbody")
        body = ET.SubElement(worldbody, "body")
        joint = ET.SubElement(body, "joint", name="shoulder_l_flex", type="hinge")
        builder.set_initial_pose(worldbody)
        assert joint.get("ref") is not None, "shoulder joint should have ref set"  # type: ignore

    def test_set_initial_pose_shoulder_ref(self) -> None:
        from mujoco_models.exercises.bench_press.bench_press_model import (
            _INITIAL_SHOULDER_FLEX,
        )

        builder = BenchPressModelBuilder()
        worldbody = ET.Element("worldbody")
        body = ET.SubElement(worldbody, "body")
        ET.SubElement(body, "joint", name="shoulder_r_flex", type="hinge")
        builder.set_initial_pose(worldbody)
        joint = worldbody.find(".//joint")
        # ref is stored in radians (matches <compiler angle='radian'>)
        assert float(joint.get("ref")) == pytest.approx(  # type: ignore
            _INITIAL_SHOULDER_FLEX, rel=1e-4
        )

    def test_build_has_bench_body(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        body_names = {b.get("name") for b in root.findall(".//body")}  # type: ignore
        assert "bench" in body_names

    def test_build_has_pelvis_weld(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        welds = root.findall(".//weld")
        weld_names = {w.get("name") for w in welds}  # type: ignore
        assert "pelvis_to_bench" in weld_names

    def test_build_has_actuators(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        actuator = root.find("actuator")
        assert actuator is not None
        assert len(actuator.findall("position")) > 0

    def test_build_has_sensors(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        sensor = root.find("sensor")
        assert sensor is not None
        assert len(sensor.findall("jointpos")) > 0

    def test_default_config_gravity(self) -> None:
        builder = BenchPressModelBuilder()
        assert builder.gravity[2] < 0
