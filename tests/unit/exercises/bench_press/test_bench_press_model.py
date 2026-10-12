# SPDX-License-Identifier: MIT
"""Tests for bench press model builder."""

import math
import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises.bench_press.bench_press_model import (
    BENCH_HEIGHT,
    BenchPressModelBuilder,
    build_bench_press_model,
)
from mujoco_models.shared.parity._canonical import spec
from mujoco_models.shared.parity.standard import STANDARD


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

    def test_set_initial_pose_leaves_refs_at_zero(self) -> None:
        """``ref`` never poses a segment (#407): the start pose is the keyframe."""
        builder = BenchPressModelBuilder()
        worldbody = ET.Element("worldbody")
        body = ET.SubElement(worldbody, "body")
        joint = ET.SubElement(body, "joint", name="elbow_l_flex", type="hinge")
        builder.set_initial_pose(worldbody)
        assert joint.get("ref") is None

    def test_start_pose_is_the_spec_unrack_keyframe(self) -> None:
        """The keyframe angles are the standard's bench ``start`` via-pose."""
        want = spec.pose_coordinates(STANDARD, "bench_press", "start")
        assert BenchPressModelBuilder().keyframe_angle_offsets() == want
        root = ET.fromstring(build_bench_press_model())
        names = [j.get("name") for j in root.iter("joint") if j.get("name")]
        qpos = [float(v) for v in root.find("keyframe/key").get("qpos").split()]  # type: ignore
        # qpos: hinges in document order, then barbell free joints.
        hinge_qpos = dict(zip(names, qpos, strict=False))
        assert hinge_qpos["shoulder_l_flex"] == pytest.approx(1.5)
        assert hinge_qpos["shoulder_l_adduct"] == pytest.approx(-0.3376)
        assert hinge_qpos["elbow_l_flex"] == 0.0

    def test_lifter_is_supine_on_the_bench(self) -> None:
        root = ET.fromstring(build_bench_press_model())
        pelvis = next(b for b in root.iter("body") if b.get("name") == "pelvis")
        quat = [float(v) for v in pelvis.get("quat", "").split()]
        assert quat == pytest.approx(
            [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0], abs=1e-5
        )
        assert pelvis.find("freejoint") is None
        assert pelvis.find("joint") is None

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
