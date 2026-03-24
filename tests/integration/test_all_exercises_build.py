"""Integration tests: verify all five exercise models build end-to-end.

Each model must produce well-formed MJCF XML with the correct structure:
<mujoco> root with option, compiler, default, worldbody, and equality sections.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable

import pytest

from mujoco_models.exercises.bench_press.bench_press_model import (
    build_bench_press_model,
)
from mujoco_models.exercises.clean_and_jerk.clean_and_jerk_model import (
    build_clean_and_jerk_model,
)
from mujoco_models.exercises.deadlift.deadlift_model import build_deadlift_model
from mujoco_models.exercises.snatch.snatch_model import build_snatch_model
from mujoco_models.exercises.squat.squat_model import build_squat_model

ALL_BUILDERS = [
    ("back_squat", build_squat_model),
    ("bench_press", build_bench_press_model),
    ("deadlift", build_deadlift_model),
    ("snatch", build_snatch_model),
    ("clean_and_jerk", build_clean_and_jerk_model),
]


class TestAllExercisesBuild:
    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_produces_valid_mjcf(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        assert root.tag == "mujoco"

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_model_name_matches(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        assert root.get("model") == name  # type: ignore

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_has_gravity(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        gravity = option.get("gravity")
        assert gravity is not None
        assert "-9.806650" in gravity

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_z_up_gravity(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        gravity = option.get("gravity")
        assert gravity is not None
        parts = [float(x) for x in gravity.split()]
        assert parts[0] == pytest.approx(0.0)
        assert parts[1] == pytest.approx(0.0)
        assert parts[2] == pytest.approx(-9.80665)

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_has_worldbody(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        assert root.find("worldbody") is not None  # type: ignore

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_has_equality(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        assert root.find("equality") is not None  # type: ignore

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_minimum_body_count(self, name: str, builder: Callable[[], str]) -> None:
        """Every exercise should have at least 18 bodies (15 body + 3 barbell)."""
        xml_str = builder()
        root = ET.fromstring(xml_str)
        bodies = root.findall(".//body")
        assert len(bodies) >= 18

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_all_masses_positive(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        for body in root.findall(".//body"):
            inertial = body.find("inertial")
            if inertial is not None:
                mass = float(inertial.get("mass"))  # type: ignore
                assert mass > 0, f"{body.get('name')} mass={mass}"  # type: ignore

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_barbell_present(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        body_names = {b.get("name") for b in root.findall(".//body")}  # type: ignore
        assert "barbell_shaft" in body_names
        assert "barbell_left_sleeve" in body_names
        assert "barbell_right_sleeve" in body_names

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_has_compiler(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        compiler = root.find("compiler")
        assert compiler is not None
        assert compiler.get("angle") == "radian"  # type: ignore

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_has_ground_plane(self, name: str, builder: Callable[[], str]) -> None:
        xml_str = builder()
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        ground_geoms = [
            g for g in worldbody.findall("geom") if g.get("name") == "ground"  # type: ignore
        ]
        assert len(ground_geoms) == 1
        assert ground_geoms[0].get("type") == "plane"  # type: ignore

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_actuator_sensor_count(self, name: str, builder: Callable[[], str]) -> None:
        """Every exercise should have exactly 14 position actuators and 14 jointpos sensors."""
        xml_str = builder()
        root = ET.fromstring(xml_str)
        actuators = root.findall(".//actuator/position")
        sensors = root.findall(".//sensor/jointpos")
        assert len(actuators) == 14
        assert len(sensors) == 14
