"""Integration tests: verify all exercise models build end-to-end.

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
from mujoco_models.exercises.gait.gait_model import build_gait_model
from mujoco_models.exercises.sit_to_stand.sit_to_stand_model import (
    build_sit_to_stand_model,
)
from mujoco_models.exercises.snatch.snatch_model import build_snatch_model
from mujoco_models.exercises.squat.squat_model import build_squat_model

ALL_BUILDERS = [
    ("back_squat", build_squat_model),
    ("bench_press", build_bench_press_model),
    ("deadlift", build_deadlift_model),
    ("snatch", build_snatch_model),
    ("clean_and_jerk", build_clean_and_jerk_model),
    ("gait", build_gait_model),
    ("sit_to_stand", build_sit_to_stand_model),
]

# Exercises that build a barbell (uses_barbell=True). Gait and sit-to-stand
# are bodyweight movements and intentionally omit the barbell entirely.
BARBELL_BUILDERS = [
    (name, builder)
    for name, builder in ALL_BUILDERS
    if name not in {"gait", "sit_to_stand"}
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
        """Every exercise should have at least 15 body segments.

        Barbell exercises add 3 more bodies (shaft + 2 sleeves); bodyweight
        exercises (gait, sit-to-stand) only need the 15 body segments.
        """
        xml_str = builder()
        root = ET.fromstring(xml_str)
        bodies = root.findall(".//body")
        expected_min = 18 if name not in {"gait", "sit_to_stand"} else 15
        assert len(bodies) >= expected_min

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
        "name,builder", BARBELL_BUILDERS, ids=[n for n, _ in BARBELL_BUILDERS]
    )
    def test_barbell_present(self, name: str, builder: Callable[[], str]) -> None:
        """Barbell exercises must materialise a shaft and both sleeves."""
        xml_str = builder()
        root = ET.fromstring(xml_str)
        body_names = {b.get("name") for b in root.findall(".//body")}  # type: ignore
        assert "barbell_shaft" in body_names
        assert "barbell_left_sleeve" in body_names
        assert "barbell_right_sleeve" in body_names

    @pytest.mark.parametrize(
        "name",
        ["gait", "sit_to_stand"],
    )
    def test_bodyweight_exercises_omit_barbell(self, name: str) -> None:
        """Gait and sit-to-stand must not emit any barbell bodies."""
        builder = dict(ALL_BUILDERS)[name]
        xml_str = builder()
        root = ET.fromstring(xml_str)
        body_names = {b.get("name") for b in root.findall(".//body")}  # type: ignore
        assert "barbell_shaft" not in body_names
        assert "barbell_left_sleeve" not in body_names
        assert "barbell_right_sleeve" not in body_names

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
            g
            for g in worldbody.findall("geom")
            if g.get("name") == "ground"  # type: ignore
        ]
        assert len(ground_geoms) == 1
        assert ground_geoms[0].get("type") == "plane"  # type: ignore

    @pytest.mark.parametrize(
        "name,builder", ALL_BUILDERS, ids=[n for n, _ in ALL_BUILDERS]
    )
    def test_actuator_sensor_count(self, name: str, builder: Callable[[], str]) -> None:
        """Every exercise should have exactly 28 position actuators and 28 jointpos sensors.

        Joint breakdown (multi-DOF upgrade):
          lumbar (3) + neck (1) + 2*shoulder (3) + 2*elbow (1) + 2*wrist (2)
          + 2*hip (3) + 2*knee (1) + 2*ankle (2) = 28
        """
        xml_str = builder()
        root = ET.fromstring(xml_str)
        actuators = root.findall(".//actuator/position")
        sensors = root.findall(".//sensor/jointpos")
        assert len(actuators) == 28
        assert len(sensors) == 28
