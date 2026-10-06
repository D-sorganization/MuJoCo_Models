# SPDX-License-Identifier: MIT
"""Engine-free checks of the canonical axis convention (MuJoCo_Models#410).

MuJoCo's world frame is the canonical frame (X forward, Y left, Z up), and
body frames are world-aligned at the all-zero pose, so a joint's ``axis``
attribute IS its canonical axis.  The table below is copied from the fleet
standard (kinematics block, Repository_Management#2011) on purpose: it is an
independent restatement, not read back from the code under test.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.exercises.squat.squat_model import SquatModelBuilder

# coordinate -> (right/midline axis, mirrored on the left?)
_AXES: dict[str, tuple[tuple[int, int, int], bool]] = {
    "lumbar_flex": ((0, 1, 0), False),
    "lumbar_lateral": ((-1, 0, 0), False),
    "lumbar_rotate": ((0, 0, 1), False),
    "neck_flex": ((0, 1, 0), False),
    "shoulder_{s}_flex": ((0, -1, 0), False),
    "shoulder_{s}_adduct": ((1, 0, 0), True),
    "shoulder_{s}_rotate": ((0, 0, 1), True),
    "elbow_{s}_flex": ((0, -1, 0), False),
    "wrist_{s}_flex": ((0, -1, 0), False),
    "wrist_{s}_deviate": ((1, 0, 0), True),
    "hip_{s}_flex": ((0, -1, 0), False),
    "hip_{s}_adduct": ((1, 0, 0), True),
    "hip_{s}_rotate": ((0, 0, 1), True),
    "knee_{s}_flex": ((0, -1, 0), False),
    "ankle_{s}_flex": ((0, -1, 0), False),
    "ankle_{s}_invert": ((1, 0, 0), True),
}


def _expected() -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for name, (axis, mirror) in _AXES.items():
        if "{s}" not in name:
            out[name] = (axis[0] + 0.0, axis[1] + 0.0, axis[2] + 0.0)
            continue
        for side in ("l", "r"):
            f = -1.0 if (mirror and side == "l") else 1.0
            out[name.format(s=side)] = tuple(f * a + 0.0 for a in axis)  # type: ignore[assignment]
    return out


@pytest.fixture(scope="module")
def squat_root() -> ET.Element:
    return ET.fromstring(SquatModelBuilder().build())


def _vec(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split())


def test_every_hinge_axis_matches_the_standard(squat_root: ET.Element) -> None:
    joints = {
        j.get("name"): _vec(j.get("axis", ""))
        for j in squat_root.iter("joint")
        if j.get("type") == "hinge"
    }
    expected = _expected()
    assert set(joints) == set(expected)
    for name, axis in expected.items():
        assert joints[name] == pytest.approx(axis), name


def test_no_human_body_is_rotated_at_the_zero_pose(squat_root: ET.Element) -> None:
    """Axis literals are canonical only while body frames stay world-aligned."""
    human = {"pelvis", "torso", "head"} | {
        f"{seg}_{s}"
        for seg in ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")
        for s in ("l", "r")
    }
    for body in squat_root.iter("body"):
        if body.get("name") in human:
            assert body.get("quat") is None and body.get("euler") is None


@pytest.mark.parametrize("exercise", sorted(EXERCISE_REGISTRY))
def test_bilateral_bodies_sit_on_their_own_side(exercise: str) -> None:
    root = ET.fromstring(EXERCISE_REGISTRY[exercise]().build())
    bodies = {b.get("name"): b for b in root.iter("body")}
    for seg in ("thigh", "upper_arm"):  # the two offset roots
        left = _vec(bodies[f"{seg}_l"].get("pos", ""))
        right = _vec(bodies[f"{seg}_r"].get("pos", ""))
        assert left[1] > 0.0 > right[1], seg
        assert left[1] == pytest.approx(-right[1])
        assert left[0] == pytest.approx(0.0) and right[0] == pytest.approx(0.0)
    for seg in ("shank", "forearm", "hand", "foot"):  # chained: on the limb axis
        for side in ("l", "r"):
            pos = _vec(bodies[f"{seg}_{side}"].get("pos", ""))
            assert pos[0] == pos[1] == 0.0


def test_barbell_lies_along_y_with_left_sleeve_at_plus_y() -> None:
    root = ET.fromstring(SquatModelBuilder().build())
    pos = {
        b.get("name"): _vec(b.get("pos", ""))
        for b in root.iter("body")
        if str(b.get("name")).startswith("barbell_")
    }
    shaft = pos["barbell_shaft"]
    left, right = pos["barbell_left_sleeve"], pos["barbell_right_sleeve"]
    assert left[1] > shaft[1] > right[1]
    assert left[1] - shaft[1] == pytest.approx(shaft[1] - right[1])
    for sleeve in (left, right):
        assert sleeve[0] == pytest.approx(shaft[0])
        assert sleeve[2] == pytest.approx(shaft[2])


def test_barbell_geoms_are_horizontal_cylinders_along_y() -> None:
    """The euler angle is radians (the compiler is angle='radian')."""
    root = ET.fromstring(SquatModelBuilder().build())
    for body in root.iter("body"):
        if str(body.get("name")).startswith("barbell_"):
            geom = body.find("geom")
            assert geom is not None
            euler = _vec(geom.get("euler", ""))
            assert euler == pytest.approx((1.5707963, 0.0, 0.0), abs=1e-6)
