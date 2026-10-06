# SPDX-License-Identifier: MIT
"""Real-engine kinematic direction checks (MuJoCo_Models#410, RM#2011).

Measures each coordinate's rotation axis exactly as the fingerprint adapter
does and compares it with the fleet standard, plus one physical sanity probe:
positive hip flexion swings the shank FORWARD (+X), never sideways.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.shared.parity import fingerprint as fp_mod
from mujoco_models.shared.parity._canonical import conformance, kinematics

STD = conformance.load_standard()
EXERCISES = sorted(EXERCISE_REGISTRY)


@pytest.mark.parametrize("exercise", EXERCISES)
def test_axes_and_sides_match_the_standard(exercise: str) -> None:
    fp = fp_mod.fingerprint(exercise)
    assert set(fp["coordinate_axes"]) == set(kinematics.expected_axes(STD))
    assert kinematics.check_axes(fp, STD) == []
    assert kinematics.check_sides(fp, STD) == []


def _shank_in_pelvis_frame(model: mujoco.MjModel, hip: str, angle: float) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    adr = model.jnt_qposadr[model.joint("ground_pelvis").id]
    data.qpos[adr : adr + 7] = [0, 0, 0, 1, 0, 0, 0]
    data.qpos[model.jnt_qposadr[model.joint(hip).id]] += angle
    mujoco.mj_kinematics(model, data)
    pelvis = model.body("pelvis").id
    rel = data.xpos[model.body("shank_l").id] - data.xpos[pelvis]
    return data.xmat[pelvis].reshape(3, 3).T @ rel


@pytest.mark.parametrize("exercise", ["squat", "bench_press"])
def test_positive_hip_flexion_moves_the_shank_forward(exercise: str) -> None:
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY[exercise]().build())
    rest = _shank_in_pelvis_frame(model, "hip_l_flex", 0.0)
    flexed = _shank_in_pelvis_frame(model, "hip_l_flex", np.radians(30.0))
    delta = flexed - rest
    assert delta[0] > 0.1  # forward (+X)
    assert abs(delta[1]) < 1e-6  # no lateral drift
    assert delta[2] > 0.0  # a hanging shank rises as it swings forward


def test_positive_lumbar_flexion_moves_the_head_forward() -> None:
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY["squat"]().build())
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    head0 = data.xpos[model.body("head").id].copy()
    data.qpos[model.jnt_qposadr[model.joint("lumbar_flex").id]] += np.radians(30.0)
    mujoco.mj_kinematics(model, data)
    delta = data.xpos[model.body("head").id] - head0
    assert delta[0] > 0.1 and abs(delta[1]) < 1e-6


# --- initial-pose geometry in the real engine (keyframe) ----------------------
BARBELL_EXERCISES = ["squat", "bench_press", "deadlift", "snatch", "clean_and_jerk"]


def _keyframe(exercise: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY[exercise]().build())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("exercise", BARBELL_EXERCISES)
def test_bar_is_level_and_lateral_at_the_keyframe(exercise: str) -> None:
    model, data = _keyframe(exercise)
    for name in ("barbell_shaft", "barbell_left_sleeve", "barbell_right_sleeve"):
        geom = model.geom(f"{name}_geom").id
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]  # cylinder axis (local Z)
        assert abs(axis[1]) == pytest.approx(1.0, abs=1e-9), name
    left = data.xpos[model.body("barbell_left_sleeve").id]
    right = data.xpos[model.body("barbell_right_sleeve").id]
    assert left[1] > 0.0 > right[1]
    assert left[2] == pytest.approx(right[2]) and left[0] == pytest.approx(right[0])


@pytest.mark.parametrize("exercise", ["bench_press"])
def test_hand_welds_hold_at_the_keyframe(exercise: str) -> None:
    """Bar pose relative to each hand equals the weld relpose (position and angle)."""
    model, data = _keyframe(exercise)
    for side in ("l", "r"):
        eq = model.equality(f"barbell_to_hand_{side}").id
        hand = model.body(f"hand_{side}").id
        bar = model.body("barbell_shaft").id
        rel_pos = data.xmat[hand].reshape(3, 3).T @ (data.xpos[bar] - data.xpos[hand])
        assert rel_pos == pytest.approx(model.eq_data[eq][3:6], abs=1e-6)
        want = np.zeros(4)
        mujoco.mju_mulQuat(want, data.xquat[hand], model.eq_data[eq][6:10])
        assert abs(want @ data.xquat[bar]) == pytest.approx(1.0, abs=1e-6)


def test_bench_lifter_is_supine_with_hands_over_the_shoulders() -> None:
    model, data = _keyframe("bench_press")
    pelvis = data.xmat[model.body("pelvis").id].reshape(3, 3)
    assert pelvis[:, 0] == pytest.approx([0, 0, 1], abs=1e-9)  # chest faces +Z
    assert pelvis[:, 2] == pytest.approx([-1, 0, 0], abs=1e-9)  # head toward -X
    assert pelvis[:, 1] == pytest.approx([0, 1, 0], abs=1e-9)  # left stays +Y
    for side in ("l", "r"):
        shoulder = data.xpos[model.body(f"upper_arm_{side}").id]
        hand = data.xpos[model.body(f"hand_{side}").id]
        assert hand[:2] == pytest.approx(shoulder[:2], abs=1e-6)
        assert hand[2] > shoulder[2] + 0.5  # arms point at the ceiling


def test_bench_has_no_human_equipment_penetration_except_the_grip() -> None:
    model, data = _keyframe("bench_press")
    bench = model.geom("bench_contact").id
    for c in data.contact[: data.ncon]:
        if bench in (c.geom1, c.geom2):
            assert c.dist >= -1e-6


@pytest.mark.parametrize(
    "exercise",
    ["squat", "deadlift", "snatch", "clean_and_jerk", "gait", "sit_to_stand"],
)
def test_standing_feet_rest_on_the_ground(exercise: str) -> None:
    model, data = _keyframe(exercise)
    for side in ("l", "r"):
        geom = model.geom(f"foot_{side}_contact").id
        half = model.geom_size[geom]
        rot = data.geom_xmat[geom].reshape(3, 3)
        lowest = data.geom_xpos[geom][2] - np.abs(rot[2]) @ half
        assert lowest == pytest.approx(0.0, abs=1e-6)  # on the floor, not in it


def test_feet_point_forward_and_width_is_lateral() -> None:
    model, _ = _keyframe("gait")
    half = model.geom_size[model.geom("foot_l_contact").id]
    assert half[0] > half[1]  # length (X) exceeds width (Y)
