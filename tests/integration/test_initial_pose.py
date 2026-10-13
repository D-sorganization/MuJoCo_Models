# SPDX-License-Identifier: MIT
"""Integration tests for applying real exercise initial poses (#407)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.kinematics import initial_state
from mujoco_models.model_pack import list_exercises
from mujoco_models.shared.parity._canonical import conformance, spec
from mujoco_models.shared.parity.standard import STANDARD

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

EXERCISES = list_exercises()
LIFTS = ("squat", "deadlift", "bench_press", "snatch", "clean_and_jerk")
_ANGLE_TOL = float(STANDARD["tolerances"]["limit_abs_rad"])
_FLOOR_BAR_Z = 0.225  # Men's Olympic plate radius (0.450 m / 2)


def test_deadlift_bar_and_joint_angles_at_initial_state() -> None:
    """Deadlift bar starts on floor within 5 mm and knee/hip angles match start pose."""
    model, data = initial_state("deadlift")

    # Bar height at t=0 is on the floor (plate radius above z=0) within 5 mm
    bar_id = model.body("barbell_shaft").id
    bar_z = float(data.xpos[bar_id][2])
    assert abs(bar_z - _FLOOR_BAR_Z) <= 0.005, (
        f"Deadlift bar height {bar_z:.4f} m not within 5 mm of floor level {_FLOOR_BAR_Z} m"
    )

    # Knee and hip angles match declared start pose
    want_coords = spec.pose_coordinates(STANDARD, "deadlift", "start")
    for jname in ("knee_l_flex", "hip_l_flex"):
        jid = model.joint(jname).id
        adr = model.jnt_qposadr[jid]
        got_angle = float(data.qpos[adr]) - float(model.qpos0[adr])
        expected_angle = want_coords[jname]
        assert got_angle == pytest.approx(expected_angle, abs=_ANGLE_TOL), (
            f"Deadlift {jname}: expected {expected_angle}, got {got_angle}"
        )


@pytest.mark.parametrize("exercise", EXERCISES)
def test_all_exercises_initial_state_matches_declared_start_pose(exercise: str) -> None:
    """After initial_state, qpos equals the declared initial pose for every exercise."""
    model, data = initial_state(exercise)

    if exercise in LIFTS:
        want = dict.fromkeys(conformance.expected_coordinates(STANDARD), 0.0)
        want.update(spec.pose_coordinates(STANDARD, exercise, "start"))
        for name, expected_val in want.items():
            jid = model.joint(name).id
            adr = model.jnt_qposadr[jid]
            got_val = float(data.qpos[adr]) - float(model.qpos0[adr])
            assert got_val == pytest.approx(expected_val, abs=_ANGLE_TOL), (
                f"{exercise} {name}: expected {expected_val}, got {got_val}"
            )
    else:
        # Bodyweight movements (gait, sit_to_stand) default to keyframe 0 pose
        assert model.nkey > 0
        np.testing.assert_allclose(data.qpos, model.key_qpos[0], atol=1e-5)


@pytest.mark.parametrize("exercise", EXERCISES)
def test_no_joint_limit_or_equality_violations_at_initial_state(exercise: str) -> None:
    """No joint-limit or equality-constraint violations at t=0."""
    model, data = initial_state(exercise)

    # 1. Joint limits
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            adr = model.jnt_qposadr[j]
            val = float(data.qpos[adr])
            lower, upper = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
            assert lower - 1e-4 <= val <= upper + 1e-4, (
                f"{exercise} {model.joint(j).name} ({val:.4f}) outside range [{lower:.4f}, {upper:.4f}]"
            )

    # 2. Equality constraints (weld, connect)
    # In MuJoCo, data.efc_pos contains constraint violations; at initial state with mj_forward,
    # equality violations must be negligible (< 5 mm / 0.01 rad).
    if model.neq > 0 and data.nefc > 0:
        eq_residuals = [
            abs(float(data.efc_pos[i]))
            for i in range(data.nefc)
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        ]
        if eq_residuals:
            max_eq_violation = max(eq_residuals)
            assert max_eq_violation < 0.01, (
                f"{exercise} equality constraint violation {max_eq_violation:.5f} exceeds tolerance"
            )
