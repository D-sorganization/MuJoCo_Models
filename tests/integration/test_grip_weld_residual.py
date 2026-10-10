# SPDX-License-Identifier: MIT
"""Grip-weld residual at the initial pose, in the real MuJoCo engine (issue #408).

Each barbell exercise that grips the bar in both hands welds ``barbell_shaft``
to ``hand_l`` and ``hand_r`` via an equality constraint whose ``relpose``
encodes the grip width.  At the exercise's keyframe, ``mj_forward`` must find
both hand welds already (nearly) satisfied -- otherwise the solver starts by
fighting a constraint that is violated by construction, instead of resolving
the actual dynamics of the lift.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.exercises.bench_press.bench_press_model import (
    build_bench_press_model,
)
from mujoco_models.exercises.clean_and_jerk.clean_and_jerk_model import (
    build_clean_and_jerk_model,
)
from mujoco_models.exercises.deadlift.deadlift_model import build_deadlift_model
from mujoco_models.exercises.snatch.snatch_model import build_snatch_model

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

# Acceptance threshold from issue #408: under 5 mm position residual.
_MAX_RESIDUAL_M = 0.005

GRIP_BUILDERS = {
    "deadlift": build_deadlift_model,
    "snatch": build_snatch_model,
    "clean_and_jerk": build_clean_and_jerk_model,
    "bench_press": build_bench_press_model,
}


def _weld_position_residual(
    model: mujoco.MjModel, data: mujoco.MjData, eq_id: int
) -> float:
    """Translational residual (meters) of a weld equality at the current state.

    MuJoCo's weld ``eq_data`` layout is ``[anchor(3), relpose_pos(3),
    relpose_quat(4), torquescale(1)]``; the constraint is satisfied when
    ``body2`` sits at ``body1`` plus ``relpose_pos`` rotated into ``body1``'s
    frame.
    """
    obj1 = model.eq_obj1id[eq_id]
    obj2 = model.eq_obj2id[eq_id]
    relpose_pos = model.eq_data[eq_id][3:6]
    rot1 = np.zeros(9)
    mujoco.mju_quat2Mat(rot1, data.xquat[obj1])
    expected_pos2 = data.xpos[obj1] + rot1.reshape(3, 3) @ relpose_pos
    return float(np.linalg.norm(data.xpos[obj2] - expected_pos2))


def _grip_weld_residuals(xml: str) -> dict[str, float]:
    """Return ``{weld_name: residual_m}`` for hand-to-barbell welds at key 0."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    residuals: dict[str, float] = {}
    for eq_id in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
        if name is None or "hand" not in name:
            continue
        residuals[name] = _weld_position_residual(model, data, eq_id)
    return residuals


@pytest.mark.parametrize("exercise", sorted(GRIP_BUILDERS))
def test_grip_weld_residual_under_5mm_at_initial_pose(exercise: str) -> None:
    """Both grip welds must be (almost) satisfied at the exercise's start pose.

    Precondition: the exercise builder emits a keyframe and at least one
    hand-to-barbell weld constraint; a model missing either is a test-setup
    bug, not the condition under test, so it fails loudly instead of
    reporting a vacuous pass.
    """
    builder = GRIP_BUILDERS[exercise]
    residuals = _grip_weld_residuals(builder())
    assert residuals, f"{exercise}: no hand-to-barbell weld constraints found"
    for name, residual in residuals.items():
        assert residual < _MAX_RESIDUAL_M, (
            f"{exercise}: {name} residual {residual * 1000:.2f} mm "
            f">= {_MAX_RESIDUAL_M * 1000:.0f} mm at the initial pose"
        )
