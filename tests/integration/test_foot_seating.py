# SPDX-License-Identifier: MIT
"""Feet are seated at the contact-law equilibrium penetration (issue #447).

The soles used to be built exactly on the floor, so a model that starts there
is out of static equilibrium (the soft contact carries almost no force at zero
depth) and settles at t=0.  The builders now seat the soles ``delta`` into the
floor, where ``delta`` solves the solver's own contact law: the depth at which
the foot-floor constraint force supports the weight, i.e. the root's vertical
acceleration at the keyframe is zero with the joints held rigid (the crouched
spec starts are not in servo equilibrium; their sag is MuJoCo_Models#460).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.exercises.clean_and_jerk.clean_and_jerk_model import (
    build_clean_and_jerk_model,
)
from mujoco_models.exercises.deadlift.deadlift_model import build_deadlift_model
from mujoco_models.exercises.gait.gait_model import (
    GaitModelBuilder,
    build_gait_model,
)
from mujoco_models.exercises.sit_to_stand.sit_to_stand_model import (
    build_sit_to_stand_model,
)
from mujoco_models.exercises.snatch.snatch_model import build_snatch_model
from mujoco_models.exercises.squat.squat_model import build_squat_model
from mujoco_models.shared.body.foot_seating import (
    ROOT_VERTICAL_DOF,
    equilibrium_foot_penetration,
    hold_hinges_rigid,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

BUILDERS = {
    "squat": build_squat_model,
    "deadlift": build_deadlift_model,
    "snatch": build_snatch_model,
    "clean_and_jerk": build_clean_and_jerk_model,
    "gait": build_gait_model,
    "sit_to_stand": build_sit_to_stand_model,
}
_DRIFT_STEPS = 20
# Root vertical drift over the first 20 ms.  Flush soles drift 0.36-0.41 mm.
_MAX_DRIFT_M = 1e-4
_RESIDUAL_ACC = 1e-3  # m/s^2, root vertical acceleration at the keyframe


def _start(exercise: str):
    model = mujoco.MjModel.from_xml_string(BUILDERS[exercise]())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def _sole_depths(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    depths = []
    for side in "lr":
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{side}_contact")
        rot = data.geom_xmat[gid].reshape(3, 3)
        low = data.geom_xpos[gid][2] - np.abs(rot[2]) @ model.geom_size[gid]
        depths.append(-float(low))
    return depths


@pytest.mark.parametrize("exercise", sorted(BUILDERS))
def test_initial_penetration_equals_derived_equilibrium(exercise: str) -> None:
    model, data = _start(exercise)
    expected = equilibrium_foot_penetration(BUILDERS[exercise]())
    assert expected > 1e-4, "equilibrium depth must be a real, positive penetration"
    for depth in _sole_depths(model, data):
        assert depth == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("exercise", sorted(BUILDERS))
def test_start_pose_is_in_vertical_equilibrium(exercise: str) -> None:
    """With the joints held rigid the seated contact carries lifter and bar."""
    model, data = _start(exercise)
    hold_hinges_rigid(model)
    mujoco.mj_forward(model, data)
    assert abs(data.qacc[ROOT_VERTICAL_DOF]) < _RESIDUAL_ACC


@pytest.mark.parametrize("exercise", sorted(BUILDERS))
def test_root_does_not_settle_at_start(exercise: str) -> None:
    model, data = _start(exercise)
    z0 = float(data.qpos[ROOT_VERTICAL_DOF])
    for _ in range(_DRIFT_STEPS):
        mujoco.mj_step(model, data)
        assert abs(float(data.qpos[ROOT_VERTICAL_DOF]) - z0) < _MAX_DRIFT_M


def test_equilibrium_is_independent_of_current_seating() -> None:
    """Deriving on an unseated build gives the depth the seated build uses."""
    builder = GaitModelBuilder()
    builder.seat_feet_at_equilibrium = False
    unseated = builder.build()
    builder.seat_feet_at_equilibrium = True
    seated = builder.build()
    assert equilibrium_foot_penetration(unseated) == pytest.approx(
        equilibrium_foot_penetration(seated), abs=1e-8
    )


def test_equilibrium_rejects_model_without_keyframe() -> None:
    xml = (
        '<mujoco><worldbody><geom name="foot_l_contact" type="box" '
        'size="1 1 1"/></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="keyframe"):
        equilibrium_foot_penetration(xml)
