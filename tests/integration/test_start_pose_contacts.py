# SPDX-License-Identifier: MIT
"""Start-pose contact audit in the real MuJoCo engine (issue #427).

The barbell is held by the grip weld, so bar-versus-lifter and bar-versus-bar
contacts are non-physical constraint fights: they inject up to hundreds of kN
and hide the true ground reaction.  At the keyframe of every barbell lift the
only contacts must be foot/equipment versus floor (and lifter versus bench).
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
from mujoco_models.exercises.squat.squat_model import build_squat_model

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

BUILDERS = {
    "squat": build_squat_model,
    "deadlift": build_deadlift_model,
    "snatch": build_snatch_model,
    "clean_and_jerk": build_clean_and_jerk_model,
    "bench_press": build_bench_press_model,
}
STANDING = ("squat", "deadlift", "snatch", "clean_and_jerk")
_SETTLE_STEPS = 1000
_GRF_TOLERANCE = 0.02  # fraction of (lifter + bar) weight, issue #427
_MAX_NON_GROUND_N = 1.0
_GRAVITY = 9.80665


def _geom_body_name(model: mujoco.MjModel, geom_id: int) -> str:
    body_id = model.geom_bodyid[geom_id]
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _start_state(exercise: str, steps: int = 0):
    model = mujoco.MjModel.from_xml_string(BUILDERS[exercise]())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return model, data


def _contacts(model: mujoco.MjModel, data: mujoco.MjData):
    """Yield (geom_a, geom_b, normal_force) for every active contact."""
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        mujoco.mj_contactForce(model, data, i, wrench)
        yield contact.geom1, contact.geom2, float(wrench[0])


def _is_bar(model: mujoco.MjModel, geom_id: int) -> bool:
    return _geom_body_name(model, geom_id).startswith("barbell")


def _lifted_weight(model: mujoco.MjModel) -> float:
    """Weight (N) of every body that is not welded to the world (lifter + bar)."""
    moving = model.body_weldid != 0
    return float(model.body_mass[moving].sum() * _GRAVITY)


def _ground_normal_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    return sum(f for a, b, f in _contacts(model, data) if ground in (a, b))


@pytest.mark.parametrize("exercise", sorted(BUILDERS))
def test_no_bar_body_or_intra_bar_contacts_at_start(exercise: str) -> None:
    model, data = _start_state(exercise)
    offenders = [
        (_geom_name(model, a), _geom_name(model, b), round(f, 1))
        for a, b, f in _contacts(model, data)
        if _is_bar(model, a)
        and _is_bar(model, b)
        or (
            _is_bar(model, a) != _is_bar(model, b)
            and "ground" not in (_geom_name(model, a), _geom_name(model, b))
        )
    ]
    assert not offenders, f"{exercise}: bar contacts at start pose: {offenders}"


@pytest.mark.parametrize("exercise", sorted(BUILDERS))
def test_non_ground_forces_small_at_start(exercise: str) -> None:
    """Issue #427 acceptance: non-ground, non-bench normal force below 1 N."""
    model, data = _start_state(exercise)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    total = sum(
        abs(f)
        for a, b, f in _contacts(model, data)
        if ground not in (a, b)
        and "bench" not in _geom_body_name(model, a) + _geom_body_name(model, b)
    )
    assert total < _MAX_NON_GROUND_N, f"{exercise}: {total:.1f} N non-ground force"


@pytest.mark.parametrize("exercise", STANDING)
def test_feet_touch_floor_at_start(exercise: str) -> None:
    model, data = _start_state(exercise)
    foot_floor = [
        (a, b)
        for a, b, _ in _contacts(model, data)
        if {_geom_name(model, a), _geom_name(model, b)} & {"foot_l_contact"}
        and "ground" in (_geom_name(model, a), _geom_name(model, b))
    ]
    assert foot_floor, f"{exercise}: no foot-floor contact at the start pose"


_FOOT_FLOAT = (
    "Soles are built exactly on the floor (sole_z 0 to 9e-17 m), so a 0.5 mm "
    "float removes every foot-floor contact. A 1 mm foot-geom margin fixes it "
    "but changes the qfrc_inverse root force that "
    "tests/parity/test_engine_physics.py::"
    "test_inverse_dynamics_static_standing_pose_balances_gravity pins to m*g "
    "(784.5 N expected), so it needs a decision on that test, not a silent edit."
)


@pytest.mark.xfail(strict=True, reason=_FOOT_FLOAT)
@pytest.mark.parametrize("exercise", STANDING)
def test_foot_contact_survives_sub_millimetre_float(exercise: str) -> None:
    """The soles are built exactly on the floor, so contact must not hinge on ULPs.

    A 0.5 mm lift (or a body of a different height, whose pelvis height rounds
    the sole to +1e-17 m) must still register foot-floor contacts; the foot
    contact geoms carry a 1 mm margin for that (issue #427, harness pose at
    sole_z = 9e-17 m reported 0 ground contacts).
    """
    model = mujoco.MjModel.from_xml_string(BUILDERS[exercise]())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] += 5e-4
    mujoco.mj_forward(model, data)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert any(ground in (a, b) for a, b, _ in _contacts(model, data))


_SQUAT_SAG = (
    "Squat keyframe is not in servo equilibrium: the force-limited joint "
    "servos let the lifter sag ~3 cm under bar plus body weight, so the "
    "1 s mean GRF is 0.938 of weight (6.2 % low, measured). Pose issue, not "
    "a contact defect; tracked by the initial-pose work (MuJoCo_Models#407)."
)


def _mean_grf_over_settle(exercise: str) -> tuple[float, float]:
    """Mean vertical GRF and lifted weight over the first ``_SETTLE_STEPS`` steps.

    A single ``mj_forward`` GRF is not a static value: feet start exactly on
    the soft floor, so the instantaneous normal force is only a fraction of
    weight (0.23 to 0.34 measured) until the contact settles.  The mean over
    the settle window equals weight when the pose is in equilibrium.
    """
    model, data = _start_state(exercise)
    samples = []
    for _ in range(_SETTLE_STEPS):
        mujoco.mj_step(model, data)
        samples.append(_ground_normal_force(model, data))
    return float(np.mean(samples)), _lifted_weight(model)


@pytest.mark.parametrize(
    "exercise",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(strict=True, reason=_SQUAT_SAG)
            if name == "squat"
            else (),
        )
        for name in STANDING
    ],
)
def test_static_grf_matches_weight(exercise: str) -> None:
    grf, weight = _mean_grf_over_settle(exercise)
    assert grf == pytest.approx(weight, rel=_GRF_TOLERANCE), (
        f"{exercise}: GRF {grf:.1f} N vs weight {weight:.1f} N"
    )
