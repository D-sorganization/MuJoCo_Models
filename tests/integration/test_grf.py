# SPDX-License-Identifier: MIT
"""Integration tests for standing GRF and center of pressure (CoP) API (#406)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.grf import (
    StandingGRF,
    compute_grf,
    standing_grf,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

GRAVITY_MPS2 = 9.80665


def _settle_model(
    exercise: str, settle_s: float = 0.3
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY[exercise]().build())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
    n_steps = round(settle_s / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    return model, data


def _foot_bounds(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> tuple[float, float, float, float]:
    """Return (min_x, max_x, min_y, max_y) for foot contact geom."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{side}_contact")
    pos = data.geom_xpos[gid]
    rot = data.geom_xmat[gid].reshape(3, 3)
    half = np.abs(rot) @ model.geom_size[gid]
    return (
        float(pos[0] - half[0]),
        float(pos[0] + half[0]),
        float(pos[1] - half[1]),
        float(pos[1] + half[1]),
    )


@pytest.mark.parametrize("exercise", ["gait", "squat"])
def test_standing_vertical_grf_matches_weight_and_cop_in_polygon(
    exercise: str,
) -> None:
    """Standing model at rest has total Fz ~ m*g (<2%) and CoP inside support polygon."""
    model, data = _settle_model(exercise)
    grf: StandingGRF = compute_grf(model, data)

    # Compute expected static weight (all bodies except world).
    total_mass = sum(
        float(model.body_mass[b])
        for b in range(model.nbody)
        if model.body(b).name != "world"
    )
    expected_weight = total_mass * GRAVITY_MPS2

    fz = float(grf.total.force_n[2])
    rel_error = abs(fz - expected_weight) / expected_weight
    assert rel_error < 0.02, (
        f"{exercise}: vertical GRF {fz:.2f} N diverged from weight "
        f"{expected_weight:.2f} N by {rel_error * 100:.2f}% (tol 2%)"
    )

    # CoP must be computed and finite
    assert grf.cop_m is not None
    cop = grf.cop_m
    assert np.all(np.isfinite(cop))
    assert grf.total.in_contact is True

    # Support polygon bounding box across both feet
    l_min_x, l_max_x, l_min_y, l_max_y = _foot_bounds(model, data, "l")
    r_min_x, r_max_x, r_min_y, r_max_y = _foot_bounds(model, data, "r")
    poly_min_x = min(l_min_x, r_min_x)
    poly_max_x = max(l_max_x, r_max_x)
    poly_min_y = min(l_min_y, r_min_y)
    poly_max_y = max(l_max_y, r_max_y)

    assert poly_min_x <= cop[0] <= poly_max_x, (
        f"CoP x={cop[0]:.4f} outside support polygon [{poly_min_x:.4f}, {poly_max_x:.4f}]"
    )
    assert poly_min_y <= cop[1] <= poly_max_y, (
        f"CoP y={cop[1]:.4f} outside support polygon [{poly_min_y:.4f}, {poly_max_y:.4f}]"
    )

    # Check each individual foot
    for side, (min_x, max_x, min_y, max_y) in [
        ("l", (l_min_x, l_max_x, l_min_y, l_max_y)),
        ("r", (r_min_x, r_max_x, r_min_y, r_max_y)),
    ]:
        foot_key = f"foot_{side}"
        assert foot_key in grf.per_foot
        foot = grf.per_foot[foot_key]
        assert foot.in_contact is True
        assert foot.force_n[2] > 0.0
        assert foot.cop_m is not None
        assert min_x <= foot.cop_m[0] <= max_x
        assert min_y <= foot.cop_m[1] <= max_y


def test_zero_contact_returns_defined_result() -> None:
    """Mid-air / zero-contact state returns 0 force and None CoP without NaN."""
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY["gait"]().build())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # Lift the pelvis 2 meters up so feet are far from the floor
    pelvis_jid = model.joint("ground_pelvis").id
    qpos_adr = model.jnt_qposadr[pelvis_jid]
    data.qpos[qpos_adr + 2] += 2.0
    mujoco.mj_forward(model, data)
    assert data.ncon == 0

    grf = compute_grf(model, data)
    assert np.all(grf.total.force_n == 0.0)
    assert np.all(grf.total.moment_about_origin_nm == 0.0)
    assert grf.cop_m is None
    assert grf.total.cop_m is None
    assert grf.total.in_contact is False

    for side in ("foot_l", "foot_r"):
        assert side in grf.per_foot
        f = grf.per_foot[side]
        assert np.all(f.force_n == 0.0)
        assert f.cop_m is None
        assert f.in_contact is False


def test_standing_grf_convenience_function() -> None:
    """standing_grf settles keyframe and returns a StandingGRF."""
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY["gait"]().build())
    grf = standing_grf(model, settle_s=0.3)
    assert isinstance(grf, StandingGRF)
    assert grf.total.in_contact is True
    assert grf.cop_m is not None
    assert grf.total.force_n[2] > 500.0
