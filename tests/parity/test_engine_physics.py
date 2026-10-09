# SPDX-License-Identifier: MIT
"""Real-engine physics regressions for issue #390 (barbell, feet, contacts)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.dynamics import inverse_dynamics
from mujoco_models.exercises.deadlift.deadlift_model import build_deadlift_model
from mujoco_models.exercises.gait.gait_model import build_gait_model
from mujoco_models.exercises.squat.squat_model import build_squat_model

BARBELL_BUILDERS = {
    "squat": build_squat_model,
    "deadlift": build_deadlift_model,
}


def _load(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("name", sorted(BARBELL_BUILDERS))
def test_barbell_has_freejoint_not_world_pinned(name: str) -> None:
    """The barbell must couple to the lifter, not be rigidly fixed to the world."""
    model, _ = _load(BARBELL_BUILDERS[name]())
    bar = model.body("barbell_shaft")
    assert bar.dofnum[0] == 6, "barbell_shaft needs a freejoint (6 DOF)"
    # 34 body DOF (6 root + 28 hinges) + shaft + two sleeves, each a freejoint
    # coupled to the shaft / lifter by weld constraints.
    assert model.nv == 34 + 3 * 6


def test_squat_impulse_moves_pelvis() -> None:
    """A force on the pelvis must displace it (it is not welded to the world)."""
    model, data = _load(build_squat_model())
    pelvis = model.body("pelvis").id
    z0 = data.xpos[pelvis].copy()
    for _ in range(100):
        data.xfrc_applied[pelvis, 2] = 20000.0
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.xpos[pelvis] - z0) > 0.2


@pytest.mark.parametrize("name", ["gait"])
def test_foot_box_bottom_on_ground_at_keyframe(name: str) -> None:
    model, data = _load(build_gait_model())
    for side in ("l", "r"):
        gid = model.geom(f"foot_{side}_contact").id
        bottom = data.geom_xpos[gid][2] - model.geom_size[gid][2]
        assert bottom == pytest.approx(0.0, abs=1e-3)


def test_only_foot_box_collides_with_floor() -> None:
    model, _ = _load(build_gait_model())
    ground = model.geom("ground").id
    colliders = set()
    for g in range(model.ngeom):
        if g == ground:
            continue
        hit = (model.geom_contype[g] & model.geom_conaffinity[ground]) or (
            model.geom_contype[ground] & model.geom_conaffinity[g]
        )
        if hit:
            colliders.add(model.geom(g).name)
    assert colliders == {"foot_l_contact", "foot_r_contact"}


def test_no_self_contacts_between_body_segments() -> None:
    model, data = _load(build_gait_model())
    mujoco.mj_forward(model, data)
    for c in data.contact[: data.ncon]:
        names = {model.geom(c.geom1).name, model.geom(c.geom2).name}
        assert "ground" in names, f"self contact {names}"
    assert data.ncon > 0  # feet are in contact with the ground
    assert np.abs(data.efc_force).max() < 1e4


def test_keyframe_ctrl_matches_pose() -> None:
    model, _ = _load(build_gait_model())
    assert model.nkey == 1
    ctrl = model.key_ctrl[0]
    assert ctrl.shape == (model.nu,)
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        assert ctrl[a] == pytest.approx(model.key_qpos[0][model.jnt_qposadr[jid]])
    assert np.isfinite(model.actuator_gainprm[:, 0]).all()
    assert (model.actuator_forcelimited == 1).all()


def test_inverse_dynamics_static_standing_pose_balances_gravity() -> None:
    """At a static standing pose, inverse dynamics must balance gravity (issue #405).

    The returned generalized force must balance gravity: qfrc_inverse matches
    mj_inverse output to 1e-9, and root vertical force is about m*g.
    """
    model, data = _load(build_gait_model())
    q = data.qpos.copy()
    qvel = np.zeros(model.nv)
    qacc = np.zeros(model.nv)

    torques = inverse_dynamics("gait", q, qvel, qacc)

    # 1. Compare with direct mj_inverse call to 1e-9
    d_direct = mujoco.MjData(model)
    d_direct.qpos[:] = q
    d_direct.qvel[:] = qvel
    d_direct.qacc[:] = qacc
    mujoco.mj_inverse(model, d_direct)
    np.testing.assert_allclose(torques, d_direct.qfrc_inverse, atol=1e-9)

    # 2. Root vertical force balances gravity (about m * g)
    total_mass = float(np.sum(model.body_mass))
    g = float(abs(model.opt.gravity[2]))
    expected_root_vertical_f = total_mass * g
    # Root pelvis freejoint vertical translation is DOF index 2 (Z-up)
    assert torques[2] == pytest.approx(expected_root_vertical_f, rel=1e-3)


def test_inverse_dynamics_dynamic_motion_matches_mj_inverse() -> None:
    """Non-zero velocity and acceleration match mj_inverse output to 1e-9."""
    model, data = _load(build_gait_model())
    rng = np.random.default_rng(2026)
    q = data.qpos.copy()
    qvel = rng.standard_normal(model.nv) * 0.5
    qacc = rng.standard_normal(model.nv) * 2.0

    torques = inverse_dynamics("gait", q, qvel, qacc)

    d_direct = mujoco.MjData(model)
    d_direct.qpos[:] = q
    d_direct.qvel[:] = qvel
    d_direct.qacc[:] = qacc
    mujoco.mj_inverse(model, d_direct)
    np.testing.assert_allclose(torques, d_direct.qfrc_inverse, atol=1e-9)
