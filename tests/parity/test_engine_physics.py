# SPDX-License-Identifier: MIT
"""Real-engine physics regressions for issue #390 (barbell, feet, contacts)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.dynamics import inverse_dynamics, inverse_dynamics_with_contacts
from mujoco_models.exercises.deadlift.deadlift_model import build_deadlift_model
from mujoco_models.exercises.gait.gait_model import build_gait_model
from mujoco_models.exercises.squat.squat_model import build_squat_model
from mujoco_models.shared.body.foot_seating import hold_hinges_rigid

BARBELL_BUILDERS = {
    "squat": build_squat_model,
    "deadlift": build_deadlift_model,
}


def _load(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    return _load_model(mujoco.MjModel.from_xml_string(xml))


def _load_model(model: mujoco.MjModel) -> tuple[mujoco.MjModel, mujoco.MjData]:
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


def _static_hold_torque(xml: str) -> np.ndarray:
    """Hinge torque that carries the keyframe with every hinge held rigid."""
    model = mujoco.MjModel.from_xml_string(xml)
    hold_hinges_rigid(model)
    _, data = _load_model(model)
    return np.asarray(data.qfrc_bias - data.qfrc_constraint - data.qfrc_passive)


def test_keyframe_ctrl_matches_pose() -> None:
    """Servo set points are the pose plus the static-hold feed-forward.

    Gait is seated like the lifts (issue #453), so its keyframe ctrl carries
    the ``tau / kp`` offset that holds the pose against its own weight.
    """
    xml = build_gait_model()
    model, _ = _load(xml)
    assert model.nkey == 1
    ctrl = model.key_ctrl[0]
    assert ctrl.shape == (model.nu,)
    hold = _static_hold_torque(xml)
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        offset = hold[model.jnt_dofadr[jid]] / model.actuator_gainprm[a, 0]
        pose = model.key_qpos[0][model.jnt_qposadr[jid]]
        assert ctrl[a] == pytest.approx(pose + offset)
    assert np.isfinite(model.actuator_gainprm[:, 0]).all()
    assert (model.actuator_forcelimited == 1).all()


def test_inverse_dynamics_static_standing_pose_balances_gravity() -> None:
    """At a static standing pose, inverse dynamics must balance gravity (issue #405).

    The returned generalized force must balance gravity: qfrc_inverse matches
    mj_inverse output to 1e-9, and the root vertical force plus the foot
    contact force is about m*g.  The soles are seated at the contact
    equilibrium (issue #453), so the contacts carry most of the weight and the
    root entry is only the residual they do not explain.
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

    # 2. Root residual plus contact force balances gravity (about m * g)
    total_mass = float(np.sum(model.body_mass))
    g = float(abs(model.opt.gravity[2]))
    expected_root_vertical_f = total_mass * g
    report = inverse_dynamics_with_contacts("gait", q, qvel, qacc)
    assert report.contact_count > 0
    # Root pelvis freejoint vertical translation is DOF index 2 (Z-up)
    supported = report.root_residual[2] + report.root_contact[2]
    assert supported == pytest.approx(expected_root_vertical_f, rel=1e-3)


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
