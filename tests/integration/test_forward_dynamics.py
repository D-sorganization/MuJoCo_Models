# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Tests for forward dynamics controller and rollout API (issue #404).

Verifies that:
1. Without a controller, the 140 kg barbell squat collapses within 1.5 s.
2. The public PD + gravity compensation controller sustains the 5 s squat
   rollout without collapse (pelvis height stays well above threshold).
3. RMS joint tracking error remains bounded below 0.01 rad.
4. Trajectory tracking in forward dynamics correctly follows moving joint
   trajectories over 5 seconds.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.controllers import (
    PDGravityController,
    RolloutResult,
    simulate_rollout,
)
from mujoco_models.exceptions import ValidationError
from mujoco_models.exercises.squat.squat_model import SquatModelBuilder


@pytest.fixture
def squat_model() -> mujoco.MjModel:
    """Compiled MuJoCo squat model."""
    return mujoco.MjModel.from_xml_string(SquatModelBuilder().build())


@pytest.mark.unit
def test_uncontrolled_squat_collapses(squat_model: mujoco.MjModel) -> None:
    """Without active control, the squat collapses under the 140 kg barbell."""
    data = mujoco.MjData(squat_model)
    mujoco.mj_resetDataKeyframe(squat_model, data, 0)
    mujoco.mj_fwdPosition(squat_model, data)
    dt = squat_model.opt.timestep

    pelvis_id = squat_model.body("pelvis").id
    initial_z = data.xpos[pelvis_id, 2]
    assert initial_z > 0.95

    # Zero out controls (uncontrolled simulation)
    data.ctrl[:] = 0.0
    for _ in range(int(2.0 / dt)):
        mujoco.mj_step(squat_model, data)

    final_z = data.xpos[pelvis_id, 2]
    # Pelvis collapses to near ground (< 0.3 m)
    assert final_z < 0.30


@pytest.mark.unit
def test_controlled_squat_5s_rollout_does_not_collapse(
    squat_model: mujoco.MjModel,
) -> None:
    """The controller maintains 5 s squat hold with RMS error < 0.01 rad."""
    result = simulate_rollout(
        squat_model,
        duration=5.0,
        controller=PDGravityController(kp_scale=1.0),
        pelvis_collapse_threshold=0.60,
    )

    assert isinstance(result, RolloutResult)
    assert not result.collapsed
    assert result.min_pelvis_height > 0.90  # Started at ~1.005 m
    assert result.rms_tracking_error < 0.01  # Documented threshold < 10 mrad
    assert len(result.time) == len(result.qpos)
    assert len(result.time) == len(result.tracking_errors)


@pytest.mark.unit
def test_controller_tracks_joint_trajectory(squat_model: mujoco.MjModel) -> None:
    """The controller accurately tracks a moving joint trajectory."""
    data = mujoco.MjData(squat_model)
    mujoco.mj_resetDataKeyframe(squat_model, data, 0)
    mujoco.mj_fwdPosition(squat_model, data)

    n_frames = 51
    q_start = data.qpos.copy()
    trajectory = np.tile(q_start, (n_frames, 1))

    # Add gentle knee, hip, and ankle flexion/extension trajectory
    knee_l = squat_model.jnt_qposadr[squat_model.joint("knee_l_flex").id]
    knee_r = squat_model.jnt_qposadr[squat_model.joint("knee_r_flex").id]
    hip_l = squat_model.jnt_qposadr[squat_model.joint("hip_l_flex").id]
    hip_r = squat_model.jnt_qposadr[squat_model.joint("hip_r_flex").id]
    ankle_l = squat_model.jnt_qposadr[squat_model.joint("ankle_l_flex").id]
    ankle_r = squat_model.jnt_qposadr[squat_model.joint("ankle_r_flex").id]

    for f in range(n_frames):
        delta = 0.05 * np.sin(np.pi * f / (n_frames - 1))
        trajectory[f, knee_l] -= delta
        trajectory[f, knee_r] -= delta
        trajectory[f, hip_l] += delta
        trajectory[f, hip_r] += delta
        trajectory[f, ankle_l] += delta * 0.5
        trajectory[f, ankle_r] += delta * 0.5

    result = simulate_rollout(
        squat_model,
        trajectory=trajectory,
        duration=5.0,
        pelvis_collapse_threshold=0.60,
    )

    assert not result.collapsed
    assert result.min_pelvis_height > 0.85
    assert result.rms_tracking_error < 0.01  # Documented threshold < 10 mrad


@pytest.mark.unit
def test_rollout_input_validation(squat_model: mujoco.MjModel) -> None:
    """Input validation and Design-by-Contract errors."""
    with pytest.raises(ValidationError, match="duration must be positive"):
        simulate_rollout(squat_model, duration=-1.0)

    with pytest.raises(ValidationError, match="Unknown exercise"):
        simulate_rollout("nonexistent_exercise")

    bad_traj = np.zeros((5, 10))  # wrong nq
    with pytest.raises(ValidationError, match="trajectory"):
        simulate_rollout(squat_model, trajectory=bad_traj)
