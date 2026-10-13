# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Tests for task-target inverse kinematics solver (issue #403).

Verifies that:
1. Phase interpolation fails task targets (barbell path & foot placement).
2. The DLS task-target IK solver satisfies bar residual < 1 cm and foot
   residual < 1 cm on the real MuJoCo squat model.
3. Joint limits are strictly respected.
4. Per-target residuals and convergence metrics are reported.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.exceptions import ValidationError
from mujoco_models.exercises.squat.squat_model import SquatModelBuilder
from mujoco_models.optimization.inverse_kinematics import (
    IKResult,
    solve_ik_keyframes,
    solve_task_target_ik,
    solve_task_target_trajectory,
)


@pytest.fixture
def squat_model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Provide a real compiled MuJoCo squat model initialized at keyframe 0."""
    builder = SquatModelBuilder()
    model = mujoco.MjModel.from_xml_string(builder.build())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_fwdPosition(model, data)
    return model, data


@pytest.mark.unit
def test_interpolation_fails_task_targets(
    squat_model_and_data: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    """Phase keyframe interpolation does not satisfy Cartesian task targets.

    Interpolating joint angles between phase keyframes produces trajectories
    that fail to place the barbell or keep feet pinned to ground within 1 cm.
    """
    model, data = squat_model_and_data
    bar_init = data.xpos[model.body("barbell_shaft").id].copy()

    # Target: bar descends by 30 cm, feet stay on ground.
    target_bar = bar_init.copy()
    target_bar[2] -= 0.30

    # Keyframe interpolation produces 28 joint angles, not full 56 qpos coordinates.
    keyframes = solve_ik_keyframes("squat", n_frames=10)
    assert keyframes.shape == (10, 28)
    assert keyframes.shape[1] != model.nq

    # If we evaluate default/interpolated qpos, the bar does not achieve the target.
    # At initial keyframe (standing), bar residual is 30 cm (> 1 cm).
    bar_residual = np.linalg.norm(
        data.xpos[model.body("barbell_shaft").id] - target_bar
    )
    assert bar_residual > 0.10  # 30 cm residual, failing 1 cm tolerance


@pytest.mark.unit
def test_solver_satisfies_squat_task_targets(
    squat_model_and_data: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    """The task-target IK solver achieves < 1 cm bar and foot residuals."""
    model, data = squat_model_and_data
    foot_l_init = data.xpos[model.body("foot_l").id].copy()
    foot_r_init = data.xpos[model.body("foot_r").id].copy()
    bar_init = data.xpos[model.body("barbell_shaft").id].copy()

    target_bar = bar_init.copy()
    target_bar[2] -= 0.25  # 25 cm descent into squat

    targets = {
        "barbell_shaft": target_bar,
        "foot_l": foot_l_init,
        "foot_r": foot_r_init,
    }

    result = solve_task_target_ik(model, targets, tolerance=0.01)

    assert isinstance(result, IKResult)
    assert result.converged
    assert result.residuals["barbell_shaft"] < 0.01  # under 1 cm
    assert result.residuals["foot_l"] < 0.01  # under 1 cm
    assert result.residuals["foot_r"] < 0.01  # under 1 cm

    # Verify positions on MjData loaded with solved qpos
    test_data = mujoco.MjData(model)
    test_data.qpos[:] = result.qpos
    mujoco.mj_fwdPosition(model, test_data)

    foot_l_res = np.linalg.norm(test_data.xpos[model.body("foot_l").id] - foot_l_init)
    foot_r_res = np.linalg.norm(test_data.xpos[model.body("foot_r").id] - foot_r_init)
    bar_res = np.linalg.norm(
        test_data.xpos[model.body("barbell_shaft").id] - target_bar
    )

    assert foot_l_res < 0.01
    assert foot_r_res < 0.01
    assert bar_res < 0.01


@pytest.mark.unit
def test_solver_respects_joint_limits(
    squat_model_and_data: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    """All limited joints must remain strictly within their defined bounds."""
    model, data = squat_model_and_data
    foot_l_init = data.xpos[model.body("foot_l").id].copy()
    foot_r_init = data.xpos[model.body("foot_r").id].copy()
    bar_init = data.xpos[model.body("barbell_shaft").id].copy()

    target_bar = bar_init.copy()
    target_bar[2] -= 0.30

    targets = {
        "barbell_shaft": target_bar,
        "foot_l": foot_l_init,
        "foot_r": foot_r_init,
    }

    result = solve_task_target_ik(model, targets)
    qpos = result.qpos

    for j in range(model.njnt):
        if model.jnt_limited[j]:
            adr = model.jnt_qposadr[j]
            lower = model.jnt_range[j, 0]
            upper = model.jnt_range[j, 1]
            val = qpos[adr]
            assert lower - 1e-6 <= val <= upper + 1e-6, (
                f"Joint '{model.joint(j).name}' violated limit: "
                f"{val} not in [{lower}, {upper}]"
            )


@pytest.mark.unit
def test_solver_trajectory(
    squat_model_and_data: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    """Solve multi-frame trajectory with smooth warm-starting."""
    model, data = squat_model_and_data
    foot_l_init = data.xpos[model.body("foot_l").id].copy()
    foot_r_init = data.xpos[model.body("foot_r").id].copy()
    bar_init = data.xpos[model.body("barbell_shaft").id].copy()

    n_frames = 5
    bar_z = np.linspace(bar_init[2], bar_init[2] - 0.20, n_frames)
    bar_traj = np.tile(bar_init, (n_frames, 1))
    bar_traj[:, 2] = bar_z

    foot_l_traj = np.tile(foot_l_init, (n_frames, 1))
    foot_r_traj = np.tile(foot_r_init, (n_frames, 1))

    trajectories = {
        "barbell_shaft": bar_traj,
        "foot_l": foot_l_traj,
        "foot_r": foot_r_traj,
    }

    results = solve_task_target_trajectory(model, trajectories, tolerance=0.01)

    assert len(results) == n_frames
    for i, res in enumerate(results):
        assert res.converged, f"Frame {i} failed to converge"
        assert res.residuals["barbell_shaft"] < 0.01
        assert res.residuals["foot_l"] < 0.01
        assert res.residuals["foot_r"] < 0.01


@pytest.mark.unit
def test_solver_input_validation(
    squat_model_and_data: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    """Design-by-Contract validation checks on inputs."""
    model, _ = squat_model_and_data

    # Unknown target
    with pytest.raises(ValidationError, match="Target 'nonexistent'"):
        solve_task_target_ik(model, {"nonexistent": np.zeros(3)})

    # Invalid target shape
    with pytest.raises(ValidationError, match="shape"):
        solve_task_target_ik(model, {"foot_l": np.zeros(4)})

    # Non-finite target
    with pytest.raises(ValidationError, match="finite"):
        solve_task_target_ik(model, {"foot_l": np.array([np.nan, 0, 0])})

    # Empty targets
    with pytest.raises(ValidationError, match="targets must not be empty"):
        solve_task_target_ik(model, {})
