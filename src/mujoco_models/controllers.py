# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Controller API and forward dynamics simulation rollout for MuJoCo models.

Provides feedback control with feedforward gravity/bias compensation for
actuated joints, trajectory tracking, and multi-second forward dynamics rollouts.

Design by Contract:
    - Duration must be strictly positive.
    - Model must be valid and contain actuators for control.
    - Rollout tracking error reports Euclidean joint difference (radians).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mujoco_models.dynamics import _resolve_model
from mujoco_models.exceptions import ValidationError
from mujoco_models.shared.contracts.preconditions import (
    require_finite,
    require_positive,
    require_shape,
)

if TYPE_CHECKING:
    from mujoco_models.exercises.base import ExerciseModelBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutResult:
    """Result of forward dynamics simulation rollout.

    Attributes:
        time: Simulation time timestamps, shape ``(n_steps,)``.
        qpos: History of generalized coordinates, shape ``(n_steps, nq)``.
        qvel: History of generalized velocities, shape ``(n_steps, nv)``.
        tracking_errors: History of mean joint tracking error, shape ``(n_steps,)``.
        rms_tracking_error: Root-mean-square joint tracking error across rollout.
        min_pelvis_height: Minimum vertical position of pelvis body (metres).
        collapsed: Whether pelvis fell below collapse threshold.
    """

    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    tracking_errors: np.ndarray
    rms_tracking_error: float
    min_pelvis_height: float
    collapsed: bool


class PDGravityController:
    """PD controller with feedforward gravity and bias force compensation.

    Computes position servo setpoints with static feedforward offset:
    ``ctrl[i] = q_des[i] + (qfrc_bias - qfrc_constraint - qfrc_passive) / kp``.
    """

    def __init__(self, kp_scale: float = 1.0) -> None:
        self.kp_scale = kp_scale

    def compute_ctrl(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_qpos: np.ndarray,
    ) -> np.ndarray:
        """Compute actuator control commands for current state."""
        ctrl = np.zeros(model.nu, dtype=np.float64)
        torque = data.qfrc_bias - data.qfrc_constraint - data.qfrc_passive

        for i in range(model.nu):
            joint_id = model.actuator_trnid[i, 0]
            q_adr = model.jnt_qposadr[joint_id]
            dof_adr = model.jnt_dofadr[joint_id]
            kp = float(model.actuator_gainprm[i, 0]) * self.kp_scale
            tau_ff = float(torque[dof_adr])
            ctrl[i] = target_qpos[q_adr] + tau_ff / max(kp, 1e-6)

        return ctrl


def _validate_trajectory(trajectory: ArrayLike | None, nq: int) -> np.ndarray | None:
    """Validate shape and finite values of trajectory array."""
    if trajectory is None:
        return None
    require_finite(trajectory, "trajectory")
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != nq:
        msg = f"trajectory must have shape (n_frames, {nq}), got {arr.shape}"
        raise ValidationError(msg)
    return arr


def _interpolate_frame(traj_arr: np.ndarray, t: float, duration: float) -> np.ndarray:
    """Interpolate desired qpos from trajectory at timestamp t."""
    frac = t / max(duration, 1e-6)
    frame_float = frac * (len(traj_arr) - 1)
    idx0 = int(np.floor(frame_float))
    idx1 = min(idx0 + 1, len(traj_arr) - 1)
    alpha = frame_float - idx0
    return (1.0 - alpha) * traj_arr[idx0] + alpha * traj_arr[idx1]


def simulate_rollout(
    exercise: str | ExerciseModelBuilder | mujoco.MjModel,
    trajectory: ArrayLike | None = None,
    *,
    duration: float = 5.0,
    dt: float | None = None,
    controller: PDGravityController | None = None,
    q_init: ArrayLike | None = None,
    pelvis_collapse_threshold: float = 0.50,
) -> RolloutResult:
    """Simulate a forward dynamics rollout tracking a trajectory or static hold.

    Args:
        exercise: Target exercise model (name, builder, or compiled MjModel).
        trajectory: Optional trajectory of shape ``(n_frames, nq)``. If None,
            the initial configuration is maintained as a static hold.
        duration: Total rollout simulation duration in seconds. Must be > 0.
        dt: Optional integration timestep. Defaults to ``model.opt.timestep``.
        controller: Optional controller instance. Defaults to
            :class:`PDGravityController`.
        q_init: Optional initial generalized coordinates ``(nq,)``. Defaults to
            keyframe 0 or ``model.qpos0``.
        pelvis_collapse_threshold: Minimum acceptable pelvis height (metres)
            before marking simulation as collapsed.

    Returns:
        :class:`RolloutResult` containing time series trajectories, tracking
        error metrics, and collapse status.

    Raises:
        ValidationError: If duration <= 0 or inputs fail shape/finite checks.
    """
    require_positive(duration, "duration")
    model = _resolve_model(exercise)
    timestep = float(dt) if dt is not None else float(model.opt.timestep)
    require_positive(timestep, "timestep")

    data = mujoco.MjData(model)
    if q_init is not None:
        require_finite(q_init, "q_init")
        require_shape(q_init, (model.nq,), "q_init")
        data.qpos[:] = np.asarray(q_init, dtype=np.float64)
    elif model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        data.qpos[:] = model.qpos0.copy()

    mujoco.mj_fwdPosition(model, data)
    ctrl_obj = controller if controller is not None else PDGravityController()
    traj_arr = _validate_trajectory(trajectory, model.nq)

    n_steps = max(1, int(np.round(duration / timestep)))
    time_series = np.linspace(0.0, duration, n_steps)
    qpos_hist = np.zeros((n_steps, model.nq), dtype=np.float64)
    qvel_hist = np.zeros((n_steps, model.nv), dtype=np.float64)
    tracking_err_hist = np.zeros(n_steps, dtype=np.float64)
    min_pelvis_z = float("inf")

    pelvis_id = 1
    for b in range(model.nbody):
        if model.body(b).name == "pelvis":
            pelvis_id = b
            break

    actuated_q_adrs = [
        model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)
    ]

    for step in range(n_steps):
        t = time_series[step]
        if traj_arr is not None:
            q_des = _interpolate_frame(traj_arr, t, duration)
        else:
            q_des = data.qpos.copy() if step == 0 else qpos_hist[0]

        data.ctrl[:] = ctrl_obj.compute_ctrl(model, data, q_des)
        mujoco.mj_step(model, data)

        qpos_hist[step] = data.qpos.copy()
        qvel_hist[step] = data.qvel.copy()

        pelvis_z = float(data.xpos[pelvis_id, 2])
        if pelvis_z < min_pelvis_z:
            min_pelvis_z = pelvis_z

        if actuated_q_adrs:
            errs = np.abs(data.qpos[actuated_q_adrs] - q_des[actuated_q_adrs])
            tracking_err_hist[step] = float(np.mean(errs))

    rms_err = float(np.sqrt(np.mean(np.square(tracking_err_hist))))
    collapsed = min_pelvis_z < pelvis_collapse_threshold

    return RolloutResult(
        time=time_series,
        qpos=qpos_hist,
        qvel=qvel_hist,
        tracking_errors=tracking_err_hist,
        rms_tracking_error=rms_err,
        min_pelvis_height=min_pelvis_z,
        collapsed=collapsed,
    )


__all__ = ["PDGravityController", "RolloutResult", "simulate_rollout"]
