# SPDX-License-Identifier: MIT
"""Trajectory optimization for barbell exercise models using MuJoCo.

Implements cost functions and trajectory optimization configuration for
planning biomechanically optimal barbell exercise movements.

Design by Contract:
    - All array inputs are validated for shape and finite values.
    - Config values are validated for physical plausibility.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mujoco_models.exceptions import ValidationError

from .costs import compute_balance_cost, compute_bar_path_cost
from .exercise_objectives import ExerciseObjective
from .inverse_kinematics import _phase_to_array

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectoryConfig:
    """Configuration for trajectory optimization.

    Attributes:
        n_timesteps: Number of discrete time steps in trajectory.
        dt: Time step duration in seconds.
        max_iterations: Maximum optimizer iterations before stopping.
        convergence_tol: Cost change threshold for convergence.
        control_weight: Penalty weight on joint torque magnitudes.
        state_weight: Weight for tracking target trajectory.
        terminal_weight: Weight for reaching the final target pose.
        balance_weight: Weight for keeping CoM over base of support.
    """

    n_timesteps: int = 100
    dt: float = 0.01
    max_iterations: int = 200
    convergence_tol: float = 1e-4
    control_weight: float = 1e-3
    state_weight: float = 1.0
    terminal_weight: float = 10.0
    balance_weight: float = 5.0

    def __post_init__(self) -> None:
        """Validate configuration values are physically plausible."""
        if self.n_timesteps < 1:
            msg = f"n_timesteps must be >= 1, got {self.n_timesteps}"
            raise ValidationError(msg)
        if self.dt <= 0.0:
            msg = f"dt must be positive, got {self.dt}"
            raise ValidationError(msg)
        if self.max_iterations < 1:
            msg = f"max_iterations must be >= 1, got {self.max_iterations}"
            raise ValidationError(msg)
        if self.convergence_tol <= 0.0:
            msg = f"convergence_tol must be positive, got {self.convergence_tol}"
            raise ValidationError(msg)
        _validate_non_negative_weight("control_weight", self.control_weight)
        _validate_non_negative_weight("state_weight", self.state_weight)
        _validate_non_negative_weight("terminal_weight", self.terminal_weight)
        _validate_non_negative_weight("balance_weight", self.balance_weight)

    @property
    def total_time(self) -> float:
        """Total trajectory duration in seconds."""
        return self.n_timesteps * self.dt


def _validate_non_negative_weight(name: str, value: float) -> None:
    """Validate that a weight is non-negative."""
    if value < 0.0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValidationError(msg)


@dataclass
class TrajectoryResult:
    """Result of trajectory optimization.

    Attributes:
        joint_positions: Joint angles at each timestep,
            shape (n_timesteps, n_joints).
        joint_velocities: Joint angular velocities,
            shape (n_timesteps, n_joints).
        joint_torques: Applied joint torques,
            shape (n_timesteps, n_actuators).
        time: Time values for each step, shape (n_timesteps,).
        cost: Final optimization cost value.
        converged: Whether the optimizer converged.
        iterations: Number of iterations performed.
    """

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    time: np.ndarray
    cost: float
    converged: bool
    iterations: int

    def __post_init__(self) -> None:
        """Validate result arrays have consistent shapes."""
        n_t = len(self.time)
        if self.joint_positions.shape[0] != n_t:
            msg = (
                f"joint_positions has {self.joint_positions.shape[0]}"
                f" timesteps, expected {n_t}"
            )
            raise ValidationError(msg)
        if self.joint_velocities.shape[0] != n_t:
            msg = (
                f"joint_velocities has "
                f"{self.joint_velocities.shape[0]}"
                f" timesteps, expected {n_t}"
            )
            raise ValidationError(msg)
        if self.joint_torques.shape[0] != n_t:
            msg = (
                f"joint_torques has {self.joint_torques.shape[0]}"
                f" timesteps, expected {n_t}"
            )
            raise ValidationError(msg)


def interpolate_phases(
    objective: ExerciseObjective,
    n_frames: int = 50,
) -> np.ndarray:
    """Interpolate between phases to generate keyframes.

    Linearly interpolates joint angles between successive phases
    of an exercise objective, producing a dense keyframe array.

    Args:
        objective: The exercise objective whose phases will be
            interpolated.
        n_frames: Number of output frames (rows).

    Returns:
        Array of shape ``(n_frames, n_joints)`` with linearly
        interpolated joint angles.  Joint ordering is alphabetical.
    Raises:
        ValidationError: If n_frames < 2 or objective has no phases.
    """
    if n_frames < 2:
        raise ValidationError(f"n_frames must be >= 2, got {n_frames}")
    if not objective.phases:
        raise ValidationError("objective must have at least one phase")
    joint_names = objective.joint_names
    n_joints = len(joint_names)

    fractions = np.linspace(0.0, 1.0, n_frames)
    keyframes = np.zeros((n_frames, n_joints))

    # OPTIMIZATION: Vectorized interpolation along the frames dimension using np.interp.
    # This avoids a python loop computing the piecewise interpolation frame by frame.
    phases = objective.phases
    phase_fractions = np.array([p.fraction for p in phases])
    phase_targets = np.array([_phase_to_array(p, joint_names) for p in phases])

    for j in range(n_joints):
        keyframes[:, j] = np.interp(fractions, phase_fractions, phase_targets[:, j])

    return keyframes


__all__ = [
    "TrajectoryConfig",
    "TrajectoryResult",
    "compute_balance_cost",
    "compute_bar_path_cost",
    "interpolate_phases",
]
