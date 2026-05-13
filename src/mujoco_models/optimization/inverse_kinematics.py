# SPDX-License-Identifier: MIT
"""Inverse kinematics keyframe generation for barbell exercises.

Generates joint-angle keyframes by interpolating between the phases
defined in exercise objectives. This provides initial trajectories
for downstream trajectory optimization.

Design by Contract:
    - Exercise name must map to a registered objective.
    - Number of frames must be >= 2.
    - Output keyframes have shape (n_frames, n_joints).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging

import numpy as np

from mujoco_models.exceptions import ValidationError
from mujoco_models.optimization.exercise_objectives import (
    OBJECTIVE_REGISTRY,
    ExerciseObjective,
    Phase,
)

logger = logging.getLogger(__name__)


def solve_ik_keyframes(
    exercise_name: str,
    n_frames: int = 50,
) -> np.ndarray:
    """Generate joint angle keyframes by interpolating phases.

    Produces a smooth trajectory through the movement phases defined
    in the exercise objective using linear interpolation between
    consecutive phase targets.

    Args:
        exercise_name: Name of the exercise (must be a key in
            OBJECTIVE_REGISTRY).
        n_frames: Number of output keyframes (must be >= 2).

    Returns:
        Joint angle array with shape (n_frames, n_joints), where
        joints are in sorted alphabetical order matching
        ``objective.joint_names``.

    Raises:
        ValidationError: If exercise_name is unknown or n_frames < 2.
    """
    if n_frames < 2:
        msg = f"n_frames must be >= 2, got {n_frames}"
        raise ValidationError(msg)

    objective = _lookup_objective(exercise_name)
    joint_names = objective.joint_names
    n_joints = len(joint_names)

    if not objective.phases:
        msg = f"Exercise '{exercise_name}' has no phases defined"
        raise ValidationError(msg)

    fractions = np.linspace(0.0, 1.0, n_frames)
    keyframes = np.zeros((n_frames, n_joints))

    # OPTIMIZATION: Vectorized interpolation along the frames dimension using np.interp.
    # This avoids a python loop computing the piecewise interpolation frame by frame.
    phases = objective.phases
    phase_fractions = np.array([p.fraction for p in phases])
    phase_targets = np.array([_phase_to_array(p, joint_names) for p in phases])

    for j in range(n_joints):
        keyframes[:, j] = np.interp(fractions, phase_fractions, phase_targets[:, j])

    logger.debug(
        "Generated %d keyframes for '%s' with %d joints",
        n_frames,
        exercise_name,
        n_joints,
    )
    return keyframes


def _lookup_objective(
    exercise_name: str,
) -> ExerciseObjective:
    """Look up an exercise objective by name.

    Args:
        exercise_name: Exercise name key.

    Returns:
        The matching ExerciseObjective.

    Raises:
        ValidationError: If the name is not registered.
    """
    if exercise_name not in OBJECTIVE_REGISTRY:
        available = sorted(OBJECTIVE_REGISTRY.keys())
        msg = f"Unknown exercise '{exercise_name}'. Available: {available}"
        raise ValidationError(msg)
    return OBJECTIVE_REGISTRY[exercise_name]


def _phase_to_array(
    phase: Phase,
    joint_names: list[str],
) -> np.ndarray:
    """Convert a phase's target joints to a numpy array.

    Missing joints default to 0.0 (neutral position).

    Args:
        phase: Movement phase with target joint angles.
        joint_names: Sorted list of all joint names.

    Returns:
        Array of joint angles, shape (n_joints,).
    """
    return np.array([phase.target_joints.get(name, 0.0) for name in joint_names])
