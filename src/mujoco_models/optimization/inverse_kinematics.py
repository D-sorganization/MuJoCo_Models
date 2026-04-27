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
        ValueError: If exercise_name is unknown or n_frames < 2.
    """
    if n_frames < 2:
        msg = f"n_frames must be >= 2, got {n_frames}"
        raise ValueError(msg)

    objective = _lookup_objective(exercise_name)
    joint_names = objective.joint_names
    n_joints = len(joint_names)

    if not objective.phases:
        msg = f"Exercise '{exercise_name}' has no phases defined"
        raise ValueError(msg)

    fractions = np.linspace(0.0, 1.0, n_frames)
    keyframes = np.zeros((n_frames, n_joints))

    for frame_idx, frac in enumerate(fractions):
        angles = _interpolate_at_fraction(frac, objective.phases, joint_names)
        keyframes[frame_idx] = angles

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
        ValueError: If the name is not registered.
    """
    if exercise_name not in OBJECTIVE_REGISTRY:
        available = sorted(OBJECTIVE_REGISTRY.keys())
        msg = f"Unknown exercise '{exercise_name}'. Available: {available}"
        raise ValueError(msg)
    return OBJECTIVE_REGISTRY[exercise_name]


def _interpolate_at_fraction(
    fraction: float,
    phases: list[Phase],
    joint_names: list[str],
) -> np.ndarray:
    """Interpolate joint angles at a given movement fraction.

    Finds the two bracketing phases and linearly interpolates
    between their target joint angles.

    Args:
        fraction: Position in the movement, 0.0 to 1.0.
        phases: Ordered list of movement phases.
        joint_names: Sorted list of joint names.

    Returns:
        Array of joint angles, shape (n_joints,).
    """
    # Clamp to phase boundaries
    if fraction <= phases[0].fraction:
        return _phase_to_array(phases[0], joint_names)
    if fraction >= phases[-1].fraction:
        return _phase_to_array(phases[-1], joint_names)

    # Find bracketing phases
    for i in range(len(phases) - 1):
        if phases[i].fraction <= fraction <= phases[i + 1].fraction:
            phase_a = phases[i]
            phase_b = phases[i + 1]
            span = phase_b.fraction - phase_a.fraction
            if span < 1e-12:
                return _phase_to_array(phase_b, joint_names)
            t = (fraction - phase_a.fraction) / span
            arr_a = _phase_to_array(phase_a, joint_names)
            arr_b = _phase_to_array(phase_b, joint_names)
            return arr_a + t * (arr_b - arr_a)

    # Fallback (should not reach here with valid phases)
    return _phase_to_array(phases[-1], joint_names)


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
