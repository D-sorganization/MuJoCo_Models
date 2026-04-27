# SPDX-License-Identifier: MIT
"""Trajectory optimization and exercise objectives for barbell models.

Provides exercise-specific optimization objectives, trajectory optimization,
and inverse kinematics keyframe generation for MuJoCo barbell exercise models.
"""

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    BENCH_PRESS_OBJECTIVE,
    CLEAN_AND_JERK_OBJECTIVE,
    DEADLIFT_OBJECTIVE,
    EXERCISE_OBJECTIVES,
    SNATCH_OBJECTIVE,
    SQUAT_OBJECTIVE,
    ExerciseObjective,
    Phase,
    get_exercise_objective,
)
from mujoco_models.optimization.inverse_kinematics import solve_ik_keyframes
from mujoco_models.optimization.trajectory_optimizer import (
    TrajectoryConfig,
    TrajectoryResult,
    compute_balance_cost,
    compute_bar_path_cost,
    interpolate_phases,
)

__all__ = [
    "BENCH_PRESS_OBJECTIVE",
    "CLEAN_AND_JERK_OBJECTIVE",
    "DEADLIFT_OBJECTIVE",
    "EXERCISE_OBJECTIVES",
    "ExerciseObjective",
    "Phase",
    "SNATCH_OBJECTIVE",
    "SQUAT_OBJECTIVE",
    "TrajectoryConfig",
    "TrajectoryResult",
    "compute_balance_cost",
    "compute_bar_path_cost",
    "get_exercise_objective",
    "interpolate_phases",
    "solve_ik_keyframes",
]
