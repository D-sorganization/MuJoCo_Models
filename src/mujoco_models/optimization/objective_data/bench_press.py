"""Bench press exercise objective data.

All joint angles are in radians.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)

_BENCH_LOCKOUT: dict[str, float] = {
    "shoulder_l_flex": 1.57,
    "shoulder_r_flex": 1.57,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

_BENCH_BOTTOM: dict[str, float] = {
    "shoulder_l_flex": 0.52,
    "shoulder_r_flex": 0.52,
    "elbow_l_flex": -1.57,
    "elbow_r_flex": -1.57,
}

BENCH_PRESS_OBJECTIVE: ExerciseObjective = ExerciseObjective(
    name="bench_press",
    start_pose=dict(_BENCH_LOCKOUT),
    end_pose=dict(_BENCH_LOCKOUT),
    phases=[
        Phase("lockout_start", 0.0, dict(_BENCH_LOCKOUT)),
        Phase(
            "descent",
            0.25,
            {
                "shoulder_l_flex": 1.05,
                "shoulder_r_flex": 1.05,
                "elbow_l_flex": -0.79,
                "elbow_r_flex": -0.79,
            },
        ),
        Phase("chest", 0.5, dict(_BENCH_BOTTOM)),
        Phase(
            "press",
            0.75,
            {
                "shoulder_l_flex": 1.05,
                "shoulder_r_flex": 1.05,
                "elbow_l_flex": -0.79,
                "elbow_r_flex": -0.79,
            },
        ),
        Phase("lockout_finish", 1.0, dict(_BENCH_LOCKOUT)),
    ],
    bar_path_constraint="j_curve",
    balance_mode="supine",
)
