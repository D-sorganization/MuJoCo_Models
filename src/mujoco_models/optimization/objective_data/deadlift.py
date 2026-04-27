# SPDX-License-Identifier: MIT
"""Deadlift exercise objective data.

All joint angles are in radians.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)
from mujoco_models.optimization.objective_data.shared_poses import STANDING_POSE

_DEADLIFT_BOTTOM: dict[str, float] = {
    "hip_l_flex": 1.40,
    "hip_r_flex": 1.40,
    "knee_l_flex": -1.05,
    "knee_r_flex": -1.05,
    "ankle_l_flex": 0.35,
    "ankle_r_flex": 0.35,
    "shoulder_l_flex": -0.17,
    "shoulder_r_flex": -0.17,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

DEADLIFT_OBJECTIVE: ExerciseObjective = ExerciseObjective(
    name="deadlift",
    start_pose=dict(_DEADLIFT_BOTTOM),
    end_pose=dict(STANDING_POSE),
    phases=[
        Phase("floor", 0.0, dict(_DEADLIFT_BOTTOM)),
        Phase(
            "below_knee",
            0.25,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -0.52,
                "knee_r_flex": -0.52,
                "ankle_l_flex": 0.17,
                "ankle_r_flex": 0.17,
            },
        ),
        Phase(
            "above_knee",
            0.5,
            {
                "hip_l_flex": 0.70,
                "hip_r_flex": 0.70,
                "knee_l_flex": -0.26,
                "knee_r_flex": -0.26,
                "ankle_l_flex": 0.09,
                "ankle_r_flex": 0.09,
            },
        ),
        Phase("lockout", 1.0, dict(STANDING_POSE)),
    ],
    bar_path_constraint="vertical",
    balance_mode="bilateral_stance",
)
