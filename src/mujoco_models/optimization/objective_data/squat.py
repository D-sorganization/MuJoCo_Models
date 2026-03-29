"""Squat exercise objective data.

All joint angles are in radians.
"""

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)
from mujoco_models.optimization.objective_data.shared_poses import STANDING_POSE

SQUAT_OBJECTIVE: ExerciseObjective = ExerciseObjective(
    name="back_squat",
    start_pose=dict(STANDING_POSE),
    end_pose=dict(STANDING_POSE),
    phases=[
        Phase("descent_start", 0.0, dict(STANDING_POSE)),
        Phase(
            "half_depth",
            0.25,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -1.05,
                "knee_r_flex": -1.05,
                "ankle_l_flex": 0.3,
                "ankle_r_flex": 0.3,
            },
        ),
        Phase(
            "bottom",
            0.5,
            {
                "hip_l_flex": 2.09,
                "hip_r_flex": 2.09,
                "knee_l_flex": -2.09,
                "knee_r_flex": -2.09,
                "ankle_l_flex": 0.6,
                "ankle_r_flex": 0.6,
            },
        ),
        Phase(
            "drive",
            0.75,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -1.05,
                "knee_r_flex": -1.05,
                "ankle_l_flex": 0.3,
                "ankle_r_flex": 0.3,
            },
        ),
        Phase("lockout", 1.0, dict(STANDING_POSE)),
    ],
    bar_path_constraint="vertical",
    balance_mode="bilateral_stance",
)
