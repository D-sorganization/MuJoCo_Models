"""Snatch exercise objective data.

All joint angles are in radians.
"""

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)
from mujoco_models.optimization.objective_data.shared_poses import STANDING_POSE

_SNATCH_START: dict[str, float] = {
    "hip_l_flex": 1.40,
    "hip_r_flex": 1.40,
    "knee_l_flex": -1.22,
    "knee_r_flex": -1.22,
    "ankle_l_flex": 0.44,
    "ankle_r_flex": 0.44,
    "shoulder_l_flex": -0.35,
    "shoulder_r_flex": -0.35,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

_SNATCH_OVERHEAD: dict[str, float] = {
    "hip_l_flex": 2.09,
    "hip_r_flex": 2.09,
    "knee_l_flex": -2.09,
    "knee_r_flex": -2.09,
    "ankle_l_flex": 0.6,
    "ankle_r_flex": 0.6,
    "shoulder_l_flex": 3.14,
    "shoulder_r_flex": 3.14,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

SNATCH_OBJECTIVE: ExerciseObjective = ExerciseObjective(
    name="snatch",
    start_pose=dict(_SNATCH_START),
    end_pose=dict(STANDING_POSE),
    phases=[
        Phase("first_pull", 0.0, dict(_SNATCH_START)),
        Phase(
            "transition",
            0.20,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -0.87,
                "knee_r_flex": -0.87,
                "ankle_l_flex": 0.26,
                "ankle_r_flex": 0.26,
                "shoulder_l_flex": 0.17,
                "shoulder_r_flex": 0.17,
                "elbow_l_flex": 0.0,
                "elbow_r_flex": 0.0,
            },
        ),
        Phase(
            "second_pull",
            0.40,
            {
                "hip_l_flex": 0.17,
                "hip_r_flex": 0.17,
                "knee_l_flex": -0.17,
                "knee_r_flex": -0.17,
                "ankle_l_flex": -0.35,
                "ankle_r_flex": -0.35,
                "shoulder_l_flex": 1.57,
                "shoulder_r_flex": 1.57,
                "elbow_l_flex": -1.05,
                "elbow_r_flex": -1.05,
            },
        ),
        Phase(
            "turnover",
            0.55,
            {
                "hip_l_flex": 0.70,
                "hip_r_flex": 0.70,
                "knee_l_flex": -0.70,
                "knee_r_flex": -0.70,
                "shoulder_l_flex": 2.62,
                "shoulder_r_flex": 2.62,
                "elbow_l_flex": -0.52,
                "elbow_r_flex": -0.52,
            },
        ),
        Phase("catch", 0.70, dict(_SNATCH_OVERHEAD)),
        Phase(
            "recovery",
            1.0,
            {
                **STANDING_POSE,
                "shoulder_l_flex": 3.14,
                "shoulder_r_flex": 3.14,
            },
        ),
    ],
    bar_path_constraint="s_curve",
    balance_mode="bilateral_stance",
)
