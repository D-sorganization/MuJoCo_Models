"""Functional movement objective data (gait, sit-to-stand).

All joint angles are in radians.
"""

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)
from mujoco_models.optimization.objective_data.shared_poses import STANDING_POSE

# ---------- GAIT (walking) ----------

_GAIT_HEEL_STRIKE: dict[str, float] = {
    "hip_l_flex": 0.35,
    "hip_r_flex": -0.17,
    "knee_l_flex": -0.09,
    "knee_r_flex": -0.35,
    "ankle_l_flex": 0.17,
    "ankle_r_flex": -0.09,
}

GAIT_OBJECTIVE: ExerciseObjective = ExerciseObjective(
    name="gait",
    start_pose=dict(_GAIT_HEEL_STRIKE),
    end_pose=dict(_GAIT_HEEL_STRIKE),
    phases=[
        Phase("right_heel_strike", 0.0, dict(_GAIT_HEEL_STRIKE)),
        Phase(
            "right_loading",
            0.1,
            {
                "hip_l_flex": 0.26,
                "hip_r_flex": -0.09,
                "knee_l_flex": -0.17,
                "knee_r_flex": -0.26,
                "ankle_l_flex": 0.09,
                "ankle_r_flex": -0.17,
            },
        ),
        Phase(
            "right_midstance",
            0.3,
            {
                "hip_l_flex": 0.0,
                "hip_r_flex": 0.09,
                "knee_l_flex": -0.09,
                "knee_r_flex": -0.09,
                "ankle_l_flex": -0.09,
                "ankle_r_flex": 0.09,
            },
        ),
        Phase(
            "right_terminal_stance",
            0.5,
            {
                "hip_l_flex": -0.17,
                "hip_r_flex": 0.26,
                "knee_l_flex": -0.09,
                "knee_r_flex": -0.09,
                "ankle_l_flex": -0.26,
                "ankle_r_flex": 0.17,
            },
        ),
        Phase(
            "right_pre_swing",
            0.6,
            {
                "hip_l_flex": -0.09,
                "hip_r_flex": 0.35,
                "knee_l_flex": -0.35,
                "knee_r_flex": -0.09,
                "ankle_l_flex": -0.35,
                "ankle_r_flex": 0.09,
            },
        ),
        Phase(
            "right_initial_swing",
            0.7,
            {
                "hip_l_flex": 0.09,
                "hip_r_flex": 0.26,
                "knee_l_flex": -0.52,
                "knee_r_flex": -0.09,
                "ankle_l_flex": -0.09,
                "ankle_r_flex": 0.0,
            },
        ),
        Phase(
            "right_mid_swing",
            0.85,
            {
                "hip_l_flex": 0.26,
                "hip_r_flex": 0.09,
                "knee_l_flex": -0.35,
                "knee_r_flex": -0.17,
                "ankle_l_flex": 0.0,
                "ankle_r_flex": 0.09,
            },
        ),
        Phase(
            "right_terminal_swing",
            1.0,
            dict(_GAIT_HEEL_STRIKE),
        ),
    ],
    bar_path_constraint="none",
    balance_mode="alternating_stance",
)


# ---------- SIT-TO-STAND ----------

_SEATED_POSE: dict[str, float] = {
    "hip_l_flex": 1.57,
    "hip_r_flex": 1.57,
    "knee_l_flex": -1.57,
    "knee_r_flex": -1.57,
    "ankle_l_flex": 0.26,
    "ankle_r_flex": 0.26,
    "shoulder_l_flex": 0.0,
    "shoulder_r_flex": 0.0,
    "elbow_l_flex": -1.57,
    "elbow_r_flex": -1.57,
}

SIT_TO_STAND_OBJECTIVE: ExerciseObjective = ExerciseObjective(
    name="sit_to_stand",
    start_pose=dict(_SEATED_POSE),
    end_pose=dict(STANDING_POSE),
    phases=[
        Phase("seated", 0.0, dict(_SEATED_POSE)),
        Phase(
            "forward_lean",
            0.15,
            {
                "hip_l_flex": 1.75,
                "hip_r_flex": 1.75,
                "knee_l_flex": -1.57,
                "knee_r_flex": -1.57,
                "ankle_l_flex": 0.35,
                "ankle_r_flex": 0.35,
                "shoulder_l_flex": 0.35,
                "shoulder_r_flex": 0.35,
                "elbow_l_flex": -1.22,
                "elbow_r_flex": -1.22,
            },
        ),
        Phase(
            "momentum_generation",
            0.30,
            {
                "hip_l_flex": 1.40,
                "hip_r_flex": 1.40,
                "knee_l_flex": -1.40,
                "knee_r_flex": -1.40,
                "ankle_l_flex": 0.35,
                "ankle_r_flex": 0.35,
                "shoulder_l_flex": 0.52,
                "shoulder_r_flex": 0.52,
                "elbow_l_flex": -0.87,
                "elbow_r_flex": -0.87,
            },
        ),
        Phase(
            "seat_off",
            0.45,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -1.22,
                "knee_r_flex": -1.22,
                "ankle_l_flex": 0.26,
                "ankle_r_flex": 0.26,
                "shoulder_l_flex": 0.35,
                "shoulder_r_flex": 0.35,
                "elbow_l_flex": -0.52,
                "elbow_r_flex": -0.52,
            },
        ),
        Phase(
            "rising",
            0.75,
            {
                "hip_l_flex": 0.52,
                "hip_r_flex": 0.52,
                "knee_l_flex": -0.52,
                "knee_r_flex": -0.52,
                "ankle_l_flex": 0.09,
                "ankle_r_flex": 0.09,
                "shoulder_l_flex": 0.17,
                "shoulder_r_flex": 0.17,
                "elbow_l_flex": -0.26,
                "elbow_r_flex": -0.26,
            },
        ),
        Phase("standing", 1.0, dict(STANDING_POSE)),
    ],
    bar_path_constraint="none",
    balance_mode="bilateral_stance",
)
