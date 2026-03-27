"""Exercise objective data constants for Olympic lifts.

Contains kinematic target data for snatch and clean & jerk.
Separated from exercise_objective_data.py to keep modules under the
300-line budget. Functional movements are in
exercise_objective_data_functional.py.

All joint angles are in radians.
"""

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)

# ---------- Shared standing reference ----------

_STANDING_POSE: dict[str, float] = {
    "hip_l_flex": 0.0,
    "hip_r_flex": 0.0,
    "knee_l_flex": 0.0,
    "knee_r_flex": 0.0,
    "ankle_l_flex": 0.0,
    "ankle_r_flex": 0.0,
    "shoulder_l_flex": 0.0,
    "shoulder_r_flex": 0.0,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}


# ---------- SNATCH ----------

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

SNATCH_OBJECTIVE = ExerciseObjective(
    name="snatch",
    start_pose=dict(_SNATCH_START),
    end_pose=dict(_STANDING_POSE),
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
                **_STANDING_POSE,
                "shoulder_l_flex": 3.14,
                "shoulder_r_flex": 3.14,
            },
        ),
    ],
    bar_path_constraint="s_curve",
    balance_mode="bilateral_stance",
)


# ---------- CLEAN AND JERK ----------

_CLEAN_START: dict[str, float] = {
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

_FRONT_RACK: dict[str, float] = {
    "hip_l_flex": 0.0,
    "hip_r_flex": 0.0,
    "knee_l_flex": 0.0,
    "knee_r_flex": 0.0,
    "ankle_l_flex": 0.0,
    "ankle_r_flex": 0.0,
    "shoulder_l_flex": 1.57,
    "shoulder_r_flex": 1.57,
    "elbow_l_flex": -2.36,
    "elbow_r_flex": -2.36,
}

_JERK_DIP: dict[str, float] = {
    "hip_l_flex": 0.35,
    "hip_r_flex": 0.35,
    "knee_l_flex": -0.52,
    "knee_r_flex": -0.52,
    "ankle_l_flex": 0.17,
    "ankle_r_flex": 0.17,
    "shoulder_l_flex": 1.57,
    "shoulder_r_flex": 1.57,
    "elbow_l_flex": -2.36,
    "elbow_r_flex": -2.36,
}

_JERK_CATCH: dict[str, float] = {
    "hip_l_flex": 0.35,
    "hip_r_flex": 0.35,
    "knee_l_flex": -0.52,
    "knee_r_flex": -0.52,
    "ankle_l_flex": 0.17,
    "ankle_r_flex": 0.17,
    "shoulder_l_flex": 3.14,
    "shoulder_r_flex": 3.14,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

_CJ_FINISH: dict[str, float] = {
    **_STANDING_POSE,
    "shoulder_l_flex": 3.14,
    "shoulder_r_flex": 3.14,
}

CLEAN_AND_JERK_OBJECTIVE = ExerciseObjective(
    name="clean_and_jerk",
    start_pose=dict(_CLEAN_START),
    end_pose=dict(_CJ_FINISH),
    phases=[
        # Clean phases
        Phase("clean_pull", 0.0, dict(_CLEAN_START)),
        Phase(
            "clean_transition",
            0.15,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -0.52,
                "knee_r_flex": -0.52,
                "ankle_l_flex": 0.17,
                "ankle_r_flex": 0.17,
                "shoulder_l_flex": 0.35,
                "shoulder_r_flex": 0.35,
                "elbow_l_flex": -0.52,
                "elbow_r_flex": -0.52,
            },
        ),
        Phase(
            "clean_extension",
            0.30,
            {
                "hip_l_flex": 0.17,
                "hip_r_flex": 0.17,
                "knee_l_flex": -0.17,
                "knee_r_flex": -0.17,
                "ankle_l_flex": -0.26,
                "ankle_r_flex": -0.26,
                "shoulder_l_flex": 0.87,
                "shoulder_r_flex": 0.87,
                "elbow_l_flex": -1.57,
                "elbow_r_flex": -1.57,
            },
        ),
        Phase("clean_catch", 0.45, dict(_FRONT_RACK)),
        # Jerk phases
        Phase("jerk_dip", 0.55, dict(_JERK_DIP)),
        Phase(
            "jerk_drive",
            0.70,
            {
                "hip_l_flex": -0.09,
                "hip_r_flex": -0.09,
                "knee_l_flex": 0.0,
                "knee_r_flex": 0.0,
                "ankle_l_flex": -0.26,
                "ankle_r_flex": -0.26,
                "shoulder_l_flex": 2.36,
                "shoulder_r_flex": 2.36,
                "elbow_l_flex": -1.05,
                "elbow_r_flex": -1.05,
            },
        ),
        Phase("jerk_catch", 0.85, dict(_JERK_CATCH)),
        Phase("recovery", 1.0, dict(_CJ_FINISH)),
    ],
    bar_path_constraint="s_curve",
    balance_mode="bilateral_stance",
)
