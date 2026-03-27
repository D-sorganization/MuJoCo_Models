"""Exercise objective data constants for barbell movements.

Contains the kinematic target data (poses, phases, joint angles) for
each exercise. Separated from exercise_objectives.py to keep modules
under the size budget.

All joint angles are in radians.
"""

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)

# ---------- Standing pose (shared by squat, deadlift) ----------

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


# ---------- SQUAT ----------

SQUAT_OBJECTIVE = ExerciseObjective(
    name="back_squat",
    start_pose=dict(_STANDING_POSE),
    end_pose=dict(_STANDING_POSE),
    phases=[
        Phase(
            "descent_start",
            0.0,
            dict(_STANDING_POSE),
        ),
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
        Phase(
            "lockout",
            1.0,
            dict(_STANDING_POSE),
        ),
    ],
    bar_path_constraint="vertical",
    balance_mode="bilateral_stance",
)


# ---------- DEADLIFT ----------

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

DEADLIFT_OBJECTIVE = ExerciseObjective(
    name="deadlift",
    start_pose=dict(_DEADLIFT_BOTTOM),
    end_pose=dict(_STANDING_POSE),
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
        Phase(
            "lockout",
            1.0,
            dict(_STANDING_POSE),
        ),
    ],
    bar_path_constraint="vertical",
    balance_mode="bilateral_stance",
)


# ---------- BENCH PRESS ----------

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

BENCH_PRESS_OBJECTIVE = ExerciseObjective(
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

# ---------- GAIT (walking) ----------

_GAIT_HEEL_STRIKE: dict[str, float] = {
    "hip_l_flex": 0.35,
    "hip_r_flex": -0.17,
    "knee_l_flex": -0.09,
    "knee_r_flex": -0.35,
    "ankle_l_flex": 0.17,
    "ankle_r_flex": -0.09,
}

GAIT_OBJECTIVE = ExerciseObjective(
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

SIT_TO_STAND_OBJECTIVE = ExerciseObjective(
    name="sit_to_stand",
    start_pose=dict(_SEATED_POSE),
    end_pose=dict(_STANDING_POSE),
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
        Phase("standing", 1.0, dict(_STANDING_POSE)),
    ],
    bar_path_constraint="none",
    balance_mode="bilateral_stance",
)
