# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Functional movement objective data (gait, sit-to-stand).

Phase names and timing (``t_frac``) come from the vendored exercise spec
(standard v2.0/v2.1). Targets remain pack-local. All joint angles are in radians.
"""

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)
from mujoco_models.optimization.objective_data.shared_poses import STANDING_POSE
from mujoco_models.shared.parity._canonical import spec
from mujoco_models.shared.parity.standard import STANDARD


def _build_functional_phases(
    exercise: str, targets: dict[str, dict[str, float]]
) -> list[Phase]:
    """Build phases taking ordered names and t_frac from the vendored spec."""
    phases = spec.exercise_spec(STANDARD, exercise)["phases"]
    return [
        Phase(str(p["name"]), float(p["t_frac"]), dict(targets[str(p["name"])]))
        for p in phases
    ]


# ---------- GAIT (walking) ----------

_GAIT_HEEL_STRIKE: dict[str, float] = {
    "hip_l_flex": 0.35,
    "hip_r_flex": -0.17,
    "knee_l_flex": -0.09,
    "knee_r_flex": -0.35,
    "ankle_l_flex": 0.17,
    "ankle_r_flex": -0.09,
}

_GAIT_TARGETS: dict[str, dict[str, float]] = {
    "initial_contact": dict(_GAIT_HEEL_STRIKE),
    "loading_response": {
        "hip_l_flex": 0.26,
        "hip_r_flex": -0.09,
        "knee_l_flex": -0.17,
        "knee_r_flex": -0.26,
        "ankle_l_flex": 0.09,
        "ankle_r_flex": -0.17,
    },
    "mid_stance": {
        "hip_l_flex": 0.0,
        "hip_r_flex": 0.09,
        "knee_l_flex": -0.09,
        "knee_r_flex": -0.09,
        "ankle_l_flex": -0.09,
        "ankle_r_flex": 0.09,
    },
    "terminal_stance": {
        "hip_l_flex": -0.17,
        "hip_r_flex": 0.26,
        "knee_l_flex": -0.09,
        "knee_r_flex": -0.09,
        "ankle_l_flex": -0.26,
        "ankle_r_flex": 0.17,
    },
    "pre_swing": {
        "hip_l_flex": -0.09,
        "hip_r_flex": 0.35,
        "knee_l_flex": -0.35,
        "knee_r_flex": -0.09,
        "ankle_l_flex": -0.35,
        "ankle_r_flex": 0.09,
    },
    "initial_swing": {
        "hip_l_flex": 0.09,
        "hip_r_flex": 0.26,
        "knee_l_flex": -0.52,
        "knee_r_flex": -0.09,
        "ankle_l_flex": -0.09,
        "ankle_r_flex": 0.0,
    },
    "mid_swing": {
        "hip_l_flex": 0.26,
        "hip_r_flex": 0.09,
        "knee_l_flex": -0.35,
        "knee_r_flex": -0.17,
        "ankle_l_flex": 0.0,
        "ankle_r_flex": 0.09,
    },
    "terminal_swing": dict(_GAIT_HEEL_STRIKE),
}

GAIT_OBJECTIVE: ExerciseObjective = ExerciseObjective(
    name="gait",
    start_pose=dict(_GAIT_HEEL_STRIKE),
    end_pose=dict(_GAIT_HEEL_STRIKE),
    phases=_build_functional_phases("gait", _GAIT_TARGETS),
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

_SIT_TO_STAND_TARGETS: dict[str, dict[str, float]] = {
    "seated": dict(_SEATED_POSE),
    "trunk_flexion": {
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
    "seat_off": {
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
    "rising": {
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
    "extension_end": {
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
    "stabilisation": dict(STANDING_POSE),
}

SIT_TO_STAND_OBJECTIVE: ExerciseObjective = ExerciseObjective(
    name="sit_to_stand",
    start_pose=dict(_SEATED_POSE),
    end_pose=dict(STANDING_POSE),
    phases=_build_functional_phases("sit_to_stand", _SIT_TO_STAND_TARGETS),
    bar_path_constraint="none",
    balance_mode="bilateral_stance",
)
