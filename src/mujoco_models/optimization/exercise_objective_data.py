"""Exercise objective data constants for barbell movements.

Re-exports from per-exercise modules under ``objective_data/``.
This file is kept for backward compatibility.

All joint angles are in radians.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

from mujoco_models.optimization.objective_data.bench_press import (
    BENCH_PRESS_OBJECTIVE,
)
from mujoco_models.optimization.objective_data.clean_and_jerk import (
    CLEAN_AND_JERK_OBJECTIVE,
)
from mujoco_models.optimization.objective_data.deadlift import DEADLIFT_OBJECTIVE
from mujoco_models.optimization.objective_data.functional import (
    GAIT_OBJECTIVE,
    SIT_TO_STAND_OBJECTIVE,
)
from mujoco_models.optimization.objective_data.snatch import SNATCH_OBJECTIVE
from mujoco_models.optimization.objective_data.squat import SQUAT_OBJECTIVE

__all__ = [
    "BENCH_PRESS_OBJECTIVE",
    "CLEAN_AND_JERK_OBJECTIVE",
    "DEADLIFT_OBJECTIVE",
    "GAIT_OBJECTIVE",
    "SIT_TO_STAND_OBJECTIVE",
    "SNATCH_OBJECTIVE",
    "SQUAT_OBJECTIVE",
]
