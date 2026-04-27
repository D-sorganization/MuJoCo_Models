# SPDX-License-Identifier: MIT
"""Per-exercise objective data modules.

Each module defines kinematic target data (poses, phases, joint angles)
for a single exercise or small group of related movements.  All joint
angles are in radians.
"""

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
from mujoco_models.optimization.objective_data.shared_poses import STANDING_POSE
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
    "STANDING_POSE",
]
