# SPDX-License-Identifier: MIT
"""Squat exercise objective data.

Phases, timing and via-poses come from the vendored exercise spec
(standard v2.0, see :mod:`spec_lifts`).  All joint angles are in radians.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import ExerciseObjective
from mujoco_models.optimization.objective_data.spec_lifts import lift_objective

SQUAT_OBJECTIVE: ExerciseObjective = lift_objective(
    "squat",
    name="back_squat",
    bar_path_constraint="vertical",
    balance_mode="bilateral_stance",
)
