# SPDX-License-Identifier: MIT
"""Clean and jerk exercise objective data.

Phases, timing and via-poses come from the vendored exercise spec
(standard v2.0, see :mod:`spec_lifts`).  All joint angles are in radians.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

from mujoco_models.optimization.exercise_objectives import ExerciseObjective
from mujoco_models.optimization.objective_data.spec_lifts import lift_objective

CLEAN_AND_JERK_OBJECTIVE: ExerciseObjective = lift_objective(
    "clean_and_jerk",
    name="clean_and_jerk",
    bar_path_constraint="s_curve",
    balance_mode="bilateral_stance",
)
