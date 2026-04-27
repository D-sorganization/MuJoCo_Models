"""Exercise objective data constants for Olympic lifts.

Re-exports from per-exercise modules under ``objective_data/``.
This file is kept for backward compatibility.

All joint angles are in radians.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization


from __future__ import annotations

from mujoco_models.optimization.objective_data.clean_and_jerk import (
    CLEAN_AND_JERK_OBJECTIVE,
)
from mujoco_models.optimization.objective_data.snatch import SNATCH_OBJECTIVE

__all__ = [
    "CLEAN_AND_JERK_OBJECTIVE",
    "SNATCH_OBJECTIVE",
]
