"""Exercise objective data for functional movements (gait, sit-to-stand).

Re-exports from per-exercise modules under ``objective_data/``.
This file is kept for backward compatibility.

All joint angles are in radians.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization


from __future__ import annotations

from mujoco_models.optimization.objective_data.functional import (
    GAIT_OBJECTIVE,
    SIT_TO_STAND_OBJECTIVE,
)

__all__ = [
    "GAIT_OBJECTIVE",
    "SIT_TO_STAND_OBJECTIVE",
]
