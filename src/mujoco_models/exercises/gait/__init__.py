# SPDX-License-Identifier: MIT
"""Gait (walking) exercise model builder."""

from mujoco_models.exercises.gait.gait_model import (
    GaitModelBuilder,
    build_gait_model,
)

__all__ = ["GaitModelBuilder", "build_gait_model"]
