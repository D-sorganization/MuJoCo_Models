"""Gait (walking) exercise model builder."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from mujoco_models.exercises.gait.gait_model import (
    GaitModelBuilder,
    build_gait_model,
)

__all__ = ["GaitModelBuilder", "build_gait_model"]
