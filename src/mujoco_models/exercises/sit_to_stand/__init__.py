# SPDX-License-Identifier: MIT
"""Sit-to-stand exercise model builder."""

from mujoco_models.exercises.sit_to_stand.sit_to_stand_model import (
    SitToStandModelBuilder,
    build_sit_to_stand_model,
)

__all__ = ["SitToStandModelBuilder", "build_sit_to_stand_model"]
