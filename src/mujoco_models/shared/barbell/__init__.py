"""Barbell model generation for MuJoCo MJCF.

Provides factory functions to create standard Olympic barbell bodies
with correct mass, geometry, and inertia properties.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization


from mujoco_models.shared.barbell.barbell_model import (
    BarbellSpec,
    create_barbell_bodies,
)

__all__ = ["BarbellSpec", "create_barbell_bodies"]
