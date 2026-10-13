# SPDX-License-Identifier: MIT
"""Configuration classes for MuJoCo exercise model builders."""

from __future__ import annotations

from dataclasses import dataclass, field

from mujoco_models.shared.barbell import BarbellSpec
from mujoco_models.shared.body import BodyModelSpec


@dataclass(frozen=True)
class ExerciseConfig:
    """Configuration common to all exercise models."""

    body_spec: BodyModelSpec = field(default_factory=BodyModelSpec)
    barbell_spec: BarbellSpec = field(default_factory=BarbellSpec.mens_olympic)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.80665)
