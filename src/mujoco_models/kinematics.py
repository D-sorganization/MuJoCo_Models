# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Forward kinematics API for MuJoCo exercise models.

Wraps :func:`mujoco.mj_kinematics` with boundary validation and returns the
world-frame pose of every body segment keyed by segment name.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mujoco_models.dynamics import _resolve_model
from mujoco_models.exercises.base import ExerciseModelBuilder
from mujoco_models.shared.contracts.preconditions import (
    require_finite,
    require_shape,
)


@dataclass(frozen=True)
class SegmentPose:
    """World-frame pose of a body segment.

    Attributes:
        position: Segment origin in metres, shape ``(3,)``.
        orientation: Rotation matrix (segment to world), shape ``(3, 3)``.
    """

    position: np.ndarray
    orientation: np.ndarray


def forward_kinematics(
    exercise: str | ExerciseModelBuilder | mujoco.MjModel,
    q: ArrayLike,
) -> dict[str, SegmentPose]:
    """Compute world-frame segment poses for generalized coordinates *q*.

    Args:
        exercise: Exercise name, :class:`ExerciseModelBuilder`, or compiled
            :class:`mujoco.MjModel`.
        q: Generalized coordinates ``qpos`` with shape ``(nq,)``.

    Returns:
        Mapping from segment (body) name to :class:`SegmentPose`. The world
        body is included under its model name (``"world"``).

    Raises:
        ValidationError: If *exercise* is invalid, or *q* has the wrong shape
            or contains non-finite values (``ValidationError`` is a
            ``ValueError``).
    """
    model = _resolve_model(exercise)

    require_finite(q, "q")
    require_shape(q, (model.nq,), "q")

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(q, dtype=float)
    mujoco.mj_kinematics(model, data)

    return {
        model.body(i).name: SegmentPose(
            position=data.xpos[i].copy(),
            orientation=data.xmat[i].reshape(3, 3).copy(),
        )
        for i in range(model.nbody)
    }


__all__ = ["SegmentPose", "forward_kinematics"]
