# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Dynamics computation API for MuJoCo exercise models.

Provides public wrappers around MuJoCo dynamics routines (such as
:func:`mujoco.mj_inverse`) with input validation, Design-by-Contract
preconditions, and support for exercise names or model builders.
"""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mujoco_models.exceptions import ValidationError
from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.exercises.base import ExerciseModelBuilder
from mujoco_models.shared.contracts.preconditions import (
    require_finite,
    require_shape,
)


def _resolve_model(
    exercise: str | ExerciseModelBuilder | mujoco.MjModel,
) -> mujoco.MjModel:
    """Resolve an exercise specification to a compiled MjModel.

    Args:
        exercise: Exercise name (from EXERCISE_REGISTRY),
            ExerciseModelBuilder instance, or compiled MjModel.

    Returns:
        Compiled mujoco.MjModel.

    Raises:
        ValidationError: If exercise cannot be resolved to an MjModel.
    """
    if isinstance(exercise, mujoco.MjModel):
        return exercise

    if isinstance(exercise, ExerciseModelBuilder):
        return mujoco.MjModel.from_xml_string(exercise.build())

    if isinstance(exercise, str):
        if exercise in EXERCISE_REGISTRY:
            return mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY[exercise]().build())
        available = sorted(EXERCISE_REGISTRY.keys())
        msg = f"Unknown exercise {exercise!r}. Available exercises: {available}"
        raise ValidationError(msg)

    msg = (
        "exercise must be an exercise name (str), ExerciseModelBuilder, "
        f"or mujoco.MjModel, got {type(exercise).__name__}"
    )
    raise ValidationError(msg)


def inverse_dynamics(
    exercise: str | ExerciseModelBuilder | mujoco.MjModel,
    q: ArrayLike,
    qvel: ArrayLike | None = None,
    qacc: ArrayLike | None = None,
) -> np.ndarray:
    """Compute generalized forces balancing state and acceleration via mj_inverse.

    Calculates joint torques and generalized forces:
    ``qfrc_inverse = M(q) * qacc + C(q, qvel) + G(q)``.

    Args:
        exercise: Target exercise model specified as an exercise name
            (e.g. ``'gait'``, ``'squat'``), an :class:`ExerciseModelBuilder`
            instance, or a compiled :class:`mujoco.MjModel`.
        q: Generalized coordinates ``qpos`` with shape ``(nq,)``.
        qvel: Generalized velocities with shape ``(nv,)``. Defaults to
            zeros if omitted or None.
        qacc: Generalized accelerations with shape ``(nv,)``. Defaults to
            zeros if omitted or None.

    Returns:
        Generalized forces array ``qfrc_inverse`` with shape ``(nv,)``.
        For hinge joints, entries are joint torques (N·m); for translational
        freejoint degrees of freedom, entries are forces (N).

    Raises:
        ValidationError: If *exercise* is invalid, or if *q*, *qvel*, or *qacc*
            contain non-finite values or violate expected shapes.
    """
    model = _resolve_model(exercise)

    require_finite(q, "q")
    require_shape(q, (model.nq,), "q")
    q_arr = np.asarray(q, dtype=float)

    if qvel is None:
        qvel_arr = np.zeros(model.nv, dtype=float)
    else:
        require_finite(qvel, "qvel")
        require_shape(qvel, (model.nv,), "qvel")
        qvel_arr = np.asarray(qvel, dtype=float)

    if qacc is None:
        qacc_arr = np.zeros(model.nv, dtype=float)
    else:
        require_finite(qacc, "qacc")
        require_shape(qacc, (model.nv,), "qacc")
        qacc_arr = np.asarray(qacc, dtype=float)

    data = mujoco.MjData(model)
    data.qpos[:] = q_arr
    data.qvel[:] = qvel_arr
    data.qacc[:] = qacc_arr

    mujoco.mj_inverse(model, data)
    torques = data.qfrc_inverse.copy()

    require_finite(torques, "qfrc_inverse")
    require_shape(torques, (model.nv,), "qfrc_inverse")
    return torques


__all__ = ["inverse_dynamics"]
