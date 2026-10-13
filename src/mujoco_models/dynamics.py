# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Dynamics computation API for MuJoCo exercise models.

Provides public wrappers around MuJoCo dynamics routines (such as
:func:`mujoco.mj_inverse`) with input validation, Design-by-Contract
preconditions, and support for exercise names or model builders.

Active contacts (issue #453)
----------------------------
``mj_inverse`` evaluates the soft-contact law at the given state and
subtracts the resulting constraint force, so ``qfrc_inverse`` satisfies
``M qacc + qfrc_bias = qfrc_inverse + qfrc_passive + qfrc_constraint``.
For the unactuated free root joint, ``qfrc_inverse`` is therefore the load
the contacts do *not* explain (the residual "hand of God" force), not the
whole weight.  :func:`inverse_dynamics_with_contacts` reports that residual
next to the contact generalized force ``J_c^T f_c``, so the contact load is
never silently folded into (or hidden from) the free joint.  Excluding the
contacts instead would put the weight on the free joint, which is not what
holds a standing body up, and would no longer match ``mj_inverse``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mujoco_models.exceptions import ValidationError
from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.exercises.base import ExerciseModelBuilder
from mujoco_models.grip import dense_efc_jacobian
from mujoco_models.shared.contracts.preconditions import (
    require_finite,
    require_shape,
)

#: Degrees of freedom of a free joint (3 translations, then 3 rotations).
FREE_JOINT_DOFS = 6
_CONTACT_CONSTRAINT_TYPES = (
    int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
    int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
    int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
)


@dataclass(frozen=True)
class ContactInverseDynamics:
    """Inverse dynamics split into the contact load and the unexplained rest.

    Attributes:
        qfrc_inverse: ``mj_inverse`` generalized forces, shape ``(nv,)``:
            joint torques, and on the free root joint the residual the
            contacts do not explain.  Identical to :func:`inverse_dynamics`.
        qfrc_contact: Generalized force of the active contacts
            ``J_c^T f_c`` at the requested state, shape ``(nv,)``.
        contact_count: Number of active contacts (``data.ncon``).
        root_dofs: Slice of the free root joint's degrees of freedom.
    """

    qfrc_inverse: np.ndarray
    qfrc_contact: np.ndarray
    contact_count: int
    root_dofs: slice

    @property
    def root_residual(self) -> np.ndarray:
        """Root force/torque the contacts do not explain (N, N*m), shape (6,)."""
        return self.qfrc_inverse[self.root_dofs]

    @property
    def root_contact(self) -> np.ndarray:
        """Contact load on the root degrees of freedom (N, N*m), shape (6,)."""
        return self.qfrc_contact[self.root_dofs]


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
    ``qfrc_inverse = M(q) * qacc + C(q, qvel) + G(q)`` minus the passive and
    constraint (contact) forces at that state.

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
        freejoint degrees of freedom, entries are forces (N).  With active
        contacts the free-joint entries are the residual the contacts do not
        explain; :func:`inverse_dynamics_with_contacts` reports the contact
        load next to it.

    Raises:
        ValidationError: If *exercise* is invalid, or if *q*, *qvel*, or *qacc*
            contain non-finite values or violate expected shapes.
    """
    model = _resolve_model(exercise)
    data = _solve_inverse(model, q, qvel, qacc)
    return _checked_generalized_force(data.qfrc_inverse, model.nv, "qfrc_inverse")


def inverse_dynamics_with_contacts(
    exercise: str | ExerciseModelBuilder | mujoco.MjModel,
    q: ArrayLike,
    qvel: ArrayLike | None = None,
    qacc: ArrayLike | None = None,
) -> ContactInverseDynamics:
    """Inverse dynamics that reports the contact load and the root residual.

    Same inputs and validation as :func:`inverse_dynamics`.  The contact
    force is the one ``mj_inverse`` uses at this state (the soft-contact law
    evaluated at ``q``, ``qvel`` and ``qacc``), so
    ``qfrc_inverse + qfrc_contact + qfrc_passive`` equals
    ``M qacc + qfrc_bias`` when no other constraint is active.

    Raises:
        ValidationError: As :func:`inverse_dynamics`, or if the model has no
            free root joint (the residual is undefined for a fixed base).
    """
    model = _resolve_model(exercise)
    root = _root_dofs(model)
    data = _solve_inverse(model, q, qvel, qacc)
    return ContactInverseDynamics(
        qfrc_inverse=_checked_generalized_force(
            data.qfrc_inverse, model.nv, "qfrc_inverse"
        ),
        qfrc_contact=_checked_generalized_force(
            _contact_generalized_force(model, data), model.nv, "qfrc_contact"
        ),
        contact_count=int(data.ncon),
        root_dofs=root,
    )


def _validated_vector(value: ArrayLike | None, size: int, name: str) -> np.ndarray:
    """Return *value* as a finite float vector of *size* (zeros if None)."""
    if value is None:
        return np.zeros(size, dtype=float)
    require_finite(value, name)
    require_shape(value, (size,), name)
    return np.asarray(value, dtype=float)


def _solve_inverse(
    model: mujoco.MjModel,
    q: ArrayLike,
    qvel: ArrayLike | None,
    qacc: ArrayLike | None,
) -> mujoco.MjData:
    """Validate the state and run ``mj_inverse`` on fresh data."""
    require_finite(q, "q")
    require_shape(q, (model.nq,), "q")
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(q, dtype=float)
    data.qvel[:] = _validated_vector(qvel, model.nv, "qvel")
    data.qacc[:] = _validated_vector(qacc, model.nv, "qacc")
    mujoco.mj_inverse(model, data)
    return data


def _checked_generalized_force(value: np.ndarray, nv: int, name: str) -> np.ndarray:
    """Postcondition: return a finite copy of shape ``(nv,)``."""
    result = np.array(value, dtype=float)
    require_finite(result, name)
    require_shape(result, (nv,), name)
    return result


def _root_dofs(model: mujoco.MjModel) -> slice:
    """Degrees of freedom of the free joint that attaches a body to the world."""
    for joint in range(model.njnt):
        is_free = model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
        if is_free and model.body_parentid[model.jnt_bodyid[joint]] == 0:
            start = int(model.jnt_dofadr[joint])
            return slice(start, start + FREE_JOINT_DOFS)
    msg = "contact-aware inverse dynamics needs a free root joint"
    raise ValidationError(msg)


def _contact_generalized_force(
    model: mujoco.MjModel, data: mujoco.MjData
) -> np.ndarray:
    """``J_c^T f_c`` summed over the contact rows of the constraint solve."""
    nefc = int(data.nefc)
    rows = np.isin(data.efc_type[:nefc], _CONTACT_CONSTRAINT_TYPES)
    jacobian = dense_efc_jacobian(model, data)
    result: np.ndarray = jacobian[rows].T @ data.efc_force[:nefc][rows]
    return result


__all__ = [
    "FREE_JOINT_DOFS",
    "ContactInverseDynamics",
    "inverse_dynamics",
    "inverse_dynamics_with_contacts",
]
