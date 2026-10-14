# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Task-target inverse kinematics and keyframe generation for barbell models.

Provides DLS inverse kinematics over MuJoCo Jacobians, with support
for Cartesian body/site targets, equality weld sync, and joint limits.

Design by Contract:
    - Target positions must be finite 3D Cartesian coordinates.
    - Limited joints remain within their defined ``jnt_range``.
    - Output residuals report Euclidean distance (metres) to each target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mujoco_models.dynamics import _resolve_model
from mujoco_models.exceptions import ValidationError
from mujoco_models.optimization.exercise_objectives import (
    OBJECTIVE_REGISTRY,
    ExerciseObjective,
    Phase,
)
from mujoco_models.shared.contracts.preconditions import (
    require_finite,
    require_shape,
)

if TYPE_CHECKING:
    from mujoco_models.exercises.base import ExerciseModelBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IKResult:
    """Result of a task-target inverse kinematics solve."""

    qpos: np.ndarray
    residuals: dict[str, float]
    converged: bool
    iterations: int


def _parse_targets(
    m: mujoco.MjModel,
    targets: dict[str, ArrayLike],
) -> dict[str, tuple[str, bool, int, int, np.ndarray]]:
    """Validate targets and map them to sites or bodies with weld parents."""
    if not targets:
        raise ValidationError("targets must not be empty")

    site_names = {m.site(i).name: i for i in range(m.nsite)}
    body_names = {m.body(i).name: i for i in range(m.nbody)}
    weld_parents = {
        int(m.eq_obj2id[i]): int(m.eq_obj1id[i])
        for i in range(m.neq)
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD
    }

    resolved: dict[str, tuple[str, bool, int, int, np.ndarray]] = {}
    for name, raw_pos in targets.items():
        require_finite(raw_pos, f"targets[{name}]")
        require_shape(raw_pos, (3,), f"targets[{name}]")
        target_pos = np.asarray(raw_pos, dtype=np.float64)
        tname = (
            "barbell_shaft"
            if name in ("bar", "barbell") and "barbell_shaft" in body_names
            else name
        )

        if tname in site_names:
            resolved[name] = (tname, True, site_names[tname], -1, target_pos)
        elif tname in body_names:
            bid = body_names[tname]
            resolved[name] = (tname, False, bid, weld_parents.get(bid, bid), target_pos)
        else:
            raise ValidationError(
                f"Target '{name}' not found in model bodies or sites."
            )
    return resolved


def solve_task_target_ik(
    model: str | ExerciseModelBuilder | mujoco.MjModel,
    targets: dict[str, ArrayLike],
    *,
    q_init: ArrayLike | None = None,
    max_iterations: int = 100,
    tolerance: float = 0.01,
    damping: float = 1e-2,
    step_size: float = 0.5,
) -> IKResult:
    """Solve inverse kinematics to reach Cartesian task targets."""
    m = _resolve_model(model)
    resolved_targets = _parse_targets(m, targets)

    d = mujoco.MjData(m)
    if q_init is not None:
        require_finite(q_init, "q_init")
        require_shape(q_init, (m.nq,), "q_init")
        d.qpos[:] = np.asarray(q_init, dtype=np.float64)
    elif m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        d.qpos[:] = m.qpos0.copy()

    mujoco.mj_fwdPosition(m, d)
    _sync_welds(m, d)
    mujoco.mj_fwdPosition(m, d)
    converged = False
    it_count = 0

    for iteration in range(max_iterations):
        it_count = iteration + 1
        err_list, jac_list, residuals = [], [], {}
        for name, (
            _,
            is_site,
            elem_id,
            attach_id,
            target_pos,
        ) in resolved_targets.items():
            if is_site:
                curr_pos = d.site_xpos[elem_id].copy()
                jacp = np.zeros((3, m.nv), dtype=np.float64)
                mujoco.mj_jacSite(m, d, jacp, None, elem_id)
            else:
                curr_pos = d.xpos[elem_id].copy()
                jacp = np.zeros((3, m.nv), dtype=np.float64)
                pt = np.ascontiguousarray(curr_pos, dtype=np.float64)
                mujoco.mj_jac(m, d, jacp, None, pt, attach_id)

            err = target_pos - curr_pos
            residuals[name] = float(np.linalg.norm(err))
            err_list.append(err)
            jac_list.append(jacp)

        if max(residuals.values()) <= tolerance:
            converged = True
            break

        e_vec = np.concatenate(err_list)
        j_mat = np.vstack(jac_list)
        jjt = j_mat @ j_mat.T + (damping**2) * np.eye(len(e_vec))
        dv = j_mat.T @ np.linalg.solve(jjt, e_vec)

        mujoco.mj_integratePos(m, d.qpos, dv, step_size)
        _clamp_joint_limits(m, d.qpos)
        mujoco.mj_normalizeQuat(m, d.qpos)
        _sync_welds(m, d)
        mujoco.mj_fwdPosition(m, d)

    final_residuals = {
        name: float(
            np.linalg.norm(
                target_pos - (d.site_xpos[elem_id] if is_site else d.xpos[elem_id])
            )
        )
        for name, (_, is_site, elem_id, _, target_pos) in resolved_targets.items()
    }
    converged = max(final_residuals.values()) <= tolerance

    return IKResult(
        qpos=d.qpos.copy(),
        residuals=final_residuals,
        converged=converged,
        iterations=it_count,
    )


def solve_task_target_trajectory(
    model: str | ExerciseModelBuilder | mujoco.MjModel,
    trajectories: dict[str, ArrayLike],
    *,
    q_init: ArrayLike | None = None,
    max_iterations: int = 100,
    tolerance: float = 0.01,
    damping: float = 1e-2,
    step_size: float = 0.5,
) -> list[IKResult]:
    """Solve multi-frame task-target IK with sequential warm-starting."""
    if not trajectories:
        raise ValidationError("trajectories must not be empty")

    n_frames: int | None = None
    traj_arrays: dict[str, np.ndarray] = {}
    for k, v in trajectories.items():
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValidationError(
                f"Trajectory for '{k}' must have shape (n_frames, 3), got {arr.shape}"
            )
        if n_frames is None:
            n_frames = arr.shape[0]
        elif arr.shape[0] != n_frames:
            raise ValidationError(
                f"Trajectory length mismatch: expected {n_frames}, got {arr.shape[0]}"
            )
        traj_arrays[k] = arr

    results: list[IKResult] = []
    current_q = q_init
    assert n_frames is not None
    for f in range(n_frames):
        frame_targets = {k: traj_arrays[k][f] for k in traj_arrays}
        res = solve_task_target_ik(
            model,
            frame_targets,
            q_init=current_q,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            step_size=step_size,
        )
        results.append(res)
        current_q = res.qpos

    return results


def _sync_welds(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Synchronize freejoint positions for bodies welded via equality constraints."""
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
            o1, o2 = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
            jadr = model.body_jntadr[o2]
            if jadr >= 0 and model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = model.jnt_qposadr[jadr]
                data.qpos[qadr : qadr + 3] = (
                    data.xpos[o1] + data.xmat[o1].reshape(3, 3) @ model.eq_data[i][3:6]
                )
                q1 = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(q1, data.xmat[o1])
                q2 = np.zeros(4, dtype=np.float64)
                mujoco.mju_mulQuat(q2, q1, model.eq_data[i][6:10])
                data.qpos[qadr + 3 : qadr + 7] = q2


def _clamp_joint_limits(model: mujoco.MjModel, qpos: np.ndarray) -> None:
    """Clamp limited joints within defined ranges."""
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            adr = model.jnt_qposadr[j]
            qpos[adr] = np.clip(qpos[adr], model.jnt_range[j, 0], model.jnt_range[j, 1])


def solve_ik_keyframes(
    exercise_name: str,
    n_frames: int = 50,
) -> np.ndarray:
    """Generate joint angle keyframes by interpolating phases."""
    if n_frames < 2:
        raise ValidationError(f"n_frames must be >= 2, got {n_frames}")

    objective = _lookup_objective(exercise_name)
    joint_names = objective.joint_names
    n_joints = len(joint_names)
    if not objective.phases:
        raise ValidationError(f"Exercise '{exercise_name}' has no phases defined")

    fractions = np.linspace(0.0, 1.0, n_frames)
    keyframes = np.zeros((n_frames, n_joints))
    phases = objective.phases
    phase_fractions = np.array([p.fraction for p in phases])
    phase_targets = np.array([_phase_to_array(p, joint_names) for p in phases])

    for j in range(n_joints):
        keyframes[:, j] = np.interp(fractions, phase_fractions, phase_targets[:, j])

    return keyframes


def _lookup_objective(exercise_name: str) -> ExerciseObjective:
    if exercise_name not in OBJECTIVE_REGISTRY:
        available = sorted(OBJECTIVE_REGISTRY.keys())
        raise ValidationError(
            f"Unknown exercise '{exercise_name}'. Available: {available}"
        )
    return OBJECTIVE_REGISTRY[exercise_name]


def _phase_to_array(phase: Phase, joint_names: list[str]) -> np.ndarray:
    return np.array([phase.target_joints.get(name, 0.0) for name in joint_names])
