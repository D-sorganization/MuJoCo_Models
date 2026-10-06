# SPDX-License-Identifier: MIT
"""Segment origins at the standard's test poses, measured in real MuJoCo.

The parity standard's reference forward kinematics (``topology``) gives every
segment origin at a few test poses; this measures the same origins in the
engine so conformance can compare them (Repository_Management#2011).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from mujoco_models.shared.parity._canonical import topology


def _origins(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray
) -> dict[str, list[float]]:
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return {model.body(b).name: data.xpos[b].tolist() for b in range(model.nbody)}


def pelvis_rotation(model: mujoco.MjModel, base_qpos: np.ndarray) -> list[list[float]]:
    """World rotation of the pelvis at the all-zero pose (supine bench: not I)."""
    data = mujoco.MjData(model)
    data.qpos[:] = base_qpos
    mujoco.mj_kinematics(model, data)
    return data.xmat[model.body("pelvis").id].reshape(3, 3).tolist()


def origins_at_test_poses(
    model: mujoco.MjModel,
    base_qpos: np.ndarray,
    std: dict[str, Any],
    engine_name: Callable[[str], str],
) -> dict[str, dict[str, list[float]]]:
    """``{pose: {body: world origin}}`` at each of the standard's test poses.

    A joint ANGLE is ``qpos - qpos0`` (MuJoCo ``ref``), so each pose angle is
    added to *base_qpos* (the all-zero pose). Precondition: every pose
    coordinate exists in *model* (an unknown name raises KeyError).
    """
    data = mujoco.MjData(model)
    out: dict[str, dict[str, list[float]]] = {}
    for pose, angles in topology.standard_poses(std).items():
        qpos = np.array(base_qpos, dtype=float)
        for coordinate, angle in angles.items():
            qpos[model.jnt_qposadr[model.joint(engine_name(coordinate)).id]] += angle
        out[pose] = _origins(model, data, qpos)
    return out
