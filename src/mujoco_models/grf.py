# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Ground reaction force (GRF) and center of pressure (CoP) measurement.

Computes resultant forces, moments about the world origin, center of pressure
on the ground plane, and free moments for each foot and for the net system
from MuJoCo contact wrenches (issue #406).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

#: Minimum vertical force [N] required to report a center of pressure.
#: Below this threshold, CoP is ill-conditioned / undefined and returned as None.
DEFAULT_MIN_FZ_N: float = 1.0


@dataclass(frozen=True)
class FootGRF:
    """Ground reaction force and center of pressure for a foot or resultant.

    Attributes:
        force_n: Resultant contact force vector [Fx, Fy, Fz] in world frame (N).
        moment_about_origin_nm: Resultant contact moment [Mx, My, Mz] about
            the world origin (N*m).
        cop_m: Center of pressure [x, y, z] on the ground plane in world frame (m),
            or None if vertical force Fz is below the threshold or no contact occurs.
        free_moment_nm: Vertical free moment [0, 0, Tz] about the CoP (N*m),
            or None if CoP is undefined.
        in_contact: True if at least one active ground contact was detected.
    """

    force_n: np.ndarray
    moment_about_origin_nm: np.ndarray
    cop_m: np.ndarray | None
    free_moment_nm: np.ndarray | None
    in_contact: bool


@dataclass(frozen=True)
class StandingGRF:
    """Ground reaction force and center of pressure breakdown for a pose.

    Attributes:
        total: Net resultant FootGRF across all ground contacts.
        per_foot: Mapping of foot body names (e.g. 'foot_l', 'foot_r') to FootGRF.
        ground_height_m: Height z of the ground plane in world frame (m).
    """

    total: FootGRF
    per_foot: dict[str, FootGRF]
    ground_height_m: float

    @property
    def total_force_n(self) -> np.ndarray:
        """Net ground reaction force vector [Fx, Fy, Fz] in world frame (N)."""
        return self.total.force_n

    @property
    def cop_m(self) -> np.ndarray | None:
        """Net center of pressure [x, y, z] on ground plane (m), or None."""
        return self.total.cop_m


def _cop(
    force: np.ndarray,
    moment: np.ndarray,
    ground_height_m: float,
    min_fz_n: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Calculate center of pressure and free moment on the plane z = ground_height_m."""
    fz = float(force[2])
    if fz < min_fz_n or not math.isfinite(fz):
        return None, None

    zg = ground_height_m
    x_cop = (zg * float(force[0]) - float(moment[1])) / fz
    y_cop = (float(moment[0]) + zg * float(force[1])) / fz
    cop = np.array([x_cop, y_cop, zg], dtype=float)

    tz = float(moment[2]) - (x_cop * float(force[1]) - y_cop * float(force[0]))
    free_moment = np.array([0.0, 0.0, tz], dtype=float)
    return cop, free_moment


def compute_grf(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ground_geom: str = "ground",
    min_fz_n: float = DEFAULT_MIN_FZ_N,
) -> StandingGRF:
    """Compute ground reaction forces and center of pressure from current contact state.

    Args:
        model: MuJoCo model.
        data: MuJoCo data containing active contacts.
        ground_geom: Name of the ground contact geometry (default: 'ground').
        min_fz_n: Minimum vertical force (N) threshold to report CoP.

    Returns:
        StandingGRF breakdown with per-foot and total reaction forces and CoP.
    """
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground_geom)
    if ground_id == -1:
        raise ValueError(f"Ground geom {ground_geom!r} not found in model")

    ground_height_m = float(data.geom_xpos[ground_id][2])

    foot_bodies = [
        model.body(b).name for b in range(model.nbody) if "foot" in model.body(b).name
    ]

    foot_forces = {name: np.zeros(3, dtype=float) for name in foot_bodies}
    foot_moments = {name: np.zeros(3, dtype=float) for name in foot_bodies}
    foot_contact_count = dict.fromkeys(foot_bodies, 0)

    total_force = np.zeros(3, dtype=float)
    total_moment = np.zeros(3, dtype=float)
    total_contacts = 0

    wrench = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        if ground_id not in (g1, g2):
            continue

        mujoco.mj_contactForce(model, data, i, wrench)
        sign = 1.0 if g1 == ground_id else -1.0
        r_frame = con.frame.reshape(3, 3)
        f_world = sign * (r_frame.T @ wrench[:3])
        tau_world = sign * (r_frame.T @ wrench[3:])
        pos = np.array(con.pos, dtype=float)
        m_world = np.cross(pos, f_world) + tau_world

        other_geom = g2 if g1 == ground_id else g1
        body_id = model.geom_bodyid[other_geom]
        body_name = model.body(body_id).name

        total_force += f_world
        total_moment += m_world
        total_contacts += 1

        if body_name not in foot_forces:
            foot_forces[body_name] = np.zeros(3, dtype=float)
            foot_moments[body_name] = np.zeros(3, dtype=float)
            foot_contact_count[body_name] = 0
        foot_forces[body_name] += f_world
        foot_moments[body_name] += m_world
        foot_contact_count[body_name] += 1

    total_cop, total_free = _cop(total_force, total_moment, ground_height_m, min_fz_n)
    total_grf = FootGRF(
        force_n=total_force,
        moment_about_origin_nm=total_moment,
        cop_m=total_cop,
        free_moment_nm=total_free,
        in_contact=(total_contacts > 0),
    )

    per_foot: dict[str, FootGRF] = {}
    for name in foot_forces:
        cop, free = _cop(
            foot_forces[name], foot_moments[name], ground_height_m, min_fz_n
        )
        per_foot[name] = FootGRF(
            force_n=foot_forces[name],
            moment_about_origin_nm=foot_moments[name],
            cop_m=cop,
            free_moment_nm=free,
            in_contact=(foot_contact_count[name] > 0),
        )

    return StandingGRF(
        total=total_grf,
        per_foot=per_foot,
        ground_height_m=ground_height_m,
    )


def standing_grf(
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    *,
    keyframe: int = 0,
    settle_s: float = 0.3,
    ground_geom: str = "ground",
    min_fz_n: float = DEFAULT_MIN_FZ_N,
) -> StandingGRF:
    """Measure standing ground reaction forces and center of pressure.

    If data is not provided, loads keyframe 0, holds the pose for settle_s,
    and measures the resulting settled GRF.

    Args:
        model: MuJoCo model.
        data: Optional pre-stepped MjData. If None, settles keyframe.
        keyframe: Keyframe index to settle when data is None (default 0).
        settle_s: Settle time in seconds before measurement (default 0.3).
        ground_geom: Ground contact geom name (default: 'ground').
        min_fz_n: Minimum vertical force (N) threshold to report CoP.

    Returns:
        StandingGRF breakdown.
    """
    if data is None:
        if model.nkey < 1:
            raise ValueError("model has no keyframe to stand in")
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, keyframe)
        if len(model.key_ctrl) > keyframe:
            data.ctrl[:] = model.key_ctrl[keyframe]
        n_steps = round(settle_s / model.opt.timestep)
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

    return compute_grf(model, data, ground_geom=ground_geom, min_fz_n=min_fz_n)
