# SPDX-License-Identifier: MIT
"""Standing ground-reaction-force measurement for the parity fingerprint.

Loads keyframe 0, holds the pose with the position servos for a settle
period, then averages the summed world-vertical foot/floor contact force.
For a body at rest that force equals the lifter's weight (issue #390).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging

import mujoco
import numpy as np

from mujoco_models.grf import compute_grf

logger = logging.getLogger(__name__)

SETTLE_S = 0.3
AVERAGE_S = 0.05
GRAVITY_MPS2 = 9.80665


def _vertical_floor_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Sum the world-vertical force the floor exerts on the feet (newtons)."""
    return float(compute_grf(model, data).total.force_n[2])


def standing_vertical_grf_n(
    model: mujoco.MjModel,
    settle_s: float = SETTLE_S,
    average_s: float = AVERAGE_S,
) -> float:
    """Return the mean vertical floor force after holding the keyframe pose.

    Precondition: the model has at least one keyframe and a ``ground`` geom.
    """
    if model.nkey < 1:
        raise ValueError("model has no keyframe to stand in")
    if not 0 < average_s < settle_s:
        raise ValueError("require 0 < average_s < settle_s")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
    n_steps = round(settle_s / model.opt.timestep)
    n_avg = round(average_s / model.opt.timestep)
    samples: list[float] = []
    for step in range(n_steps):
        mujoco.mj_step(model, data)
        if step >= n_steps - n_avg:
            samples.append(_vertical_floor_force(model, data))
    grf = float(np.mean(samples))
    logger.debug("standing GRF %.1f N over %d samples", grf, len(samples))
    return grf
