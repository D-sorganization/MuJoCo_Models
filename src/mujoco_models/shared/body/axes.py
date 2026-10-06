# SPDX-License-Identifier: MIT
"""Canonical joint axes for the human body (MuJoCo_Models#410).

MuJoCo's world frame is the fleet canonical frame (X forward, Y left, Z up) and
every human body frame is world-aligned at the all-zero pose, so a hinge's
``axis`` attribute is its canonical rotation axis.  The axes themselves are
read from the vendored parity standard (kinematics block); nothing here is a
hand-written literal, so the model and the fleet standard cannot drift apart.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

from mujoco_models.exceptions import ValidationError
from mujoco_models.shared.parity._canonical import kinematics
from mujoco_models.shared.parity.standard import STANDARD

Vec3 = tuple[float, float, float]

# Left side sits at +Y, right side at -Y (canonical Y is left).
SIDE_SIGN: dict[str, float] = {"l": 1.0, "r": -1.0}

_AXES: dict[str, Vec3] = kinematics.expected_axes(STANDARD)


def joint_axis(coordinate: str) -> Vec3:
    """Return the canonical unit axis of *coordinate* (e.g. ``"hip_l_flex"``).

    Precondition: *coordinate* is a coordinate of the standard, sides expanded.
    Raises ValidationError otherwise.
    """
    try:
        return _AXES[coordinate]
    except KeyError:
        raise ValidationError(f"unknown coordinate {coordinate!r}") from None
