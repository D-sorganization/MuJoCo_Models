# SPDX-License-Identifier: MIT
"""Barbell grip and orientation helpers for MuJoCo exercise models."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping

from mujoco_models.shared.body.axes import SIDE_SIGN
from mujoco_models.shared.utils.mjcf_helpers import add_weld_constraint

HandQuat = tuple[float, float, float, float]

_IDENTITY_QUAT: HandQuat = (1.0, 0.0, 0.0, 0.0)


def _quat_rotate_vector(
    quat: HandQuat, vec: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate *vec* by *quat* (``w, x, y, z``): the quaternion sandwich product.

    Uses the standard expansion ``v' = v + 2w(q_xyz x v) + 2 q_xyz x (q_xyz x v)``
    so callers never need a separate rotation-matrix path for a quaternion
    they already hold (MuJoCo_Models#440).
    """
    w, qx, qy, qz = quat
    vx, vy, vz = vec
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w * tx + (qy * tz - qz * ty),
        vy + w * ty + (qz * tx - qx * tz),
        vz + w * tz + (qx * ty - qy * tx),
    )


def _hand_quat_for_side(
    hand_quat: HandQuat | Mapping[str, HandQuat], side: str
) -> HandQuat:
    """Resolve a hand-orientation argument that may be per-side.

    ``hand_quat`` is either a single quaternion applied to both hands
    (bench press: both hands share the same tilt) or a mapping with one
    quaternion per side (grip exercises, where shoulder abduction tilts
    the two hands oppositely, MuJoCo_Models#440).
    """
    if isinstance(hand_quat, Mapping):
        return hand_quat[side]
    return hand_quat


def barbell_relpose_for_hand(
    side: str,
    grip_width: float | None = None,
    grip_offset: tuple[float, ...] | None = None,
    hand_quat: HandQuat = _IDENTITY_QUAT,
) -> tuple[float, ...] | None:
    """Return the weld relpose (bar relative to the hand frame) for one hand.

    ``grip_offset`` wins when provided. Otherwise ``grip_width`` sizes a
    lateral (Y) offset in the WORLD frame: the left hand is at +Y, so the
    bar centre sits at ``-grip_width`` in the world Y direction from it,
    and at ``+grip_width`` from the right hand's. That world offset is
    then rotated into the hand's own frame by ``hand_quat`` -- the bar's
    orientation relative to the hand -- so a tilted hand (shoulder
    abduction without a counter-rotating wrist, MuJoCo_Models#440) still
    gets a relpose that keeps the bar at its intended world position.
    ``hand_quat`` defaults to identity for a world-aligned hand.
    """
    if grip_offset is not None:
        return grip_offset
    if grip_width is None:
        return None

    world_offset = (0.0, -SIDE_SIGN[side] * grip_width, 0.0)
    pos = _quat_rotate_vector(hand_quat, world_offset)
    return (*pos, *hand_quat)


def attach_barbell_to_hand(
    equality: ET.Element, *, side: str, relpose: tuple[float, ...] | None
) -> None:
    """Write one hand weld to the shared equality section."""
    add_weld_constraint(
        equality,
        name=f"barbell_to_hand_{side}",
        body1=f"hand_{side}",
        body2="barbell_shaft",
        relpose=relpose,
    )


def attach_barbell_to_hands(
    equality: ET.Element,
    *,
    grip_width: float | None = None,
    grip_offset: tuple[float, ...] | None = None,
    hand_quat: HandQuat | Mapping[str, HandQuat] = _IDENTITY_QUAT,
) -> None:
    """Weld barbell shaft to both hands (DRY helper for subclasses).

    Parameters
    ----------
    equality : ET.Element
        The ``<equality>`` section of the MJCF model.
    grip_width : float or None
        Distance from the barbell shaft center to each hand along the Y-axis
        (the shaft's own axis). If None, the initial pose determines it.
    grip_offset : tuple or None
        Optional 7-element relative pose (x y z qw qx qy qz) for the grip
        offset from the hand to the barbell shaft.
    hand_quat : tuple or mapping
        Bar orientation in the hand frame, for exercises whose hands are
        not world-aligned at the keyframe (see ``_barbell_relpose_for_hand``).
        Either one quaternion shared by both hands (bench press), or a
        ``{"l": ..., "r": ...}`` mapping for hands tilted oppositely
        (MuJoCo_Models#440). With neither ``grip_width`` nor
        ``grip_offset`` a spec lift's welds get their ``relpose`` from the
        posed keyframe (see ``spec_start.weld_bar_at_keyframe``).
    """
    for side in ("l", "r"):
        relpose = barbell_relpose_for_hand(
            side,
            grip_width=grip_width,
            grip_offset=grip_offset,
            hand_quat=_hand_quat_for_side(hand_quat, side),
        )
        attach_barbell_to_hand(equality, side=side, relpose=relpose)
