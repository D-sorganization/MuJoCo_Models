# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Seat the foot contact boxes at the contact-law equilibrium depth (issue #447).

Why soles cannot sit exactly on the floor
-----------------------------------------
MuJoCo contacts are soft constraints.  The solver converts a penetration
``r`` (negative signed distance) into a reference acceleration
``a_ref = -b*v - k*d(r)*r`` and a constraint force
``f = d(r) * (a_ref - a_0) / A`` where

* ``(k, b)`` come from ``solref`` (default time constant 0.02 s, damping
  ratio 1: ``k = 1/(d_max^2 * T^2 * zeta^2)``, ``b = 2/(d_max * T)``),
* ``d(r)`` is the impedance from ``solimp`` (default ``0.9 0.95 0.001 0.5 2``:
  it rises from 0.9 at zero depth to 0.95 at a depth of 1 mm), and
* ``A`` and ``a_0`` are the contact-space inverse inertia and the
  unconstrained acceleration (gravity, servo forces).

At zero penetration ``a_ref`` is zero, so the foot-floor force is a small
fraction of the weight and the body drops until ``f`` supports it.  A model
that is built with its soles exactly on the floor therefore settles (or
bounces) at t=0, and a 0.5 mm root lift removes every contact.

The static equilibrium is the depth ``delta`` at which the contact carries
the supported weight (lifter plus bar) so the root's vertical acceleration at
the keyframe is zero.  Since ``d(r)`` varies with depth and ``A`` depends on the
whole kinematic chain, ``delta`` is a fixed point that is not worth re-deriving
in closed form; it is solved against the compiled model, so it always tracks
the ``solref``/``solimp``, the mass distribution and the gravity the model
really uses.  Measured values are 0.78-0.86 mm for the four standing lifts at
any load tried (100 to 260 kg total): ``A`` scales with the supported mass, so
the depth is nearly mass independent.

Why not balance the instantaneous ground reaction against ``m*g`` instead:
the keyframe is not in servo equilibrium (joint servos are soft, issue #407),
so limb accelerations at t=0 make the contact force differ from ``m*g``.
Forcing it to ``m*g`` seats the soles 1.1 mm (100 kg) to 6.9 mm (200 kg) deep
and the root then rises instead of settling; the root-acceleration criterion
is load independent.

The joints are held rigid for the solve (``hold_hinges_rigid``): a crouched
start (deadlift setup, UpstreamDrift#11743) is far from servo equilibrium, so
with live servos the root accelerates down at nearly ``g`` at any sole depth.
With rigid hinges the criterion asks only whether the contact carries the
whole lifter and bar.  The remaining start-up sag is servo compliance, not
contact, and is tracked by MuJoCo_Models#460.

Preconditions: the model has a keyframe, the root is the first free joint
(its z translation is DOF 2) and the foot boxes sit flat at the keyframe.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from mujoco_models.shared.body.body_helpers import foot_contact_pos

logger = logging.getLogger(__name__)

ROOT_VERTICAL_DOF = 2  # pelvis free joint, z translation
_FOOT_GEOM_NAMES = ("foot_l_contact", "foot_r_contact")
_MAX_DEPTH_M = 0.02  # bracket upper bound; far beyond any soft-contact equilibrium
_DEPTH_TOLERANCE_M = 1e-9
_MAX_ITERATIONS = 80
_RIGID_ARMATURE = 1e9  # kg m^2; freezes hinges for the equilibrium solve only


def _foot_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in _FOOT_GEOM_NAMES
    ]
    present = [i for i in ids if i >= 0]
    if not present:
        raise ValueError("model has no foot_l_contact / foot_r_contact geom")
    return present


def _sole_depth(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int]
) -> float:
    """Return the deepest sole penetration below z=0 (positive = into the floor)."""
    depths = []
    for gid in geom_ids:
        rot = data.geom_xmat[gid].reshape(3, 3)
        low = data.geom_xpos[gid][2] - np.abs(rot[2]) @ model.geom_size[gid]
        depths.append(-float(low))
    return max(depths)


def _root_acceleration(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int], depth: float
) -> float:
    """Root vertical acceleration at the keyframe with the soles ``depth`` deep.

    The whole scene (every free body: lifter and bar) is translated down, so
    the welds stay satisfied and the compiled collision bounds stay valid
    (moving only the foot geoms leaves MuJoCo's body bounding volumes stale).
    """
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    current = _sole_depth(model, data, geom_ids)
    free = model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
    data.qpos[model.jnt_qposadr[free] + 2] -= depth - current
    mujoco.mj_forward(model, data)
    if abs(_sole_depth(model, data, geom_ids) - depth) > 1e-7:
        raise ValueError("foot contact boxes are not flat at the keyframe")
    return float(data.qacc[ROOT_VERTICAL_DOF])


def hold_hinges_rigid(model: mujoco.MjModel) -> None:
    """Make every hinge effectively rigid in this solve-only model copy.

    A crouched start (deadlift setup, UpstreamDrift#11743) is not in servo
    equilibrium: at t=0 the position servos exert no torque, so the trunk
    sags and the root accelerates down at almost ``g`` whatever the sole
    depth.  A huge armature freezes the joints, so the root acceleration
    measures only whether the contact carries the whole (rigid) lifter.
    """
    hinges = model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    model.dof_armature[model.jnt_dofadr[hinges]] = _RIGID_ARMATURE


def equilibrium_foot_penetration(xml: str) -> float:
    """Return the sole depth (m) at which the foot-floor contact carries the weight.

    Bisects on the root's vertical acceleration at keyframe 0 using the
    solver's own contact law (see the module docstring), so the result is
    consistent with the model's ``solref``/``solimp`` and supported weight and
    does not depend on how deep the soles are currently seated.

    Postcondition: ``0 < result < 0.02`` and the root vertical acceleration at
    that depth is zero to within the bisection tolerance.

    Raises:
        ValueError: no keyframe, no foot geoms, feet not flat, or no
            equilibrium inside the search bracket.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nkey < 1:
        raise ValueError("model needs a keyframe to derive the foot equilibrium")
    hold_hinges_rigid(model)
    data = mujoco.MjData(model)
    geom_ids = _foot_geom_ids(model)

    def excess(depth: float) -> float:
        return _root_acceleration(model, data, geom_ids, depth)

    low, high = 0.0, _MAX_DEPTH_M
    if excess(low) >= 0.0 or excess(high) <= 0.0:
        raise ValueError("no foot-floor equilibrium in [0, 20] mm of penetration")
    for _ in range(_MAX_ITERATIONS):
        if high - low < _DEPTH_TOLERANCE_M:
            break
        mid = 0.5 * (low + high)
        if excess(mid) < 0.0:
            low = mid
        else:
            high = mid
    depth = 0.5 * (low + high)
    logger.debug("Equilibrium foot penetration %.4f mm", depth * 1e3)
    return depth


def foot_seating_offset(xml: str) -> float:
    """Metres to lower the soles so they sit at the equilibrium depth.

    The equilibrium depth minus the deepest sole's current depth at the
    keyframe: a posed start (UpstreamDrift#11743) places the soles on the
    floor only to the via-pose rounding (tens of micrometres), so the offset
    absorbs that residual instead of assuming flush soles.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    current = _sole_depth(model, data, _foot_geom_ids(model))
    return equilibrium_foot_penetration(xml) - current


def lower_foot_contact_geoms(root: ET.Element, depth: float) -> None:
    """Move every ``foot_*_contact`` geom ``depth`` metres down its body z axis.

    Precondition: ``depth >= 0``.  The soles must be flat on the floor before
    the call (they are built with the sole exactly at z=0).
    """
    if depth < 0.0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    for geom in root.iter("geom"):
        if geom.get("name") not in _FOOT_GEOM_NAMES:
            continue
        x, y, z = (float(v) for v in geom.get("pos", "0 0 0").split())
        geom.set("pos", f"{x:g} {y:g} {z - depth:.9f}")


def place_foot_soles(root: ET.Element, ankle_height: float) -> None:
    """Re-place every foot contact box so its sole is ``ankle_height`` below
    the ankle (foot body origin).

    A posed start (UpstreamDrift#11743) puts the ankle where the parity
    standard's ``pose_frame`` does, ``foot.length`` above the floor, so the
    sole must sit that far below it to stand on the floor.

    Raises:
        ValueError: ``ankle_height`` is not positive.
    """
    if not ankle_height > 0:
        raise ValueError(f"ankle_height must be positive, got {ankle_height}")
    for geom in root.iter("geom"):
        if geom.get("name") in _FOOT_GEOM_NAMES:
            geom.set("pos", foot_contact_pos(ankle_height))
