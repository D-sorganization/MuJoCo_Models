# SPDX-License-Identifier: MIT
"""Start poses of the barbell lifts, read from the vendored parity standard.

UpstreamDrift#11743 (LIFT-3), MuJoCo_Models#407/#409: each lift's generated
model starts at the standard's ``start`` via-pose -- joint coordinates, pelvis
root position and orientation, and bar centre -- never at pack-local literals.
MuJoCo uses the canonical coordinate names, so the via-pose maps onto the
joints without renaming.

Via-pose frame (``pose_frame``): canonical axes (X forward, Y left, Z up),
origin on the floor at mid-foot.  Positions are for the standard height and
scale linearly with height; joint angles are radians and do not scale.  The
root orientation is ``Rz(yaw) Ry(pitch) Rx(roll)``.

Design by Contract:
    - Preconditions: the exercise has a ``start`` via-pose in the standard and
      the height is positive; violations raise ValueError.
    - Postconditions: the hand/torso-to-bar welds hold exactly at the keyframe
      (their ``relpose`` is measured from the posed model).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_models.shared.body.foot_seating import place_foot_soles
from mujoco_models.shared.parity._canonical import spec, topology
from mujoco_models.shared.parity.standard import STANDARD

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

BAR_BODY = "barbell_shaft"
# The ankle sits this term above the floor in the via-pose frame (pose_frame).
_ANKLE_TERM: str = STANDARD["pose_frame"]["ankle_height_term"]


@dataclass(frozen=True)
class StartPose:
    """A lift's start via-pose placed for one body height."""

    coordinates: dict[str, float]
    root_position: Vec3
    root_quat: Quat
    bar_position: Vec3
    ankle_height: float


def _rpy_quat(roll: float, pitch: float, yaw: float) -> Quat:
    """Quaternion (w, x, y, z) of ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    )


def _scaled(values: list[float], scale: float) -> Vec3:
    x, y, z = (float(v) * scale for v in values)
    return (x, y, z)


def start_pose(exercise: str, height_m: float) -> StartPose:
    """The spec ``start`` via-pose of *exercise* for a body *height_m* tall.

    Raises:
        ValueError: *exercise* has no start via-pose, or *height_m* <= 0.
    """
    if not height_m > 0:
        raise ValueError(f"height_m must be positive, got {height_m}")
    poses = spec.exercise_spec(STANDARD, exercise).get("via_poses", {})
    if "start" not in poses:
        raise ValueError(f"{exercise!r} has no start via-pose in the standard")
    pose = poses["start"]
    scale = height_m / STANDARD["anthropometrics"]["height_m"]
    roll, pitch, yaw = (float(a) for a in pose["root"]["orientation_rpy_rad"])
    return StartPose(
        coordinates=spec.pose_coordinates(STANDARD, exercise, "start"),
        root_position=_scaled(pose["root"]["position_m"], scale),
        root_quat=_rpy_quat(roll, pitch, yaw),
        bar_position=_scaled(pose["bar_position_m"], scale),
        ankle_height=topology.dimension(STANDARD, _ANKLE_TERM) * scale,
    )


def plates_per_side_kg(exercise: str) -> float:
    """The standard load's plate mass on each sleeve (kg)."""
    return float(spec.exercise_spec(STANDARD, exercise)["load"]["plates_per_side_kg"])


def place_start_pose(worldbody: ET.Element, pose: StartPose) -> None:
    """Put the pelvis (root body) at the via-pose root position and orientation
    and the soles ``pose.ankle_height`` below the ankles, on the floor.

    Raises:
        ValueError: *worldbody* has no ``pelvis`` body.
    """
    pelvis = worldbody.find("body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("worldbody has no pelvis body to place")
    pelvis.set("pos", " ".join(f"{v:.6f}" for v in pose.root_position))
    pelvis.set("quat", " ".join(f"{q:.9f}" for q in pose.root_quat))
    place_foot_soles(worldbody, pose.ankle_height)


def _relpose(data: mujoco.MjData, body1: int, body2: int) -> list[float]:
    """Pose of *body2* in *body1*'s frame: ``[x, y, z, qw, qx, qy, qz]``."""
    rot1 = data.xmat[body1].reshape(3, 3)
    pos = rot1.T @ (data.xpos[body2] - data.xpos[body1])
    inv1 = np.zeros(4)
    mujoco.mju_negQuat(inv1, data.xquat[body1])
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, inv1, data.xquat[body2])
    return [*pos.tolist(), *quat.tolist()]


def weld_bar_at_keyframe(root: ET.Element, xml: str) -> None:
    """Set the ``relpose`` of every lifter-to-bar weld to its keyframe value.

    *xml* is *root* serialised.  The welds then hold exactly at the start pose
    instead of at ``qpos0``, so the solver does not start by fighting them.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    for weld in root.iter("weld"):
        if weld.get("body2") != BAR_BODY or weld.get("body1", "").startswith("barbell"):
            continue
        rel = _relpose(data, model.body(weld.get("body1")).id, model.body(BAR_BODY).id)
        weld.set("relpose", " ".join(f"{v:.9f}" for v in rel))
