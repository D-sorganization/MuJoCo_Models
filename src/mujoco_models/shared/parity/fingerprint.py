# SPDX-License-Identifier: MIT
"""Engine fingerprint: what MuJoCo actually sees for each exercise model.

Builds the exercise MJCF, loads it in the REAL engine and reports masses,
joint limits, neutral segment origins, gravity and contact friction under
schema ``model-fingerprint/v1`` so the fleet conformance checker can compare
it with the canonical standard (issue #390).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import sys
from typing import Any

import mujoco
import numpy as np

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.model_pack import manifest
from mujoco_models.optimization.exercise_objectives import get_exercise_objective
from mujoco_models.shared.parity import pose_probe
from mujoco_models.shared.parity._canonical import assemble, conformance, kinematics
from mujoco_models.shared.parity.standing import (
    GRAVITY_MPS2,
    standing_vertical_grf_n,
)

logger = logging.getLogger(__name__)

ENGINE = "mujoco"

# Engine joint/body name -> canonical name, only where they differ.  MuJoCo
# already uses the canonical names, so both tables are empty.
COORDINATE_ALIASES: dict[str, str] = {}
SEGMENT_ALIASES: dict[str, str] = {}

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)


def _zero_pose_qpos(model: mujoco.MjModel) -> np.ndarray:
    """``qpos`` of the all-zero joint-angle configuration.

    "All coordinates zero" means every hinge at ``qpos0``: MuJoCo's joint ``ref``
    is the value of the joint in the reference (XML) geometry, so the joint
    ANGLE is ``qpos - ref`` and angle 0 is ``qpos == qpos0`` (qpos=0 would
    rotate each hinge by ``-ref``).  Free joints are put at the identity pose
    (position 0, quaternion 1 0 0 0); outputs are re-based on the pelvis, so
    the root position and orientation drop out.
    """
    qpos = np.array(model.qpos0, dtype=float)
    for jid in range(model.njnt):
        if model.jnt_type[jid] == _FREE:
            adr = model.jnt_qposadr[jid]
            qpos[adr : adr + 7] = [0, 0, 0, 1, 0, 0, 0]
    return qpos


def _neutral_origins(model: mujoco.MjModel) -> dict[str, list[float]]:
    """Raw world origin of every body at the all-zero joint-angle configuration."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = _zero_pose_qpos(model)
    mujoco.mj_kinematics(model, data)
    return {model.body(b).name: data.xpos[b].tolist() for b in range(model.nbody)}


def _engine_name(canonical: str, aliases: dict[str, str]) -> str:
    """Engine-native name of a canonical coordinate or segment."""
    for engine_name, canon in aliases.items():
        if canon == canonical:
            return engine_name
    return canonical


def _rotations(
    data: mujoco.MjData, pelvis: int, segment: int
) -> tuple[list[list[float]], list[list[float]]]:
    """World rotation matrices of the pelvis and one segment."""
    return (
        data.xmat[pelvis].reshape(3, 3).tolist(),
        data.xmat[segment].reshape(3, 3).tolist(),
    )


def _measure_axis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_qpos: np.ndarray,
    *,
    joint: int,
    segment: int,
    angle: float,
) -> kinematics.Vec3:
    """Rotate one hinge by *angle* from the zero pose and measure its axis."""
    pelvis = model.body("pelvis").id
    data.qpos[:] = base_qpos
    mujoco.mj_kinematics(model, data)
    before = _rotations(data, pelvis, segment)
    data.qpos[model.jnt_qposadr[joint]] += angle
    mujoco.mj_kinematics(model, data)
    after = _rotations(data, pelvis, segment)
    return kinematics.segment_axis(before[0], before[1], after[0], after[1])


def _coordinate_axes(
    model: mujoco.MjModel, std: dict[str, Any]
) -> dict[str, kinematics.Vec3]:
    """Positive rotation axis of every coordinate, in the engine (= canonical) frame.

    Each coordinate is rotated alone by the standard's probe angle from the
    all-zero pose and the rotation of its segment relative to the pelvis is
    measured.  Coordinates or segments the model lacks are omitted, which the
    conformance check reports as ``axis.<name>.missing``.
    """
    data = mujoco.MjData(model)
    base_qpos = _zero_pose_qpos(model)
    angle = kinematics.probe_angle_rad(std)
    axes: dict[str, kinematics.Vec3] = {}
    for coordinate, segment in kinematics.axis_probes(std).items():
        joint = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            _engine_name(coordinate, COORDINATE_ALIASES),
        )
        body = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            _engine_name(segment, SEGMENT_ALIASES),
        )
        if joint < 0 or body < 0:
            logger.warning(
                "axis probe skipped: %s on %s not in model", coordinate, segment
            )
            continue
        axes[coordinate] = _measure_axis(
            model, data, base_qpos, joint=joint, segment=body, angle=angle
        )
    return axes


def _ground_friction(model: mujoco.MjModel) -> float | None:
    """Sliding friction MuJoCo uses for foot/ground (max of the two geoms)."""
    try:
        ground = float(model.geom("ground").friction[0])
    except KeyError:
        return None
    feet = [
        float(model.geom(g).friction[0])
        for g in range(model.ngeom)
        if model.geom(g).name.startswith("foot_")
        and model.geom(g).name.endswith("_contact")
    ]
    return max(ground, *feet) if feet else None


def fingerprint(exercise: str) -> dict[str, Any]:
    """Build *exercise*, load it in MuJoCo and return its fingerprint.

    Raises ValueError for an unknown exercise; any engine load failure yields
    ``loaded_in_engine=False`` with ``load_error`` set.
    """
    if exercise not in EXERCISE_REGISTRY:
        raise ValueError(f"unknown exercise {exercise!r}")
    std = conformance.load_standard()
    builder = EXERCISE_REGISTRY[exercise]()
    try:
        model = mujoco.MjModel.from_xml_string(builder.build())
    except (ValueError, RuntimeError, mujoco.FatalError) as exc:
        logger.warning("MuJoCo failed to load %s: %s", exercise, exc)
        return assemble.failed_fingerprint(ENGINE, mujoco.__version__, exercise, exc)

    masses = {model.body(b).name: float(model.body_mass[b]) for b in range(model.nbody)}
    limits = {
        model.joint(j).name: (
            float(model.jnt_range[j][0]),
            float(model.jnt_range[j][1]),
        )
        for j in range(model.njnt)
        if model.jnt_type[j] == _HINGE
    }
    pelvis_joint = model.body_jntadr[model.body("pelvis").id]
    base = _zero_pose_qpos(model)
    extras: dict[str, Any] = {}
    if not builder.uses_barbell:
        human_mass = sum(
            m for n, m in masses.items() if n in conformance.expected_segments(std)
        )
        extras["standing_vertical_grf_n"] = standing_vertical_grf_n(model)
        extras["standing_weight_n"] = human_mass * GRAVITY_MPS2
    return assemble.assemble_fingerprint(
        engine=ENGINE,
        engine_version=mujoco.__version__,
        exercise=exercise,
        std=std,
        root_joint="free" if model.jnt_type[pelvis_joint] == _FREE else "fixed",
        gravity_engine=model.opt.gravity.tolist(),
        segment_masses_kg=masses,
        coordinate_limits_rad=limits,
        segment_origins_engine_m=_neutral_origins(model),
        coordinate_axes_engine=_coordinate_axes(model, std),
        pelvis_rotation_engine=pose_probe.pelvis_rotation(model, base),
        segment_origins_test_poses_engine_m=pose_probe.origins_at_test_poses(
            model, base, std, lambda c: _engine_name(c, COORDINATE_ALIASES)
        ),
        capabilities=assemble.capabilities_from_manifest(manifest(), std),
        coordinate_aliases=COORDINATE_ALIASES,
        segment_aliases=SEGMENT_ALIASES,
        ground_friction=_ground_friction(model),
        phase_count=get_exercise_objective(exercise).n_phases,
        extras=extras,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--exercise X | --all`` with ``--out DIR``."""
    exercises = [e["id"] for e in manifest()["exercises"]]
    return assemble.run_fingerprint_cli(argv, fingerprint, exercises, ENGINE)


if __name__ == "__main__":
    sys.exit(main())
