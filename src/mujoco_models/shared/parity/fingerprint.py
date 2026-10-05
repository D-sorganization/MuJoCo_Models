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

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.model_pack import manifest
from mujoco_models.optimization.exercise_objectives import get_exercise_objective
from mujoco_models.shared.parity._canonical import assemble, conformance
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


def _neutral_origins(model: mujoco.MjModel) -> dict[str, list[float]]:
    """Raw world origin of every body at the all-zero joint-angle configuration.

    "All coordinates zero" means every hinge at ``qpos0``: MuJoCo's joint ``ref``
    is the value of the joint in the reference (XML) geometry, so the joint
    ANGLE is ``qpos - ref`` and angle 0 is ``qpos == qpos0`` (qpos=0 would
    rotate each hinge by ``-ref``).  Free joints are put at the identity pose
    (position 0, quaternion 1 0 0 0); the adapter output is re-based on the
    pelvis, so the root position drops out.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    for jid in range(model.njnt):
        if model.jnt_type[jid] == _FREE:
            adr = model.jnt_qposadr[jid]
            data.qpos[adr : adr + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(model, data)
    return {model.body(b).name: data.xpos[b].tolist() for b in range(model.nbody)}


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
