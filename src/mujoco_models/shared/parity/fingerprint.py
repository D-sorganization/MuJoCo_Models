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

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.model_pack import manifest
from mujoco_models.optimization.exercise_objectives import get_exercise_objective
from mujoco_models.shared.parity._canonical import conformance
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


def _canon(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def _human_bodies(model: mujoco.MjModel, expected: set[str]) -> dict[str, int]:
    """Map canonical segment name -> body id for the human segments."""
    out: dict[str, int] = {}
    for bid in range(model.nbody):
        canon = _canon(model.body(bid).name, SEGMENT_ALIASES)
        if canon in expected:
            out[canon] = bid
    return out


def _coordinates(
    model: mujoco.MjModel, expected: dict[str, tuple[float, float]]
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for jid in range(model.njnt):
        canon = _canon(model.joint(jid).name, COORDINATE_ALIASES)
        if model.jnt_type[jid] == _HINGE and canon in expected:
            lo, hi = (float(v) for v in model.jnt_range[jid])
            out[canon] = {"limits_rad": [lo, hi]}
    return out


def _neutral_origins(
    model: mujoco.MjModel, bodies: dict[str, int], std: dict[str, Any]
) -> dict[str, list[float]]:
    """World origin of each human body at qpos=0 (free joints at identity)."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    for jid in range(model.njnt):
        if model.jnt_type[jid] == _FREE:
            data.qpos[model.jnt_qposadr[jid] + 3] = 1.0
    mujoco.mj_kinematics(model, data)
    canon = {
        seg: conformance.to_canonical(std, ENGINE, data.xpos[bid].tolist())
        for seg, bid in bodies.items()
    }
    root = np.array(canon["pelvis"])
    return {s: (np.array(p) - root).tolist() for s, p in canon.items()}


def _ground_friction(model: mujoco.MjModel) -> float | None:
    """Sliding friction MuJoCo uses for foot/ground (max of the two geoms)."""
    try:
        ground = model.geom("ground").friction[0]
    except KeyError:
        return None
    feet = [
        float(model.geom(g).friction[0])
        for g in range(model.ngeom)
        if model.geom(g).name.endswith("_contact")
        and model.geom(g).name.startswith("foot_")
    ]
    return max(float(ground), *feet) if feet else None


def _capabilities() -> dict[str, str]:
    caps = manifest().get("capabilities", {})
    return {key: str(entry["level"]) for key, entry in caps.items()}


def _failed(exercise: str, std: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "schema": conformance.FINGERPRINT_SCHEMA,
        "engine": ENGINE,
        "engine_version": mujoco.__version__,
        "exercise": exercise,
        "standard_sha256": conformance.standard_sha256_for(std),
        "loaded_in_engine": False,
        "load_error": str(exc),
    }


def _load_model(exercise: str) -> mujoco.MjModel:
    if exercise not in EXERCISE_REGISTRY:
        raise ValueError(f"unknown exercise {exercise!r}")
    xml = EXERCISE_REGISTRY[exercise]().build()
    return mujoco.MjModel.from_xml_string(xml)


def fingerprint(exercise: str) -> dict[str, Any]:
    """Build *exercise*, load it in MuJoCo and return its fingerprint.

    Raises ValueError for an unknown exercise; any engine load failure yields
    ``loaded_in_engine=False`` with ``load_error`` set.
    """
    std = conformance.load_standard()
    if exercise not in EXERCISE_REGISTRY:
        raise ValueError(f"unknown exercise {exercise!r}")
    try:
        model = _load_model(exercise)
    except (ValueError, RuntimeError, mujoco.FatalError) as exc:
        logger.warning("MuJoCo failed to load %s: %s", exercise, exc)
        return _failed(exercise, std, exc)

    segs = set(conformance.expected_segments(std))
    bodies = _human_bodies(model, segs)
    masses = {s: float(model.body_mass[b]) for s, b in bodies.items()}
    pelvis_root = model.body_jntadr[bodies["pelvis"]]
    fp: dict[str, Any] = {
        "schema": conformance.FINGERPRINT_SCHEMA,
        "engine": ENGINE,
        "engine_version": mujoco.__version__,
        "exercise": exercise,
        "standard_sha256": conformance.standard_sha256_for(std),
        "loaded_in_engine": True,
        "load_error": None,
        "root_joint": "free" if model.jnt_type[pelvis_root] == _FREE else "fixed",
        "gravity_canonical": list(
            conformance.to_canonical(std, ENGINE, model.opt.gravity.tolist())
        ),
        "body_mass_kg": sum(masses.values()),
        "segments": {s: {"mass_kg": m} for s, m in masses.items()},
        "coordinates": _coordinates(model, conformance.expected_coordinates(std)),
        "segment_origins_neutral_m": _neutral_origins(model, bodies, std),
        "capabilities": _capabilities(),
        "phase_count": get_exercise_objective(exercise).n_phases,
    }
    friction = _ground_friction(model)
    if friction is not None:
        fp["ground_friction"] = friction
    if not EXERCISE_REGISTRY[exercise]().uses_barbell:
        fp["standing_vertical_grf_n"] = standing_vertical_grf_n(model)
        fp["standing_weight_n"] = fp["body_mass_kg"] * GRAVITY_MPS2
    return _check_postconditions(fp)


def _check_postconditions(fp: dict[str, Any]) -> dict[str, Any]:
    if fp["schema"] != conformance.FINGERPRINT_SCHEMA:
        raise ValueError("fingerprint schema mismatch")
    if not np.isfinite(fp["body_mass_kg"]) or fp["body_mass_kg"] <= 0:
        raise ValueError("fingerprint body_mass_kg must be finite and positive")
    return fp


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--exercise X | --all`` with optional ``--out DIR``."""
    parser = argparse.ArgumentParser(description="MuJoCo model fingerprint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exercise", choices=sorted(EXERCISE_REGISTRY))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    names = [e["id"] for e in manifest()["exercises"]] if args.all else [args.exercise]
    for name in names:
        text = json.dumps(fingerprint(name), indent=2, sort_keys=True)
        if args.out is None:
            sys.stdout.write(text + "\n")
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{ENGINE}_{name}.json").write_text(text + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
