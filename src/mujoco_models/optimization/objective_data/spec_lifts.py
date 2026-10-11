# SPDX-License-Identifier: MIT
"""Lift objectives built from the vendored exercise spec (standard v2.0).

UpstreamDrift#11742 (LIFT-2), Repository_Management#2060 (Option A): the
phase names, the phase timing (``t_frac``) and the start, catch and lockout
via-poses come from ``_canonical/biomech_parity_standard.json``, never from
local tables.  The spec poses only those anchors.  Every other phase target is
the linear blend of the neighbouring anchors at the phase's ``t_frac`` (held
at the last anchor after it), so this pack adds no pose content of its own; a
task-target IK solver (#403) will refine the in-between phases.

Design by Contract:
    - Preconditions: the exercise has via-poses in the standard and every
      anchored phase exists in its phase list; violations raise ValueError.
    - Postconditions: phase names and fractions equal the spec's; each
      anchored phase targets every canonical coordinate (radians, unlisted
      coordinates 0 per ``pose_frame``).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

from bisect import bisect_right
from typing import Any

from mujoco_models.optimization.exercise_objectives import ExerciseObjective, Phase
from mujoco_models.shared.parity._canonical import conformance, spec
from mujoco_models.shared.parity.standard import STANDARD

# The named phase each via-pose anchors.  Standard v2.0 names the poses but does
# not bind them to phases, so the binding is recorded here (one entry per lift).
VIA_POSE_PHASES: dict[str, dict[str, str]] = {
    "squat": {"start": "standing", "catch": "bottom", "lockout": "lockout"},
    "deadlift": {"start": "setup", "lockout": "lockout"},
    "bench_press": {"start": "unrack", "catch": "chest_touch", "lockout": "lockout"},
    "snatch": {"start": "setup", "catch": "catch", "lockout": "recovery"},
    "clean_and_jerk": {
        "start": "setup",
        "catch": "rack_catch",
        "lockout": "overhead_recovery",
    },
}

Pose = dict[str, float]


def via_pose(exercise: str, pose: str) -> Pose:
    """Joint angles (rad) of one via-pose over every canonical coordinate.

    Coordinates the spec does not list are 0 (``pose_frame``).  MuJoCo uses
    the canonical coordinate names, so no renaming is needed.
    """
    out = dict.fromkeys(conformance.expected_coordinates(STANDARD), 0.0)
    out.update(spec.pose_coordinates(STANDARD, exercise, pose))
    return out


def _blend(knots: list[tuple[float, Pose]], t: float) -> Pose:
    """Piecewise-linear pose at *t* between anchor knots, clamped at the ends."""
    times = [k[0] for k in knots]
    i = bisect_right(times, t) - 1
    if i < 0:
        return dict(knots[0][1])
    if i >= len(knots) - 1 or times[i] == t:
        return dict(knots[i][1])
    (t0, a), (t1, b) = knots[i], knots[i + 1]
    w = (t - t0) / (t1 - t0)
    return {name: a[name] + w * (b[name] - a[name]) for name in a}


def _anchor_knots(
    exercise: str, phases: list[dict[str, Any]]
) -> list[tuple[float, Pose]]:
    """(t_frac, pose) for every anchored phase, ordered by time."""
    t_of = {str(p["name"]): float(p["t_frac"]) for p in phases}
    anchors = VIA_POSE_PHASES[exercise]
    missing = sorted(set(anchors.values()) - set(t_of))
    if missing:
        raise ValueError(f"{exercise}: anchored phases {missing} not in the spec")
    if anchors["start"] != phases[0]["name"]:
        raise ValueError(f"{exercise}: the start via-pose must anchor the first phase")
    knots = [(t_of[phase], via_pose(exercise, pose)) for pose, phase in anchors.items()]
    return sorted(knots, key=lambda k: k[0])


def lift_objective(
    exercise: str, *, name: str, bar_path_constraint: str, balance_mode: str
) -> ExerciseObjective:
    """Build the objective of a barbell lift from the vendored spec.

    Raises:
        ValueError: *exercise* has no via-poses in the standard or no anchor
            binding, or an anchored phase is missing from the spec.
    """
    if exercise not in VIA_POSE_PHASES:
        raise ValueError(
            f"unknown exercise {exercise!r}; spec lifts: {sorted(VIA_POSE_PHASES)}"
        )
    phases = spec.exercise_spec(STANDARD, exercise)["phases"]
    knots = _anchor_knots(exercise, phases)
    return ExerciseObjective(
        name=name,
        start_pose=via_pose(exercise, "start"),
        end_pose=via_pose(exercise, "lockout"),
        phases=[
            Phase(str(p["name"]), float(p["t_frac"]), _blend(knots, float(p["t_frac"])))
            for p in phases
        ],
        bar_path_constraint=bar_path_constraint,
        balance_mode=balance_mode,
    )
