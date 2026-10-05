# SPDX-License-Identifier: MIT
"""Cross-repo parity standard — canonical biomechanical parameters.

Every value is computed from the vendored bundle
(``_canonical/biomech_parity_standard.json``), which is a byte-identical copy
of ``Repository_Management/shared_scripts/model_parity``.  Never hand-edit a
number here; change the bundle upstream and re-sync.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import math
from typing import Any

from mujoco_models.shared.parity._canonical import conformance


def _rad(deg: float) -> float:
    """Convert degrees to radians."""
    return math.radians(deg)


STANDARD: dict[str, Any] = conformance.load_standard()
_ANTHRO: dict[str, Any] = STANDARD["anthropometrics"]

# ── Anthropometric Standard (Winter 2009) ──────────────────────────
STANDARD_BODY_MASS: float = _ANTHRO["body_mass_kg"]  # kg
STANDARD_HEIGHT: float = _ANTHRO["height_m"]  # m

# Single segment table (mass_frac / length_frac / radius_frac) from the bundle.
SEGMENT_TABLE: dict[str, dict[str, float]] = {
    name: {
        "mass_frac": seg["mass_frac"],
        "length_frac": seg["length_frac"],
        "radius_frac": seg["radius_frac"],
    }
    for name, seg in _ANTHRO["segments"].items()
}

SEGMENT_MASS_FRACTIONS: dict[str, float] = {
    name: seg["mass_frac"] for name, seg in SEGMENT_TABLE.items()
}

SEGMENT_LENGTH_FRACTIONS: dict[str, float] = {
    name: seg["length_frac"] for name, seg in SEGMENT_TABLE.items()
}

# ── Joint Limit Standard (radians) ────────────────────────────────
# Keyed by the side-less coordinate name (``hip_flex`` for ``hip_l_flex``).
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    name.replace("_{side}", ""): (math.radians(lo), math.radians(hi))
    for name, (lo, hi) in (
        (c["name"], c["limits_deg"]) for c in STANDARD["coordinates"]
    )
}

# ── Barbell Standard (IWF/IPF) ────────────────────────────────────
_BAR: dict[str, float] = STANDARD["barbell"]["mens"]
MENS_BARBELL: dict[str, float] = {
    "total_length": _BAR["total_length_m"],
    "shaft_length": _BAR["shaft_length_m"],
    "shaft_diameter": _BAR["shaft_diameter_m"],
    "sleeve_diameter": _BAR["sleeve_diameter_m"],
    "bar_mass": _BAR["bar_mass_kg"],
}

# ── Contact Standard ──────────────────────────────────────────────
FOOT_CONTACT_DIMS: dict[str, float] = dict(STANDARD["contact"]["foot_box_m"])
GROUND_FRICTION: dict[str, float] = dict(STANDARD["contact"]["ground_friction"])

# ── Exercise Phases Standard (number of phases per exercise) ──────
# Legacy keys (``back_squat`` for ``squat``) are kept for existing callers.
_LEGACY_EXERCISES = ("squat", "deadlift", "bench_press", "snatch", "clean_and_jerk")
EXERCISE_PHASE_COUNTS: dict[str, int] = {
    STANDARD["exercises"][ex].get("legacy_key", ex): STANDARD["exercises"][ex][
        "phase_count"
    ]
    for ex in _LEGACY_EXERCISES
}

GRAVITY: tuple[float, float, float] = (0.0, 0.0, -STANDARD["frame"]["gravity_mps2"])
