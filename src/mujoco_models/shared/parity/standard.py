# SPDX-License-Identifier: MIT
"""Cross-repo parity standard — canonical biomechanical parameters.

These values MUST be identical across MuJoCo, Drake, Pinocchio, and
OpenSim repos.  Any deviation breaks cross-repo parity and must be
justified.
"""

from __future__ import annotations

import math


def _rad(deg: float) -> float:
    """Convert degrees to radians."""
    return math.radians(deg)


# ── Anthropometric Standard (Winter 2009) ──────────────────────────
STANDARD_BODY_MASS = 80.0  # kg
STANDARD_HEIGHT = 1.75  # m

SEGMENT_MASS_FRACTIONS: dict[str, float] = {
    "pelvis": 0.142,
    "torso": 0.355,
    "head": 0.081,
    "upper_arm": 0.028,
    "forearm": 0.016,
    "hand": 0.006,
    "thigh": 0.100,
    "shank": 0.047,
    "foot": 0.014,
}

SEGMENT_LENGTH_FRACTIONS: dict[str, float] = {
    "pelvis": 0.100,
    "torso": 0.288,
    "head": 0.130,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.050,
    "thigh": 0.245,
    "shank": 0.246,
    "foot": 0.040,
}

# ── Joint Limit Standard (degrees, converted to radians) ──────────
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "hip_flex": (_rad(-30), _rad(120)),
    "hip_adduct": (_rad(-45), _rad(30)),
    "hip_rotate": (_rad(-45), _rad(45)),
    "knee_flex": (_rad(-150), _rad(0)),
    "ankle_flex": (_rad(-20), _rad(50)),
    "ankle_invert": (_rad(-20), _rad(20)),
    "shoulder_flex": (_rad(-60), _rad(180)),
    "shoulder_adduct": (_rad(-30), _rad(180)),
    "shoulder_rotate": (_rad(-90), _rad(90)),
    "elbow_flex": (_rad(0), _rad(150)),
    "wrist_flex": (_rad(-70), _rad(70)),
    "wrist_deviate": (_rad(-20), _rad(30)),
    "lumbar_flex": (_rad(-30), _rad(45)),
    "lumbar_lateral": (_rad(-30), _rad(30)),
    "lumbar_rotate": (_rad(-30), _rad(30)),
    "neck_flex": (_rad(-30), _rad(30)),
}

# ── Barbell Standard (IWF/IPF) ────────────────────────────────────
MENS_BARBELL: dict[str, float] = {
    "total_length": 2.20,
    "shaft_length": 1.31,
    "shaft_diameter": 0.028,
    "sleeve_diameter": 0.050,
    "bar_mass": 20.0,
}

# ── Contact Standard ──────────────────────────────────────────────
FOOT_CONTACT_DIMS: dict[str, float] = {
    "length": 0.26,
    "width": 0.10,
    "height": 0.02,
}
GROUND_FRICTION: dict[str, float] = {
    "static": 0.8,
    "dynamic": 0.6,
}

# ── Exercise Phases Standard (number of phases per exercise) ──────
EXERCISE_PHASE_COUNTS: dict[str, int] = {
    "back_squat": 5,
    "deadlift": 5,
    "bench_press": 5,
    "snatch": 6,
    "clean_and_jerk": 8,
}

GRAVITY: tuple[float, float, float] = (0.0, 0.0, -9.80665)
