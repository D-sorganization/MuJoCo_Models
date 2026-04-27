"""Winter (2009) anthropometric segment data.

Segment mass fractions, length fractions, and radius fractions of total
body height for a simplified musculoskeletal model.

Extracted from body_model.py to keep modules under the 300-line guideline.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import math

from mujoco_models.shared.contracts.preconditions import require_positive

# Winter (2009) segment mass fractions and length fractions of total height.
SEGMENT_TABLE: dict[str, dict[str, float]] = {
    "pelvis": {"mass_frac": 0.142, "length_frac": 0.100, "radius_frac": 0.085},
    "torso": {"mass_frac": 0.355, "length_frac": 0.288, "radius_frac": 0.080},
    "head": {"mass_frac": 0.081, "length_frac": 0.130, "radius_frac": 0.060},
    "upper_arm": {"mass_frac": 0.028, "length_frac": 0.186, "radius_frac": 0.023},
    "forearm": {"mass_frac": 0.016, "length_frac": 0.146, "radius_frac": 0.018},
    "hand": {"mass_frac": 0.006, "length_frac": 0.050, "radius_frac": 0.020},
    "thigh": {"mass_frac": 0.100, "length_frac": 0.245, "radius_frac": 0.037},
    "shank": {"mass_frac": 0.047, "length_frac": 0.246, "radius_frac": 0.025},
    "foot": {"mass_frac": 0.014, "length_frac": 0.040, "radius_frac": 0.025},
}

# Unique (non-bilateral) segment names for mass fraction summation.
# Bilateral segments (upper_arm through foot) account for both sides.
_CENTRAL_SEGMENTS = {"pelvis", "torso", "head"}
_BILATERAL_SEGMENTS = {
    "upper_arm",
    "forearm",
    "hand",
    "thigh",
    "shank",
    "foot",
}


def total_mass_fraction() -> float:
    """Return the sum of all segment mass fractions (should be ~1.0).

    Central segments count once; bilateral segments count twice.
    """
    total = 0.0
    for name, props in SEGMENT_TABLE.items():
        multiplier = 2.0 if name in _BILATERAL_SEGMENTS else 1.0
        total += props["mass_frac"] * multiplier
    return total


# ---------------------------------------------------------------------------
# Multi-DOF joint limits (radians)
# ---------------------------------------------------------------------------
# Hip (3-DOF): flexion/extension, abduction/adduction, internal/external rotation
HIP_FLEX_MIN = math.radians(-30)
HIP_FLEX_MAX = math.radians(120)
HIP_ADDUCT_MIN = math.radians(-45)
HIP_ADDUCT_MAX = math.radians(30)
HIP_ROTATE_MIN = math.radians(-45)
HIP_ROTATE_MAX = math.radians(45)

# Shoulder (3-DOF): flexion/extension, abduction/adduction, internal/external rotation
SHOULDER_FLEX_MIN = math.radians(-60)
SHOULDER_FLEX_MAX = math.radians(180)
SHOULDER_ADDUCT_MIN = math.radians(-30)
SHOULDER_ADDUCT_MAX = math.radians(180)
SHOULDER_ROTATE_MIN = math.radians(-90)
SHOULDER_ROTATE_MAX = math.radians(90)

# Lumbar (3-DOF): flexion/extension, lateral flexion, axial rotation
LUMBAR_FLEX_MIN = math.radians(-30)
LUMBAR_FLEX_MAX = math.radians(45)
LUMBAR_LATERAL_MIN = math.radians(-30)
LUMBAR_LATERAL_MAX = math.radians(30)
LUMBAR_ROTATE_MIN = math.radians(-30)
LUMBAR_ROTATE_MAX = math.radians(30)

# Ankle (2-DOF): dorsiflexion/plantarflexion, inversion/eversion
ANKLE_FLEX_MIN = math.radians(-20)
ANKLE_FLEX_MAX = math.radians(50)
ANKLE_INVERT_MIN = math.radians(-20)
ANKLE_INVERT_MAX = math.radians(20)

# Wrist (2-DOF): flexion/extension, radial/ulnar deviation
WRIST_FLEX_MIN = math.radians(-70)
WRIST_FLEX_MAX = math.radians(70)
WRIST_DEVIATE_MIN = math.radians(-20)
WRIST_DEVIATE_MAX = math.radians(30)


def segment_properties(
    total_mass: float, height: float, name: str
) -> tuple[float, float, float]:
    """Return (mass, length, radius) for a named segment.

    Parameters
    ----------
    total_mass : float
        Total body mass in kg.
    height : float
        Total body height in meters.
    name : str
        Segment name (must be a key in SEGMENT_TABLE).
    """
    require_positive(total_mass, "total_mass")
    require_positive(height, "height")
    props = SEGMENT_TABLE[name]
    mass = total_mass * props["mass_frac"]
    length = height * props["length_frac"]
    radius = height * props["radius_frac"]
    return mass, length, radius
