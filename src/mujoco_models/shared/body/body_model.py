# SPDX-License-Identifier: MIT
"""MJCF body assembly facade for the full-body musculoskeletal model.

This module is a thin orchestration layer that delegates to focused
sub-modules for each anatomical region:

- ``axial_skeleton`` -- pelvis, torso, head
- ``upper_limbs``    -- shoulders, elbows, wrists
- ``lower_limbs``    -- hips, knees, ankles, foot contact

Public API (re-exported for backward compatibility):
    BodyModelSpec, _LEG_HEIGHT_FRACTION, create_full_body
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.shared.body.axial_skeleton import build_axial_skeleton
from mujoco_models.shared.body.body_anthropometrics import (
    _LEG_HEIGHT_FRACTION,
    BodyModelSpec,
)
from mujoco_models.shared.body.lower_limbs import build_lower_limbs
from mujoco_models.shared.body.upper_limbs import build_upper_limbs

logger = logging.getLogger(__name__)

# Re-exported so existing ``from mujoco_models.shared.body.body_model import
# _LEG_HEIGHT_FRACTION`` imports (including tests) continue to work.
__all__ = [
    "BodyModelSpec",
    "_LEG_HEIGHT_FRACTION",
    "create_full_body",
]


def create_full_body(
    worldbody: ET.Element,
    spec: BodyModelSpec | None = None,
) -> dict[str, ET.Element]:
    """Build the full-body model and append bodies to worldbody.

    Returns dict of body name -> ET.Element for all created bodies.

    The pelvis height is derived from Winter (2009) anthropometric
    proportions: ``height * 0.530 + pelvis_length / 2``.  For a
    1.75 m person this evaluates to approximately 1.015 m.
    """
    if spec is None:
        spec = BodyModelSpec()

    logger.info(
        "Building full-body model: mass=%.1f kg, height=%.2f m",
        spec.total_mass,
        spec.height,
    )

    # Stage 1: axial skeleton (pelvis, torso, head)
    bodies = build_axial_skeleton(worldbody, spec)

    # Stage 2: upper limbs (shoulders -> hands)
    build_upper_limbs(bodies, spec)

    # Stage 3: lower limbs (hips -> feet) + foot contact geometry
    build_lower_limbs(bodies, spec)

    logger.debug("Created %d body segments", len(bodies))
    return bodies
