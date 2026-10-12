# SPDX-License-Identifier: MIT
"""Snatch model builder for MuJoCo MJCF.

The snatch is a single continuous motion that lifts the barbell from the
floor to overhead in one movement. The lifter uses a wide (snatch) grip,
pulls the bar explosively, then drops under it into an overhead squat.

Phases:
1. First pull -- bar leaves the floor (deadlift-like, wide grip)
2. Transition / scoop -- knees re-bend, torso becomes more vertical
3. Second pull -- explosive triple extension (ankle, knee, hip)
4. Turnover -- lifter pulls under the bar, rotating arms overhead
5. Catch -- overhead squat position (deep squat, arms locked overhead)
6. Recovery -- stand up from overhead squat to full extension

Biomechanical notes:
- Grip width: the standard's snatch grip (0.90 m between palm centres)
- Primary movers: entire posterior chain, deltoids, trapezius
- Requires extreme shoulder mobility for overhead position
- Bar path is close to the body (S-curve trajectory)
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)

The barbell is welded to both hands at the standard's ``start`` via-pose
(UpstreamDrift#11743).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.exercises.spec_start import plates_per_side_kg

logger = logging.getLogger(__name__)


class SnatchModelBuilder(ExerciseModelBuilder):
    """Builds a snatch MuJoCo MJCF model with wide grip."""

    seat_feet_at_equilibrium = True
    spec_exercise = "snatch"

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the snatch model."""
        return "snatch"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld the barbell shaft to both hands at the spec grip.

        The weld ``relpose`` is measured at the spec start pose, where the
        palms already sit on the bar at the spec grip width.
        """
        self._attach_barbell_to_hands(equality)


def build_snatch_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float | None = None,
) -> str:
    """Convenience function to build a snatch model MJCF XML string.

    Default: 80 kg person, 120 kg total barbell (competitive 96 kg class).
    """
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(
            plates_per_side_kg("snatch")
            if plate_mass_per_side is None
            else plate_mass_per_side
        ),
    )
    return SnatchModelBuilder(config).build()
