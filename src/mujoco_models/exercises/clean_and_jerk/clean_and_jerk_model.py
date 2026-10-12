# SPDX-License-Identifier: MIT
"""Clean and jerk model builder for MuJoCo MJCF.

The clean and jerk is a two-part lift:

**Clean** -- bar from floor to front-rack (shoulders):
1. First pull -- bar leaves the floor (similar to deadlift)
2. Transition / scoop -- knees re-bend
3. Second pull -- explosive triple extension
4. Turnover -- elbows rotate forward, bar lands on front deltoids
5. Front squat catch -- receive bar in front rack at bottom of squat
6. Recovery -- stand from front squat

**Jerk** -- bar from shoulders to overhead:
1. Dip -- slight knee bend to load legs
2. Drive -- explosive leg drive pushes bar upward
3. Split / push / squat jerk -- lifter drops under bar
4. Recovery -- bring feet together, stand with bar overhead

Biomechanical notes:
- Grip width: the standard's clean grip (0.60 m between palm centres)
- Front rack requires significant wrist/elbow flexibility
- Jerk phase requires rapid coordination of upper and lower body
- Primary movers: entire posterior chain + quadriceps + deltoids + triceps
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


class CleanAndJerkModelBuilder(ExerciseModelBuilder):
    """Builds a clean-and-jerk MuJoCo MJCF model.

    Uses shoulder-width grip. The model supports both the clean
    (floor to shoulders) and jerk (shoulders to overhead) phases.
    """

    seat_feet_at_equilibrium = True
    spec_exercise = "clean_and_jerk"

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the clean and jerk model."""
        return "clean_and_jerk"

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


def build_clean_and_jerk_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float | None = None,
) -> str:
    """Convenience function to build a clean-and-jerk model MJCF XML string.

    Default: 80 kg person, 120 kg total barbell.
    """
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(
            plates_per_side_kg("clean_and_jerk")
            if plate_mass_per_side is None
            else plate_mass_per_side
        ),
    )
    return CleanAndJerkModelBuilder(config).build()
