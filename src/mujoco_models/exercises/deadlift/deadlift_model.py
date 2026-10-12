# SPDX-License-Identifier: MIT
"""Conventional deadlift model builder for MuJoCo MJCF.

The lifter grips the barbell at approximately shoulder width with the bar
on the floor. The motion lifts from floor to lockout (standing erect with
hips and knees fully extended).

Biomechanical notes:
- Primary movers: erector spinae, gluteus maximus, hamstrings, quadriceps
- The barbell starts on the ground (center of shaft at ~0.225 m height
  for standard 450 mm diameter plates)
- Mixed grip or double-overhand grip -- modelled as rigid hand attachment
- Hip hinge dominant pattern with simultaneous knee extension
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)

The barbell is welded to both hands. The model starts at the parity
standard's ``start`` via-pose (UpstreamDrift#11743): deep hip hinge, flat
feet, palms closed on the bar at plate-radius height over mid-foot.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.exercises.spec_start import plates_per_side_kg

logger = logging.getLogger(__name__)

PLATE_RADIUS = 0.225  # Standard 450mm diameter plate radius


class DeadliftModelBuilder(ExerciseModelBuilder):
    """Builds a conventional deadlift MuJoCo MJCF model.

    The barbell is welded to both hands and starts on the floor.
    """

    seat_feet_at_equilibrium = True
    spec_exercise = "deadlift"

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the deadlift model."""
        return "deadlift"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld the barbell shaft to both hands.

        The weld ``relpose`` is measured at the spec start pose, where the
        palms already sit on the bar at the spec grip width.
        """
        self._attach_barbell_to_hands(equality)


def build_deadlift_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float | None = None,
) -> str:
    """Convenience function to build a deadlift model MJCF XML string.

    ``plate_mass_per_side`` defaults to the standard's deadlift load.
    """
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(
            plates_per_side_kg("deadlift")
            if plate_mass_per_side is None
            else plate_mass_per_side
        ),
    )
    return DeadliftModelBuilder(config).build()
