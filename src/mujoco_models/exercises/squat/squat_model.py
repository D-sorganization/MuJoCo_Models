# SPDX-License-Identifier: MIT
"""Back squat model builder for MuJoCo MJCF.

The barbell rests across the upper trapezius / rear deltoids (high-bar
position) and is rigidly welded to the torso. The model starts at the
parity standard's ``start`` via-pose (the "unrack" standing position,
UpstreamDrift#11743), with the palms closed on the bar.

Biomechanical notes:
- Primary movers: quadriceps, gluteus maximus, hamstrings, erector spinae
- The model captures sagittal-plane kinematics (flexion/extension)
- Barbell path should remain roughly over mid-foot
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.exercises.spec_start import plates_per_side_kg
from mujoco_models.shared.utils.mjcf_helpers import add_weld_constraint

logger = logging.getLogger(__name__)


class SquatModelBuilder(ExerciseModelBuilder):
    """Builds a back-squat MuJoCo MJCF model.

    The barbell is welded to the torso at the approximate position of the
    upper trapezius (high-bar squat). For a low-bar variant, the attachment
    point would be shifted ~5 cm inferior.
    """

    seat_feet_at_equilibrium = True
    spec_exercise = "squat"

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the back squat model."""
        return "back_squat"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld the barbell shaft to the torso; the ``relpose`` is measured at
        the spec start pose, with the bar on the upper back."""
        add_weld_constraint(
            equality,
            name="barbell_to_torso",
            body1="torso",
            body2="barbell_shaft",
        )

    def _post_worldbody_hook(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Verify foot contact geometry is present for ground contact."""
        contact_geoms = [
            g
            for g in worldbody.iter("geom")
            if g.get("name", "").startswith("foot_")
            and g.get("name", "").endswith("_contact")
        ]
        if len(contact_geoms) < 2:
            logger.warning(
                "Expected 2 foot contact geoms for squat, found %d",
                len(contact_geoms),
            )
        else:
            logger.debug(
                "Squat contact setup verified: %d foot contact geoms",
                len(contact_geoms),
            )


def build_squat_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float | None = None,
) -> str:
    """Convenience function to build a squat model MJCF XML string.

    Default: 80 kg person, 1.75 m tall, the standard's squat load
    (20 kg bar + 50 kg per side).
    """
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(
            plates_per_side_kg("squat")
            if plate_mass_per_side is None
            else plate_mass_per_side
        ),
    )
    return SquatModelBuilder(config).build()
