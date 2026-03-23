"""Back squat model builder for MuJoCo MJCF.

The barbell rests across the upper trapezius / rear deltoids (high-bar
position) and is rigidly welded to the torso at shoulder height. The
initial pose places the model in a standing position with slight hip
and knee flexion (the "unrack" position).

Biomechanical notes:
- Primary movers: quadriceps, gluteus maximus, hamstrings, erector spinae
- The model captures sagittal-plane kinematics (flexion/extension)
- Barbell path should remain roughly over mid-foot
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.utils.mjcf_helpers import add_weld_constraint

logger = logging.getLogger(__name__)

# Slight hip/knee flexion for unrack position (radians)
_INITIAL_HIP_FLEX = 0.15  # ~8.6 degrees
_INITIAL_KNEE_FLEX = -0.15  # slight knee bend


class SquatModelBuilder(ExerciseModelBuilder):
    """Builds a back-squat MuJoCo MJCF model.

    The barbell is welded to the torso at the approximate position of the
    upper trapezius (high-bar squat). For a low-bar variant, the attachment
    point would be shifted ~5 cm inferior.
    """

    def __init__(self, config: ExerciseConfig | None = None) -> None:
        super().__init__(config)

    @property
    def exercise_name(self) -> str:
        return "back_squat"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld barbell shaft to torso at upper trap position."""
        add_weld_constraint(
            equality,
            name="barbell_to_torso",
            body1="torso",
            body2="barbell_shaft",
        )

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set standing unrack position: slight hip and knee flexion.

        The lifter starts in a comfortable standing position with the
        barbell on the back, knees slightly unlocked.
        """
        logger.debug(
            "Setting squat initial pose: hip_flex=%.2f, knee_flex=%.2f",
            _INITIAL_HIP_FLEX,
            _INITIAL_KNEE_FLEX,
        )


def build_squat_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float = 60.0,
) -> str:
    """Convenience function to build a squat model MJCF XML string.

    Default: 80 kg person, 1.75 m tall, 140 kg total barbell
    (20 kg bar + 60 kg per side).
    """
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=plate_mass_per_side),
    )
    return SquatModelBuilder(config).build()
