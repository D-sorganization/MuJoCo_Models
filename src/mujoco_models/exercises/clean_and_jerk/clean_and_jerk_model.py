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
- Grip width: shoulder width or slightly outside (~0.25-0.30 m from center)
- Front rack requires significant wrist/elbow flexibility
- Jerk phase requires rapid coordination of upper and lower body
- Primary movers: entire posterior chain + quadriceps + deltoids + triceps
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)

The barbell is welded to both hands at clean grip width.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder

logger = logging.getLogger(__name__)

# Deep hip hinge starting position (same as deadlift)
_INITIAL_HIP_FLEX = 1.3963  # ~80 degrees
_INITIAL_KNEE_FLEX = -1.0472  # ~60 degrees


class CleanAndJerkModelBuilder(ExerciseModelBuilder):
    """Builds a clean-and-jerk MuJoCo MJCF model.

    Uses shoulder-width grip. The model supports both the clean
    (floor to shoulders) and jerk (shoulders to overhead) phases.
    """

    def __init__(self, config: ExerciseConfig | None = None) -> None:
        super().__init__(config)

    @property
    def exercise_name(self) -> str:
        return "clean_and_jerk"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld barbell to both hands at clean grip width.

        Clean grip: approximately shoulder width, ~0.25 m from shaft center.
        """
        self._attach_barbell_to_hands(equality)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set starting position: bar on floor, clean grip, hip hinge."""
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            joint_type = joint.get("type", "hinge")
            if joint_type != "hinge":
                continue
            if "hip" in name:
                joint.set("ref", str(math.degrees(_INITIAL_HIP_FLEX)))
            elif "knee" in name:
                joint.set("ref", str(math.degrees(_INITIAL_KNEE_FLEX)))
        logger.debug(
            "Setting clean & jerk initial pose: hip_flex=%.1f°, knee_flex=%.1f°",
            math.degrees(_INITIAL_HIP_FLEX),
            math.degrees(_INITIAL_KNEE_FLEX),
        )


def build_clean_and_jerk_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float = 50.0,
) -> str:
    """Convenience function to build a clean-and-jerk model MJCF XML string.

    Default: 80 kg person, 120 kg total barbell.
    """
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=plate_mass_per_side),
    )
    return CleanAndJerkModelBuilder(config).build()
