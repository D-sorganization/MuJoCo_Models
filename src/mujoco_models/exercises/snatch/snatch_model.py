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
- Grip width: ~1.5x shoulder width (approx 0.55-0.65 m from center)
- Primary movers: entire posterior chain, deltoids, trapezius
- Requires extreme shoulder mobility for overhead position
- Bar path is close to the body (S-curve trajectory)
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)

The barbell is welded to both hands with a wide grip offset.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import (
    FLOOR_PULL_HIP_FLEX,
    FLOOR_PULL_KNEE_FLEX,
    ExerciseConfig,
    ExerciseModelBuilder,
)

logger = logging.getLogger(__name__)

# Deep hip hinge starting position — same as deadlift (shared constants).
_INITIAL_HIP_FLEX = FLOOR_PULL_HIP_FLEX
_INITIAL_KNEE_FLEX = FLOOR_PULL_KNEE_FLEX
_INITIAL_SHOULDER_ADDUCT = math.radians(45)  # ~45° abduction for wide overhead grip


class SnatchModelBuilder(ExerciseModelBuilder):
    """Builds a snatch MuJoCo MJCF model with wide grip."""

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
        """Weld barbell to both hands with wide (snatch) grip.

        Snatch grip is approximately 0.55-0.60 m from shaft center
        on each side (~1.5x shoulder width).
        """
        self._attach_barbell_to_hands(equality, grip_width=0.60)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set starting position: bar on floor, wide grip, deep hip hinge.

        Shoulder abduction opens the arms for the wide snatch grip.

        Ref values are stored in radians to match <compiler angle='radian'>.
        """
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            joint_type = joint.get("type", "hinge")
            if joint_type != "hinge":
                continue
            if name.endswith("_flex") and "hip" in name:
                joint.set("ref", str(_INITIAL_HIP_FLEX))
            elif "shoulder" in name and "adduct" in name:
                joint.set("ref", str(_INITIAL_SHOULDER_ADDUCT))
            elif "knee" in name:
                joint.set("ref", str(_INITIAL_KNEE_FLEX))
        logger.debug(
            "Setting snatch initial pose: hip_flex=%.4f rad, knee_flex=%.4f rad, "
            "shoulder_adduct=%.4f rad",
            _INITIAL_HIP_FLEX,
            _INITIAL_KNEE_FLEX,
            _INITIAL_SHOULDER_ADDUCT,
        )


def build_snatch_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float = 40.0,
) -> str:
    """Convenience function to build a snatch model MJCF XML string.

    Default: 80 kg person, 120 kg total barbell (competitive 96 kg class).
    """
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=plate_mass_per_side),
    )
    return SnatchModelBuilder(config).build()
