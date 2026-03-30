"""Gait (walking) model builder for MuJoCo MJCF.

Builds a bipedal gait analysis model without a barbell. The initial pose
places the model at right heel strike with slight forward lean and
asymmetric hip/knee angles representing early stance phase.

Biomechanical notes:
- Primary movers: hip flexors/extensors, quadriceps, hamstrings,
  gastrocnemius, tibialis anterior
- The model captures sagittal-plane gait kinematics (swing/stance)
- No barbell is attached; this is a bodyweight movement analysis
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.body import BodyModelSpec

logger = logging.getLogger(__name__)

# Initial pose: right heel strike (radians)
_INITIAL_HIP_L_FLEX = math.radians(20)  # trailing leg
_INITIAL_HIP_R_FLEX = math.radians(-10)  # stance leg
_INITIAL_KNEE_L_FLEX = math.radians(-5)
_INITIAL_KNEE_R_FLEX = math.radians(-20)
_INITIAL_ANKLE_L_FLEX = math.radians(10)
_INITIAL_ANKLE_R_FLEX = math.radians(-5)

# Mapping of joint name fragments to their initial ref values.
_GAIT_INITIAL_REFS: dict[str, float] = {
    "hip_l_flex": _INITIAL_HIP_L_FLEX,
    "hip_r_flex": _INITIAL_HIP_R_FLEX,
    "knee_l_flex": _INITIAL_KNEE_L_FLEX,
    "knee_r_flex": _INITIAL_KNEE_R_FLEX,
    "ankle_l_flex": _INITIAL_ANKLE_L_FLEX,
    "ankle_r_flex": _INITIAL_ANKLE_R_FLEX,
}


class GaitModelBuilder(ExerciseModelBuilder):
    """Builds a MJCF model configured for bipedal gait analysis.

    No barbell is used; the model is for walking/running biomechanics.
    """

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the gait model."""
        return "gait"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """No barbell for gait analysis -- intentional no-op."""

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set right heel strike pose with asymmetric limb angles.

        The right leg is in early stance (extended hip, flexed knee)
        while the left leg is in late swing (flexed hip, nearly straight knee).
        """
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            joint_type = joint.get("type", "hinge")
            if joint_type != "hinge":
                continue
            if name in _GAIT_INITIAL_REFS:
                joint.set("ref", str(_GAIT_INITIAL_REFS[name]))
        logger.debug(
            "Setting gait initial pose: hip_l=%.4f, hip_r=%.4f, "
            "knee_l=%.4f, knee_r=%.4f rad",
            _INITIAL_HIP_L_FLEX,
            _INITIAL_HIP_R_FLEX,
            _INITIAL_KNEE_L_FLEX,
            _INITIAL_KNEE_R_FLEX,
        )


def build_gait_model(body_mass: float = 80.0, height: float = 1.75) -> str:
    """Convenience function to build a gait analysis model MJCF XML string.

    Default: 80 kg person, 1.75 m tall, no barbell.
    """
    from mujoco_models.shared.barbell import BarbellSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=0.0),
    )
    return GaitModelBuilder(config).build()
