"""Sit-to-stand model builder for MuJoCo MJCF.

Builds a model for sit-to-stand (STS) transfer analysis with a chair
body welded to the ground. The initial pose places the model in a
seated position (~90 deg hip and knee flexion).

Biomechanical notes:
- Primary movers: quadriceps, gluteus maximus, erector spinae
- Movement strategy: torso forward lean for momentum, then extend
- The chair seat is at ~0.45 m (typical chair height)
- No barbell is attached; this is a bodyweight movement analysis
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.body import BodyModelSpec

logger = logging.getLogger(__name__)

# Chair dimensions (metres)
_CHAIR_SEAT_HEIGHT = 0.45
_CHAIR_SEAT_DEPTH = 0.40
_CHAIR_SEAT_WIDTH = 0.45
_CHAIR_BACK_HEIGHT = 0.40

# Initial seated pose (radians): ~90 deg hip and knee flexion
_INITIAL_HIP_FLEX = math.radians(90)
_INITIAL_KNEE_FLEX = math.radians(-90)
_INITIAL_ANKLE_FLEX = math.radians(15)

_STS_INITIAL_REFS: dict[str, float] = {
    "hip_l_flex": _INITIAL_HIP_FLEX,
    "hip_r_flex": _INITIAL_HIP_FLEX,
    "knee_l_flex": _INITIAL_KNEE_FLEX,
    "knee_r_flex": _INITIAL_KNEE_FLEX,
    "ankle_l_flex": _INITIAL_ANKLE_FLEX,
    "ankle_r_flex": _INITIAL_ANKLE_FLEX,
}


class SitToStandModelBuilder(ExerciseModelBuilder):
    """Builds a MJCF model configured for sit-to-stand analysis.

    Adds a chair body welded to ground and sets initial seated pose.
    No barbell is used.
    """

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the sit-to-stand model."""
        return "sit_to_stand"

    @property
    def uses_barbell(self) -> bool:
        """Sit-to-stand is a bodyweight movement -- no barbell is built."""
        return False

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """No barbell for sit-to-stand -- intentional no-op.

        Kept to satisfy the abstract contract; never invoked because
        ``uses_barbell`` is ``False``.
        """

    def _post_worldbody_hook(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Add a chair body welded to the ground plane.

        Decomposed into helpers per the single-responsibility rule:
        one function per geometric piece plus one for the weld.
        """
        chair = self._create_chair_body(worldbody)
        self._add_chair_seat_geom(chair)
        self._add_chair_back_geom(chair)
        self._weld_chair_to_world(equality)
        logger.debug("Added chair at seat height %.3f m", _CHAIR_SEAT_HEIGHT)

    @staticmethod
    def _create_chair_body(worldbody: ET.Element) -> ET.Element:
        """Create the parent ``<body name="chair">`` element."""
        return ET.SubElement(
            worldbody,
            "body",
            name="chair",
            pos=f"0 0 {_CHAIR_SEAT_HEIGHT / 2:.4f}",
        )

    @staticmethod
    def _add_chair_seat_geom(chair: ET.Element) -> None:
        """Add the horizontal seat box geom."""
        ET.SubElement(
            chair,
            "geom",
            name="chair_seat",
            type="box",
            size=(f"{_CHAIR_SEAT_WIDTH / 2:.4f} {_CHAIR_SEAT_DEPTH / 2:.4f} 0.02"),
            pos=f"0 0 {_CHAIR_SEAT_HEIGHT / 2:.4f}",
            rgba="0.6 0.4 0.2 1",
            contype="1",
            conaffinity="1",
        )

    @staticmethod
    def _add_chair_back_geom(chair: ET.Element) -> None:
        """Add the vertical chair-back box geom."""
        ET.SubElement(
            chair,
            "geom",
            name="chair_back",
            type="box",
            size=(f"{_CHAIR_SEAT_WIDTH / 2:.4f} 0.02 {_CHAIR_BACK_HEIGHT / 2:.4f}"),
            pos=(
                f"0 "
                f"{-_CHAIR_SEAT_DEPTH / 2:.4f} "
                f"{_CHAIR_SEAT_HEIGHT / 2 + _CHAIR_BACK_HEIGHT / 2:.4f}"
            ),
            rgba="0.6 0.4 0.2 1",
        )

    @staticmethod
    def _weld_chair_to_world(equality: ET.Element) -> None:
        """Weld the chair body to the world so it is immovable."""
        ET.SubElement(
            equality,
            "weld",
            name="chair_to_world",
            body1="chair",
            relpose="0 0 0 1 0 0 0",
        )

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set seated position: ~90 deg hip and knee flexion.

        The model starts seated on the chair with feet flat on the ground.
        """
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            joint_type = joint.get("type", "hinge")
            if joint_type != "hinge":
                continue
            if name in _STS_INITIAL_REFS:
                joint.set("ref", str(_STS_INITIAL_REFS[name]))
        logger.debug(
            "Setting sit-to-stand initial pose: hip_flex=%.4f, knee_flex=%.4f rad",
            _INITIAL_HIP_FLEX,
            _INITIAL_KNEE_FLEX,
        )


def build_sit_to_stand_model(body_mass: float = 80.0, height: float = 1.75) -> str:
    """Convenience function to build a sit-to-stand model MJCF XML string.

    Default: 80 kg person, 1.75 m tall, standard chair height.
    """
    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
    )
    return SitToStandModelBuilder(config).build()
