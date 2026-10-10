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

The barbell is welded to both hands. Initial pose has significant hip
and knee flexion to reach the bar on the ground.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import (
    FLOOR_PULL_HIP_FLEX,
    FLOOR_PULL_KNEE_FLEX,
    ExerciseConfig,
    ExerciseModelBuilder,
)

logger = logging.getLogger(__name__)

PLATE_RADIUS = 0.225  # Standard 450mm diameter plate radius

# Re-export shared constants under the local names used by tests.
_INITIAL_HIP_FLEX = FLOOR_PULL_HIP_FLEX
_INITIAL_KNEE_FLEX = FLOOR_PULL_KNEE_FLEX

# Slightly outside shoulder width (~0.22 m from center).
_GRIP_WIDTH = 0.22


class DeadliftModelBuilder(ExerciseModelBuilder):
    """Builds a conventional deadlift MuJoCo MJCF model.

    The barbell is welded to both hands and starts on the floor.
    """

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the deadlift model."""
        return "deadlift"

    @property
    def barbell_start_pos(self) -> tuple[float, float, float]:
        """Bar centre at hand height, raised for the grip's shoulder abduction."""
        rise = self._grip_vertical_rise(_GRIP_WIDTH)
        return (0.0, 0.0, self.body_spec.hand_height + rise)

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld barbell shaft to both hands at shoulder-width grip.

        Grip is slightly outside the knees (~0.22 m from center); the start
        pose abducts the shoulders to actually reach that width (see
        :meth:`keyframe_angle_offsets`), so the weld's grip width matches
        the hands' true kinematic position (MuJoCo_Models#408). The per-side
        ``hand_quat`` keeps the weld satisfied even though the wrist no
        longer counter-rotates the shoulder abduction's hand tilt
        (MuJoCo_Models#440).
        """
        self._attach_barbell_to_hands(
            equality,
            grip_width=self._achieved_grip_width(_GRIP_WIDTH),
            hand_quat=self._grip_hand_quats(_GRIP_WIDTH),
        )

    def keyframe_angle_offsets(self) -> dict[str, float]:
        """Abduct the shoulders so both hands reach the ``_GRIP_WIDTH`` grip."""
        return self._grip_pose_offsets(_GRIP_WIDTH)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set the starting position: deep hip hinge, knees flexed.

        The bar is on the floor at PLATE_RADIUS height, so the body
        must flex at the hips (~80 deg) and knees (~60 deg) to reach.

        Ref values are stored in radians to match <compiler angle='radian'>.
        """
        self.set_ref_by_name_map(
            worldbody,
            {
                "hip_l_flex": _INITIAL_HIP_FLEX,
                "hip_r_flex": _INITIAL_HIP_FLEX,
                "knee": _INITIAL_KNEE_FLEX,
            },
        )
        logger.debug(
            "Setting deadlift initial pose: hip_flex=%.4f rad, knee_flex=%.4f rad",
            _INITIAL_HIP_FLEX,
            _INITIAL_KNEE_FLEX,
        )


def build_deadlift_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float = 80.0,
) -> str:
    """Convenience function to build a deadlift model MJCF XML string."""
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=plate_mass_per_side),
    )
    return DeadliftModelBuilder(config).build()
