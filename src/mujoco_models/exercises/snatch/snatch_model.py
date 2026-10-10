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
- Grip width: ~1.5x shoulder width (approx 0.55-0.65 m from center)
- Primary movers: entire posterior chain, deltoids, trapezius
- Requires extreme shoulder mobility for overhead position
- Bar path is close to the body (S-curve trajectory)
- MuJoCo Z-up convention: gravity = (0, 0, -9.80665)

The barbell is welded to both hands with a wide grip offset.
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

# Deep hip hinge starting position — same as deadlift (shared constants).
_INITIAL_HIP_FLEX = FLOOR_PULL_HIP_FLEX
_INITIAL_KNEE_FLEX = FLOOR_PULL_KNEE_FLEX
# Snatch grip is approximately 0.55-0.60 m from shaft center on each side
# (~1.5x shoulder width).  The shoulder's own range of motion caps how wide
# ``keyframe_angle_offsets`` can actually abduct the arms (the standard's
# shoulder_adduct range stops at -30 deg of abduction), so the pose reaches
# as close to this as the joint limit allows; ``attach_barbell`` welds to
# the width it actually achieves (MuJoCo_Models#408).
_GRIP_WIDTH = 0.60


class SnatchModelBuilder(ExerciseModelBuilder):
    """Builds a snatch MuJoCo MJCF model with wide grip."""

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the snatch model."""
        return "snatch"

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
        """Weld barbell to both hands with wide (snatch) grip.

        The start pose abducts the shoulders toward ``_GRIP_WIDTH`` (see
        :meth:`keyframe_angle_offsets`); the weld uses the width that pose
        actually achieves, which the shoulder's range of motion may clamp
        below ``_GRIP_WIDTH`` (MuJoCo_Models#408).
        """
        self._attach_barbell_to_hands(
            equality, grip_width=self._achieved_grip_width(_GRIP_WIDTH)
        )

    def keyframe_angle_offsets(self) -> dict[str, float]:
        """Abduct the shoulders so both hands reach toward the wide snatch grip."""
        return self._grip_pose_offsets(_GRIP_WIDTH)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set starting position: bar on floor, deep hip hinge.

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
            "Setting snatch initial pose: hip_flex=%.4f rad, knee_flex=%.4f rad",
            _INITIAL_HIP_FLEX,
            _INITIAL_KNEE_FLEX,
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
