# SPDX-License-Identifier: MIT
"""Bench press model builder for MuJoCo MJCF.

The lifter lies supine on a bench (head toward -X, chest facing +Z). The
barbell is gripped in both hands directly above the shoulders. The model
starts in the lockout position (arms extended) and the motion descends the bar
to the chest then presses back to lockout.

Biomechanical notes:
- Primary movers: pectoralis major, anterior deltoid, triceps brachii
- The bench constrains pelvis and torso to a supine orientation
- Scapular retraction and arch are simplified (torso stays rigid on bench)
- Grip width affects shoulder abduction angle and pec activation

The bench is a platform that constrains the pelvis to a supine position at
bench height (0.43 m, standard IPF).
Canonical frame (MuJoCo world): X forward, Y left, Z up; gravity = (0, 0, -9.80665).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.body.segment_data import segment_properties
from mujoco_models.shared.contact_masks import EQUIPMENT_MASKS, apply_masks
from mujoco_models.shared.utils.mjcf_helpers import add_weld_constraint

logger = logging.getLogger(__name__)

BENCH_HEIGHT = 0.43  # IPF standard bench height (meters)

# Supine pose.  The lifter lies on the bench with the head toward -X and the
# chest facing +Z: the pelvis is the root of the body, so rotating it by -90
# degrees about Y (quaternion w, x, y, z) lays the whole body down.
_SUPINE_QUAT = (math.cos(math.pi / 4.0), 0.0, -math.sin(math.pi / 4.0), 0.0)
# With the pelvis rotated the same way and the shoulder flexed 90 degrees
# (forward = up for a supine lifter) every hand frame is the world frame
# turned 180 degrees about Y, so the bar sits in it with this orientation.
_HAND_QUAT = (0.0, 0.0, 1.0, 0.0)
_PELVIS_CLEARANCE = 0.001  # keep the pelvis box just off the bench top
_BENCH_CENTER_X = -0.30  # centred under the torso and pelvis
_BENCH_HALF_SIZE = (0.65, 0.12, 0.02)  # length (X), width (Y), thickness (Z)

# Lockout pose: the shoulder is flexed 90 deg so the arms point at the ceiling,
# hands directly above the shoulders (adduction 0) and elbows locked.  The
# shoulder flexion is applied as a keyframe angle, not a ``ref``: ``ref`` only
# relabels the zero angle and never moves a segment.
_INITIAL_SHOULDER_FLEX = math.pi / 2.0
_INITIAL_ELBOW_FLEX = 0.0  # fully extended


class BenchPressModelBuilder(ExerciseModelBuilder):
    """Builds a bench-press MuJoCo MJCF model.

    The pelvis is welded to the bench in a supine orientation at bench height.
    The barbell shaft is welded to both hands, directly above the shoulders.
    """

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the bench press model."""
        return "bench_press"

    @property
    def _pelvis_height(self) -> float:
        """World height of the supine pelvis centre (box resting on the bench)."""
        _m, _length, radius = segment_properties(
            self.body_spec.total_mass, self.body_spec.height, "pelvis"
        )
        return BENCH_HEIGHT + radius + _PELVIS_CLEARANCE

    @property
    def barbell_start_pos(self) -> tuple[float, float, float]:
        """Bar centre above the hands: over the shoulders, arms fully raised."""
        spec = self.body_spec
        shoulder_x = -(spec.shoulder_height - spec.pelvis_height)  # toward the head
        reach = spec.shoulder_height - spec.hand_height
        return (shoulder_x, 0.0, self._pelvis_height + reach)

    def keyframe_angle_offsets(self) -> dict[str, float]:
        """Raise both arms: shoulder flexion of 90 degrees at the keyframe."""
        return {
            "shoulder_l_flex": _INITIAL_SHOULDER_FLEX,
            "shoulder_r_flex": _INITIAL_SHOULDER_FLEX,
        }

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld the barbell to both hands, which are directly above the shoulders."""
        self._attach_barbell_to_hands(
            equality,
            grip_width=self.body_spec.shoulder_half_width,
            hand_quat=_HAND_QUAT,
        )

    def _post_worldbody_hook(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Lay the lifter on the bench, add the bench and weld the pelvis to it."""
        self._lay_supine(worldbody)
        self._add_bench(worldbody, equality)

    def _lay_supine(self, worldbody: ET.Element) -> None:
        """Rotate the pelvis (root body) supine and rest it on the bench top."""
        pelvis = worldbody.find("body[@name='pelvis']")
        if pelvis is None:
            logger.warning("No pelvis body found; bench lifter left upright")
            return
        pelvis.set("pos", f"0 0 {self._pelvis_height:.6f}")
        pelvis.set("quat", " ".join(f"{q:.6f}" for q in _SUPINE_QUAT))

    def _add_bench(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Add a bench body and weld the pelvis to it."""
        bench = ET.SubElement(
            worldbody,
            "body",
            name="bench",
            pos=f"{_BENCH_CENTER_X} 0 {BENCH_HEIGHT - 0.02:.6f}",
        )
        # ⚡ Bolt Optimization: Pass attributes as kwargs to ET.SubElement
        # to avoid Python call frame overhead from multiple .set() calls.
        bench_geom = ET.SubElement(
            bench,
            "geom",
            name="bench_contact",
            type="box",
            size=" ".join(f"{v:g}" for v in _BENCH_HALF_SIZE),
            rgba="0.5 0.35 0.2 1",
            condim="3",
            friction="0.8 0.005 0.0001",
        )
        apply_masks(bench_geom, EQUIPMENT_MASKS)

        add_weld_constraint(
            equality,
            name="pelvis_to_bench",
            body1="pelvis",
            body2="bench",
        )
        logger.debug("Added bench body at height %.3f m", BENCH_HEIGHT)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set the elbow reference: arms fully extended at lockout.

        The shoulder pose itself is a keyframe angle (see
        :meth:`keyframe_angle_offsets`), applied after this hook.

        Ref values are stored in radians to match <compiler angle='radian'>.
        """
        self.set_ref_by_name_map(worldbody, {"elbow": _INITIAL_ELBOW_FLEX})
        logger.debug(
            "Setting bench press initial pose: shoulder_flex=%.4f rad, elbow=%.4f rad",
            _INITIAL_SHOULDER_FLEX,
            _INITIAL_ELBOW_FLEX,
        )


def build_bench_press_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float = 50.0,
) -> str:
    """Convenience function to build a bench press model MJCF XML string."""
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=plate_mass_per_side),
    )
    return BenchPressModelBuilder(config).build()
