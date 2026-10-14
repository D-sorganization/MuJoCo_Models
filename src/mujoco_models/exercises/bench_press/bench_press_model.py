# SPDX-License-Identifier: MIT
"""Bench press model builder for MuJoCo MJCF.

The lifter lies supine on a bench (head toward -X, chest facing +Z). The
barbell is gripped in both hands above the shoulders. The model starts at the
parity standard's ``start`` (unrack) via-pose (UpstreamDrift#11743) and the
motion descends the bar to the chest then presses back to lockout.

Biomechanical notes:
- Primary movers: pectoralis major, anterior deltoid, triceps brachii
- The bench constrains pelvis and torso to a supine orientation
- Scapular retraction and arch are simplified (torso stays rigid on bench)
- Grip width affects shoulder abduction angle and pec activation

The bench is a platform (top at about 0.43 m, standard IPF) just under the
supine pelvis, which is placed at the via-pose root and welded to the world/bench
per parity standard v2.0 (root_joint weld).
Canonical frame (MuJoCo world): X forward, Y left, Z up; gravity = (0, 0, -9.80665).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.exercises.spec_start import plates_per_side_kg
from mujoco_models.shared.body.segment_data import segment_properties
from mujoco_models.shared.contact_masks import EQUIPMENT_MASKS, apply_masks
from mujoco_models.shared.utils.mjcf_helpers import add_weld_constraint

logger = logging.getLogger(__name__)

BENCH_HEIGHT = 0.43  # IPF standard bench height (meters)

# Bench centre relative to the pelvis root along X: under the torso and pelvis.
_BENCH_OFFSET_X = -0.30
_PELVIS_CLEARANCE = 0.001  # bench top sits just below the supine pelvis box
_BENCH_HALF_SIZE = (0.65, 0.12, 0.02)  # length (X), width (Y), thickness (Z)
BENCH_MASS = (
    1e-4  # near-zero mass (kg) — bench mass is borne by ground (standard parity)
)


class BenchPressModelBuilder(ExerciseModelBuilder):
    """Builds a bench-press MuJoCo MJCF model.

    The pelvis is welded to the bench in the spec supine root pose.
    The barbell shaft is welded to both hands at the spec unrack position.
    """

    spec_exercise = "bench_press"

    @property
    def exercise_name(self) -> str:
        """Return the canonical exercise name for the bench press model."""
        return "bench_press"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld the barbell to both hands; the ``relpose`` is measured at the
        spec start pose, where the palms sit on the bar."""
        self._attach_barbell_to_hands(equality)

    def _post_worldbody_hook(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Add the bench under the supine lifter and weld the pelvis to it."""
        self._add_bench(worldbody, equality)

    def _add_contact_section(self, root: ET.Element) -> ET.Element:
        """Add the shared exclusions plus the upper-arm versus bench exclusions.

        Issue #496: the body model centres each limb capsule on its proximal
        joint, so the lower half of the upper-arm capsule sits behind the
        shoulder.  Supine, that half reaches below the bench top and
        penetrated it by about 4.4 cm (about 1.2 kN per arm at the start
        keyframe).  The real arm is outside the 0.24 m wide bench, so the
        pair never carries support; torso and pelvis keep the bench contact.
        """
        contact = super()._add_contact_section(root)
        for side in ("l", "r"):
            ET.SubElement(
                contact,
                "exclude",
                name=f"exclude_upper_arm_{side}_bench",
                body1=f"upper_arm_{side}",
                body2="bench",
            )
        return contact

    def _bench_top(self, root_z: float) -> float:
        """Bench top height: just under the supine pelvis box at ``root_z``.

        The spec root height is the IPF bench (``BENCH_HEIGHT``) plus the
        pelvis half-depth, rounded to 0.1 mm; the bench follows the root so
        the pelvis never starts inside it.
        """
        _m, _length, radius = segment_properties(
            self.body_spec.total_mass, self.body_spec.height, "pelvis"
        )
        return root_z - radius - _PELVIS_CLEARANCE

    def _add_bench(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Add a bench body under the pelvis root and weld the pelvis to it."""
        root_x, _root_y, root_z = self.start_pose.root_position
        bench_x = root_x + _BENCH_OFFSET_X
        bench_z = self._bench_top(root_z) - _BENCH_HALF_SIZE[2]
        bench = ET.SubElement(
            worldbody,
            "body",
            name="bench",
            pos=f"{bench_x:.6f} 0 {bench_z:.6f}",
        )
        # ⚡ Bolt Optimization: Pass attributes as kwargs to ET.SubElement
        # to avoid Python call frame overhead from multiple .set() calls.
        bench_geom = ET.SubElement(
            bench,
            "geom",
            name="bench_contact",
            type="box",
            size=" ".join(f"{v:g}" for v in _BENCH_HALF_SIZE),
            mass=f"{BENCH_MASS:g}",
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
        logger.debug("Added bench body, top at %.4f m", bench_z + _BENCH_HALF_SIZE[2])


def build_bench_press_model(
    body_mass: float = 80.0,
    height: float = 1.75,
    plate_mass_per_side: float | None = None,
) -> str:
    """Convenience function to build a bench press model MJCF XML string."""
    from mujoco_models.shared.barbell import BarbellSpec
    from mujoco_models.shared.body import BodyModelSpec

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(
            plates_per_side_kg("bench_press")
            if plate_mass_per_side is None
            else plate_mass_per_side
        ),
    )
    return BenchPressModelBuilder(config).build()
