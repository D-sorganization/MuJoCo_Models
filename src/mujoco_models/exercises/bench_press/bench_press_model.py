"""Bench press model builder for MuJoCo MJCF.

The lifter lies supine on a bench. The barbell is gripped in both hands
at approximately shoulder width. The model starts in the lockout position
(arms extended) and the motion descends the bar to the chest then presses
back to lockout.

Biomechanical notes:
- Primary movers: pectoralis major, anterior deltoid, triceps brachii
- The bench constrains pelvis and torso to a supine orientation
- Scapular retraction and arch are simplified (torso stays rigid on bench)
- Grip width affects shoulder abduction angle and pec activation

The bench is modelled as a ground-welded platform that constrains the
pelvis to a supine position at bench height (0.43 m, standard IPF).
MuJoCo Z-up convention: gravity = (0, 0, -9.80665).
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.utils.mjcf_helpers import add_weld_constraint

logger = logging.getLogger(__name__)

BENCH_HEIGHT = 0.43  # IPF standard bench height (meters)

# Supine lockout position hints (radians)
_INITIAL_SHOULDER_FLEX = -1.5708  # arms overhead (~90 deg from standing)
_INITIAL_SHOULDER_ADDUCT = math.radians(75)  # ~75° adduction for horizontal press plane
_INITIAL_ELBOW_FLEX = 0.0  # fully extended


class BenchPressModelBuilder(ExerciseModelBuilder):
    """Builds a bench-press MuJoCo MJCF model.

    The pelvis is welded to ground in a supine orientation at bench height.
    The barbell shaft is welded to both hands at grip width.
    """

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
        """Weld barbell to both hands at grip width.

        The grip is approximately shoulder-width (~0.40 m from center
        on each side for a standard grip).
        """
        self._attach_barbell_to_hands(equality, grip_width=0.40)

    def _post_worldbody_hook(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Add bench body and weld pelvis to it."""
        self._add_bench(worldbody, equality)

    def _add_bench(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """Add a bench body and weld the pelvis to it."""
        bench = ET.SubElement(worldbody, "body")
        bench.set("name", "bench")
        bench.set("pos", f"0 0 {BENCH_HEIGHT - 0.02:.6f}")
        bench_geom = ET.SubElement(bench, "geom")
        bench_geom.set("name", "bench_contact")
        bench_geom.set("type", "box")
        bench_geom.set("size", "0.30 0.65 0.02")
        bench_geom.set("rgba", "0.5 0.35 0.2 1")
        bench_geom.set("contype", "1")
        bench_geom.set("conaffinity", "1")
        bench_geom.set("condim", "3")
        bench_geom.set("friction", "0.8 0.005 0.0001")

        add_weld_constraint(
            equality,
            name="pelvis_to_bench",
            body1="pelvis",
            body2="bench",
        )
        logger.debug("Added bench body at height %.3f m", BENCH_HEIGHT)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set supine lockout position: shoulder and elbow joint refs.

        Shoulder adduction positions the arms in the horizontal press plane.

        Ref values are stored in radians to match <compiler angle='radian'>.
        """
        self.set_ref_by_name_map(
            worldbody,
            {
                "shoulder_l_flex": _INITIAL_SHOULDER_FLEX,
                "shoulder_r_flex": _INITIAL_SHOULDER_FLEX,
                "shoulder_l_adduct": _INITIAL_SHOULDER_ADDUCT,
                "shoulder_r_adduct": _INITIAL_SHOULDER_ADDUCT,
                "elbow": _INITIAL_ELBOW_FLEX,
            },
        )
        logger.debug(
            "Setting bench press initial pose: shoulder_flex=%.4f rad, "
            "shoulder_adduct=%.4f rad, elbow=%.4f rad",
            _INITIAL_SHOULDER_FLEX,
            _INITIAL_SHOULDER_ADDUCT,
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
