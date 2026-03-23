"""Bench press model builder for MuJoCo MJCF.

The lifter lies supine on a bench. The barbell is gripped in the hands
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

from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.utils.mjcf_helpers import add_weld_constraint

BENCH_HEIGHT = 0.43  # IPF standard bench height (meters)


class BenchPressModelBuilder(ExerciseModelBuilder):
    """Builds a bench-press MuJoCo MJCF model.

    The pelvis is welded to ground in a supine orientation at bench height.
    The barbell shaft is welded to the left hand at grip width.
    """

    def __init__(self, config: ExerciseConfig | None = None) -> None:
        super().__init__(config)

    @property
    def exercise_name(self) -> str:
        return "bench_press"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Weld barbell to left hand at grip width.

        The grip is approximately shoulder-width (~0.40 m from center
        on each side for a standard grip).
        """
        add_weld_constraint(
            equality,
            name="barbell_to_left_hand",
            body1="hand_l",
            body2="barbell_shaft",
        )

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set supine lockout position.

        The ground_pelvis freejoint places the body supine on the bench
        with arms extended overhead (lockout).
        """


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
