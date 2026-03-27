"""Exercise model builders for MuJoCo MJCF.

Provides a registry mapping exercise names to their builder classes
for programmatic discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mujoco_models.exercises.bench_press.bench_press_model import (
    BenchPressModelBuilder,
)
from mujoco_models.exercises.clean_and_jerk.clean_and_jerk_model import (
    CleanAndJerkModelBuilder,
)
from mujoco_models.exercises.deadlift.deadlift_model import DeadliftModelBuilder
from mujoco_models.exercises.gait.gait_model import GaitModelBuilder
from mujoco_models.exercises.sit_to_stand.sit_to_stand_model import (
    SitToStandModelBuilder,
)
from mujoco_models.exercises.snatch.snatch_model import SnatchModelBuilder
from mujoco_models.exercises.squat.squat_model import SquatModelBuilder

if TYPE_CHECKING:
    from mujoco_models.exercises.base import ExerciseModelBuilder

EXERCISE_REGISTRY: dict[str, type[ExerciseModelBuilder]] = {
    "squat": SquatModelBuilder,
    "back_squat": SquatModelBuilder,
    "bench_press": BenchPressModelBuilder,
    "deadlift": DeadliftModelBuilder,
    "snatch": SnatchModelBuilder,
    "clean_and_jerk": CleanAndJerkModelBuilder,
    "gait": GaitModelBuilder,
    "sit_to_stand": SitToStandModelBuilder,
}

__all__ = [
    "EXERCISE_REGISTRY",
    "BenchPressModelBuilder",
    "CleanAndJerkModelBuilder",
    "DeadliftModelBuilder",
    "GaitModelBuilder",
    "SitToStandModelBuilder",
    "SnatchModelBuilder",
    "SquatModelBuilder",
]
