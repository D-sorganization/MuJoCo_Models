# SPDX-License-Identifier: MIT
"""Exercise-specific optimization objectives for barbell movements.

Defines target poses, movement phases, and constraints for each of
the five classical barbell exercises. These objectives drive trajectory
optimization by specifying kinematic targets and movement structure.

Design by Contract:
    - All joint angles are in radians.
    - Phase fractions are monotonically increasing from 0.0 to 1.0.
    - Each exercise has a valid bar path constraint and balance mode.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from mujoco_models.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Valid bar path constraint types.
VALID_BAR_PATH_CONSTRAINTS: frozenset[str] = frozenset(
    {"vertical", "j_curve", "s_curve", "none"}
)

# Valid balance modes for stance classification.
VALID_BALANCE_MODES: frozenset[str] = frozenset(
    {"bilateral_stance", "split_stance", "supine", "alternating_stance"}
)


@dataclass(frozen=True)
class Phase:
    """A single phase within an exercise movement.

    Attributes:
        name: Descriptive name for this movement phase.
        fraction: Normalized position within the total movement
            (0.0 = start, 1.0 = finish).
        target_joints: Joint name to target angle (radians)
            mapping for this phase.
    """

    name: str
    fraction: float
    target_joints: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate phase fraction is in [0, 1]."""
        if not 0.0 <= self.fraction <= 1.0:
            msg = (
                f"Phase fraction must be in [0, 1], "
                f"got {self.fraction} for phase '{self.name}'"
            )

            raise ValidationError(msg)


@dataclass(frozen=True)
class ExerciseObjective:
    """Complete optimization specification for a barbell exercise.

    Attributes:
        name: Exercise identifier (e.g. 'back_squat').
        start_pose: Joint name to angle (radians) at the start.
        end_pose: Joint name to angle (radians) at the finish.
        phases: Ordered movement phases with timing and targets.
        bar_path_constraint: Type of bar path ('vertical',
            'j_curve', 's_curve').
        balance_mode: Stance type ('bilateral_stance',
            'split_stance', 'supine').
    """

    name: str
    start_pose: dict[str, float] = field(default_factory=dict)
    end_pose: dict[str, float] = field(default_factory=dict)
    phases: list[Phase] = field(default_factory=list)
    bar_path_constraint: str = "vertical"
    balance_mode: str = "bilateral_stance"

    def __post_init__(self) -> None:
        """Validate objective configuration."""
        if not self.name:
            msg = "Exercise name must not be empty"
            raise ValidationError(msg)
        if self.bar_path_constraint not in VALID_BAR_PATH_CONSTRAINTS:
            msg = (
                f"Invalid bar_path_constraint "
                f"'{self.bar_path_constraint}', "
                f"must be one of {sorted(VALID_BAR_PATH_CONSTRAINTS)}"
            )
            raise ValidationError(msg)
        if self.balance_mode not in VALID_BALANCE_MODES:
            msg = (
                f"Invalid balance_mode '{self.balance_mode}', "
                f"must be one of {sorted(VALID_BALANCE_MODES)}"
            )
            raise ValidationError(msg)
        _validate_phase_ordering(self.phases)

    def get_phase(self, name: str) -> Phase:
        """Look up a phase by name.

        Args:
            name: Phase name to find.

        Returns:
            The matching Phase.

        Raises:
            KeyError: If no phase with that name exists.
        """
        for phase in self.phases:
            if phase.name == name:
                return phase
        msg = f"No phase named '{name}' in exercise '{self.name}'"

        raise ValidationError(msg)

    @property
    def n_phases(self) -> int:
        """Number of movement phases."""
        return len(self.phases)

    @property
    def joint_names(self) -> list[str]:
        """All joint names referenced across all phases."""
        names: set[str] = set()
        names.update(self.start_pose.keys())
        names.update(self.end_pose.keys())
        for phase in self.phases:
            names.update(phase.target_joints.keys())
        return sorted(names)


def _validate_phase_ordering(phases: list[Phase]) -> None:
    """Ensure phase fractions are monotonically non-decreasing."""
    for i in range(1, len(phases)):
        if phases[i].fraction < phases[i - 1].fraction:
            msg = (
                f"Phase fractions must be non-decreasing: "
                f"phase '{phases[i - 1].name}' "
                f"({phases[i - 1].fraction}) > "
                f"phase '{phases[i].name}' ({phases[i].fraction})"
            )

            raise ValidationError(msg)


@lru_cache(maxsize=1)
def _build_registry() -> dict[str, ExerciseObjective]:
    """Build the objective registry, importing data lazily to avoid cycles.

    Uses ``@lru_cache`` so the registry is built once and reused without
    requiring a module-level global with ``global`` statement.
    """
    from mujoco_models.optimization.objective_data import (
        BENCH_PRESS_OBJECTIVE,
        CLEAN_AND_JERK_OBJECTIVE,
        DEADLIFT_OBJECTIVE,
        GAIT_OBJECTIVE,
        SIT_TO_STAND_OBJECTIVE,
        SNATCH_OBJECTIVE,
        SQUAT_OBJECTIVE,
    )

    return {
        "back_squat": SQUAT_OBJECTIVE,
        "squat": SQUAT_OBJECTIVE,
        "deadlift": DEADLIFT_OBJECTIVE,
        "bench_press": BENCH_PRESS_OBJECTIVE,
        "snatch": SNATCH_OBJECTIVE,
        "clean_and_jerk": CLEAN_AND_JERK_OBJECTIVE,
        "gait": GAIT_OBJECTIVE,
        "sit_to_stand": SIT_TO_STAND_OBJECTIVE,
    }


def get_exercise_objective(name: str) -> ExerciseObjective:
    """Return an exercise objective by name.

    Args:
        name: Exercise name (e.g. 'squat', 'deadlift').

    Returns:
        The matching ExerciseObjective.

    Raises:
        ValidationError: If *name* is not a recognised exercise.
    """
    registry = _build_registry()
    try:
        return registry[name]
    except KeyError:
        available = ", ".join(sorted(registry))

        raise ValidationError(
            f"Unknown exercise '{name}'. Available: {available}"
        ) from None


if TYPE_CHECKING:
    # Static type declarations for names resolved at runtime via __getattr__.
    SQUAT_OBJECTIVE: ExerciseObjective
    DEADLIFT_OBJECTIVE: ExerciseObjective
    BENCH_PRESS_OBJECTIVE: ExerciseObjective
    SNATCH_OBJECTIVE: ExerciseObjective
    CLEAN_AND_JERK_OBJECTIVE: ExerciseObjective
    GAIT_OBJECTIVE: ExerciseObjective
    SIT_TO_STAND_OBJECTIVE: ExerciseObjective
    OBJECTIVE_REGISTRY: dict[str, ExerciseObjective]
    EXERCISE_OBJECTIVES: dict[str, ExerciseObjective]


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy re-exports of objective constants."""
    _OBJECTIVE_NAMES = {
        "SQUAT_OBJECTIVE",
        "DEADLIFT_OBJECTIVE",
        "BENCH_PRESS_OBJECTIVE",
        "SNATCH_OBJECTIVE",
        "CLEAN_AND_JERK_OBJECTIVE",
        "GAIT_OBJECTIVE",
        "SIT_TO_STAND_OBJECTIVE",
        "OBJECTIVE_REGISTRY",
        "EXERCISE_OBJECTIVES",
    }
    if name in {"OBJECTIVE_REGISTRY", "EXERCISE_OBJECTIVES"}:
        return _build_registry()
    if name in _OBJECTIVE_NAMES:
        return _build_registry()[name.removesuffix("_OBJECTIVE").lower()]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
