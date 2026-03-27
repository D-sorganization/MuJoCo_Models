"""Exercise-specific optimization objectives for barbell movements.

Defines target poses, movement phases, and constraints for each of
the five classical barbell exercises. These objectives drive trajectory
optimization by specifying kinematic targets and movement structure.

Design by Contract:
    - All joint angles are in radians.
    - Phase fractions are monotonically increasing from 0.0 to 1.0.
    - Each exercise has a valid bar path constraint and balance mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Valid bar path constraint types.
VALID_BAR_PATH_CONSTRAINTS = frozenset({"vertical", "j_curve", "s_curve"})

# Valid balance modes for stance classification.
VALID_BALANCE_MODES = frozenset({"bilateral_stance", "split_stance", "supine"})


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
            raise ValueError(msg)


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
            raise ValueError(msg)
        if self.bar_path_constraint not in VALID_BAR_PATH_CONSTRAINTS:
            msg = (
                f"Invalid bar_path_constraint "
                f"'{self.bar_path_constraint}', "
                f"must be one of {sorted(VALID_BAR_PATH_CONSTRAINTS)}"
            )
            raise ValueError(msg)
        if self.balance_mode not in VALID_BALANCE_MODES:
            msg = (
                f"Invalid balance_mode '{self.balance_mode}', "
                f"must be one of {sorted(VALID_BALANCE_MODES)}"
            )
            raise ValueError(msg)
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
        raise KeyError(msg)

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
            raise ValueError(msg)


# ---------- Standing pose (shared by squat, deadlift) ----------

_STANDING_POSE: dict[str, float] = {
    "hip_l_flex": 0.0,
    "hip_r_flex": 0.0,
    "knee_l_flex": 0.0,
    "knee_r_flex": 0.0,
    "ankle_l_flex": 0.0,
    "ankle_r_flex": 0.0,
    "shoulder_l_flex": 0.0,
    "shoulder_r_flex": 0.0,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}


# ---------- SQUAT ----------

SQUAT_OBJECTIVE = ExerciseObjective(
    name="back_squat",
    start_pose=dict(_STANDING_POSE),
    end_pose=dict(_STANDING_POSE),
    phases=[
        Phase(
            "descent_start",
            0.0,
            dict(_STANDING_POSE),
        ),
        Phase(
            "half_depth",
            0.25,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -1.05,
                "knee_r_flex": -1.05,
                "ankle_l_flex": 0.3,
                "ankle_r_flex": 0.3,
            },
        ),
        Phase(
            "bottom",
            0.5,
            {
                "hip_l_flex": 2.09,
                "hip_r_flex": 2.09,
                "knee_l_flex": -2.09,
                "knee_r_flex": -2.09,
                "ankle_l_flex": 0.6,
                "ankle_r_flex": 0.6,
            },
        ),
        Phase(
            "drive",
            0.75,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -1.05,
                "knee_r_flex": -1.05,
                "ankle_l_flex": 0.3,
                "ankle_r_flex": 0.3,
            },
        ),
        Phase(
            "lockout",
            1.0,
            dict(_STANDING_POSE),
        ),
    ],
    bar_path_constraint="vertical",
    balance_mode="bilateral_stance",
)


# ---------- DEADLIFT ----------

_DEADLIFT_BOTTOM: dict[str, float] = {
    "hip_l_flex": 1.40,
    "hip_r_flex": 1.40,
    "knee_l_flex": -1.05,
    "knee_r_flex": -1.05,
    "ankle_l_flex": 0.35,
    "ankle_r_flex": 0.35,
    "shoulder_l_flex": -0.17,
    "shoulder_r_flex": -0.17,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

DEADLIFT_OBJECTIVE = ExerciseObjective(
    name="deadlift",
    start_pose=dict(_DEADLIFT_BOTTOM),
    end_pose=dict(_STANDING_POSE),
    phases=[
        Phase("floor", 0.0, dict(_DEADLIFT_BOTTOM)),
        Phase(
            "below_knee",
            0.25,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -0.52,
                "knee_r_flex": -0.52,
                "ankle_l_flex": 0.17,
                "ankle_r_flex": 0.17,
            },
        ),
        Phase(
            "above_knee",
            0.5,
            {
                "hip_l_flex": 0.70,
                "hip_r_flex": 0.70,
                "knee_l_flex": -0.26,
                "knee_r_flex": -0.26,
                "ankle_l_flex": 0.09,
                "ankle_r_flex": 0.09,
            },
        ),
        Phase(
            "lockout",
            1.0,
            dict(_STANDING_POSE),
        ),
    ],
    bar_path_constraint="vertical",
    balance_mode="bilateral_stance",
)


# ---------- BENCH PRESS ----------

_BENCH_LOCKOUT: dict[str, float] = {
    "shoulder_l_flex": 1.57,
    "shoulder_r_flex": 1.57,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

_BENCH_BOTTOM: dict[str, float] = {
    "shoulder_l_flex": 0.52,
    "shoulder_r_flex": 0.52,
    "elbow_l_flex": -1.57,
    "elbow_r_flex": -1.57,
}

BENCH_PRESS_OBJECTIVE = ExerciseObjective(
    name="bench_press",
    start_pose=dict(_BENCH_LOCKOUT),
    end_pose=dict(_BENCH_LOCKOUT),
    phases=[
        Phase("lockout_start", 0.0, dict(_BENCH_LOCKOUT)),
        Phase(
            "descent",
            0.25,
            {
                "shoulder_l_flex": 1.05,
                "shoulder_r_flex": 1.05,
                "elbow_l_flex": -0.79,
                "elbow_r_flex": -0.79,
            },
        ),
        Phase("chest", 0.5, dict(_BENCH_BOTTOM)),
        Phase(
            "press",
            0.75,
            {
                "shoulder_l_flex": 1.05,
                "shoulder_r_flex": 1.05,
                "elbow_l_flex": -0.79,
                "elbow_r_flex": -0.79,
            },
        ),
        Phase("lockout_finish", 1.0, dict(_BENCH_LOCKOUT)),
    ],
    bar_path_constraint="j_curve",
    balance_mode="supine",
)


# ---------- SNATCH ----------

_SNATCH_START: dict[str, float] = {
    "hip_l_flex": 1.40,
    "hip_r_flex": 1.40,
    "knee_l_flex": -1.22,
    "knee_r_flex": -1.22,
    "ankle_l_flex": 0.44,
    "ankle_r_flex": 0.44,
    "shoulder_l_flex": -0.35,
    "shoulder_r_flex": -0.35,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

_SNATCH_OVERHEAD: dict[str, float] = {
    "hip_l_flex": 2.09,
    "hip_r_flex": 2.09,
    "knee_l_flex": -2.09,
    "knee_r_flex": -2.09,
    "ankle_l_flex": 0.6,
    "ankle_r_flex": 0.6,
    "shoulder_l_flex": 3.14,
    "shoulder_r_flex": 3.14,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

SNATCH_OBJECTIVE = ExerciseObjective(
    name="snatch",
    start_pose=dict(_SNATCH_START),
    end_pose=dict(_STANDING_POSE),
    phases=[
        Phase("first_pull", 0.0, dict(_SNATCH_START)),
        Phase(
            "transition",
            0.20,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -0.87,
                "knee_r_flex": -0.87,
                "ankle_l_flex": 0.26,
                "ankle_r_flex": 0.26,
                "shoulder_l_flex": 0.17,
                "shoulder_r_flex": 0.17,
                "elbow_l_flex": 0.0,
                "elbow_r_flex": 0.0,
            },
        ),
        Phase(
            "second_pull",
            0.40,
            {
                "hip_l_flex": 0.17,
                "hip_r_flex": 0.17,
                "knee_l_flex": -0.17,
                "knee_r_flex": -0.17,
                "ankle_l_flex": -0.35,
                "ankle_r_flex": -0.35,
                "shoulder_l_flex": 1.57,
                "shoulder_r_flex": 1.57,
                "elbow_l_flex": -1.05,
                "elbow_r_flex": -1.05,
            },
        ),
        Phase(
            "turnover",
            0.55,
            {
                "hip_l_flex": 0.70,
                "hip_r_flex": 0.70,
                "knee_l_flex": -0.70,
                "knee_r_flex": -0.70,
                "shoulder_l_flex": 2.62,
                "shoulder_r_flex": 2.62,
                "elbow_l_flex": -0.52,
                "elbow_r_flex": -0.52,
            },
        ),
        Phase("catch", 0.70, dict(_SNATCH_OVERHEAD)),
        Phase(
            "recovery",
            1.0,
            {
                **_STANDING_POSE,
                "shoulder_l_flex": 3.14,
                "shoulder_r_flex": 3.14,
            },
        ),
    ],
    bar_path_constraint="s_curve",
    balance_mode="bilateral_stance",
)


# ---------- CLEAN AND JERK ----------

_CLEAN_START: dict[str, float] = {
    "hip_l_flex": 1.40,
    "hip_r_flex": 1.40,
    "knee_l_flex": -1.05,
    "knee_r_flex": -1.05,
    "ankle_l_flex": 0.35,
    "ankle_r_flex": 0.35,
    "shoulder_l_flex": -0.17,
    "shoulder_r_flex": -0.17,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

_FRONT_RACK: dict[str, float] = {
    "hip_l_flex": 0.0,
    "hip_r_flex": 0.0,
    "knee_l_flex": 0.0,
    "knee_r_flex": 0.0,
    "ankle_l_flex": 0.0,
    "ankle_r_flex": 0.0,
    "shoulder_l_flex": 1.57,
    "shoulder_r_flex": 1.57,
    "elbow_l_flex": -2.36,
    "elbow_r_flex": -2.36,
}

_JERK_DIP: dict[str, float] = {
    "hip_l_flex": 0.35,
    "hip_r_flex": 0.35,
    "knee_l_flex": -0.52,
    "knee_r_flex": -0.52,
    "ankle_l_flex": 0.17,
    "ankle_r_flex": 0.17,
    "shoulder_l_flex": 1.57,
    "shoulder_r_flex": 1.57,
    "elbow_l_flex": -2.36,
    "elbow_r_flex": -2.36,
}

_JERK_CATCH: dict[str, float] = {
    "hip_l_flex": 0.35,
    "hip_r_flex": 0.35,
    "knee_l_flex": -0.52,
    "knee_r_flex": -0.52,
    "ankle_l_flex": 0.17,
    "ankle_r_flex": 0.17,
    "shoulder_l_flex": 3.14,
    "shoulder_r_flex": 3.14,
    "elbow_l_flex": 0.0,
    "elbow_r_flex": 0.0,
}

_CJ_FINISH: dict[str, float] = {
    **_STANDING_POSE,
    "shoulder_l_flex": 3.14,
    "shoulder_r_flex": 3.14,
}

CLEAN_AND_JERK_OBJECTIVE = ExerciseObjective(
    name="clean_and_jerk",
    start_pose=dict(_CLEAN_START),
    end_pose=dict(_CJ_FINISH),
    phases=[
        # Clean phases
        Phase("clean_pull", 0.0, dict(_CLEAN_START)),
        Phase(
            "clean_transition",
            0.15,
            {
                "hip_l_flex": 1.05,
                "hip_r_flex": 1.05,
                "knee_l_flex": -0.52,
                "knee_r_flex": -0.52,
                "ankle_l_flex": 0.17,
                "ankle_r_flex": 0.17,
                "shoulder_l_flex": 0.35,
                "shoulder_r_flex": 0.35,
                "elbow_l_flex": -0.52,
                "elbow_r_flex": -0.52,
            },
        ),
        Phase(
            "clean_extension",
            0.30,
            {
                "hip_l_flex": 0.17,
                "hip_r_flex": 0.17,
                "knee_l_flex": -0.17,
                "knee_r_flex": -0.17,
                "ankle_l_flex": -0.26,
                "ankle_r_flex": -0.26,
                "shoulder_l_flex": 0.87,
                "shoulder_r_flex": 0.87,
                "elbow_l_flex": -1.57,
                "elbow_r_flex": -1.57,
            },
        ),
        Phase("clean_catch", 0.45, dict(_FRONT_RACK)),
        # Jerk phases
        Phase("jerk_dip", 0.55, dict(_JERK_DIP)),
        Phase(
            "jerk_drive",
            0.70,
            {
                "hip_l_flex": -0.09,
                "hip_r_flex": -0.09,
                "knee_l_flex": 0.0,
                "knee_r_flex": 0.0,
                "ankle_l_flex": -0.26,
                "ankle_r_flex": -0.26,
                "shoulder_l_flex": 2.36,
                "shoulder_r_flex": 2.36,
                "elbow_l_flex": -1.05,
                "elbow_r_flex": -1.05,
            },
        ),
        Phase("jerk_catch", 0.85, dict(_JERK_CATCH)),
        Phase("recovery", 1.0, dict(_CJ_FINISH)),
    ],
    bar_path_constraint="s_curve",
    balance_mode="bilateral_stance",
)

# All objectives by exercise name for programmatic access.
OBJECTIVE_REGISTRY: dict[str, ExerciseObjective] = {
    "back_squat": SQUAT_OBJECTIVE,
    "squat": SQUAT_OBJECTIVE,
    "deadlift": DEADLIFT_OBJECTIVE,
    "bench_press": BENCH_PRESS_OBJECTIVE,
    "snatch": SNATCH_OBJECTIVE,
    "clean_and_jerk": CLEAN_AND_JERK_OBJECTIVE,
}

# Canonical mapping used by get_exercise_objective().
EXERCISE_OBJECTIVES: dict[str, ExerciseObjective] = OBJECTIVE_REGISTRY


def get_exercise_objective(name: str) -> ExerciseObjective:
    """Return an exercise objective by name.

    Args:
        name: Exercise name (e.g. 'squat', 'deadlift').

    Returns:
        The matching ExerciseObjective.

    Raises:
        ValueError: If *name* is not a recognised exercise.
    """
    try:
        return EXERCISE_OBJECTIVES[name]
    except KeyError:
        available = ", ".join(sorted(EXERCISE_OBJECTIVES))
        raise ValueError(f"Unknown exercise '{name}'. Available: {available}") from None
