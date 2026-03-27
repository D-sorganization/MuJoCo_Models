"""Tests for exercise-specific optimization objectives."""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_models.optimization.exercise_objectives import (
    EXERCISE_OBJECTIVES,
    ExerciseObjective,
    get_exercise_objective,
)
from mujoco_models.optimization.trajectory_optimizer import (
    interpolate_phases,
)

EXPECTED_NAMES = {
    "back_squat",
    "squat",
    "deadlift",
    "bench_press",
    "snatch",
    "clean_and_jerk",
    "gait",
    "sit_to_stand",
}


class TestExerciseObjectivesRegistry:
    """All objectives must be registered."""

    def test_all_objectives_present(self):
        assert set(EXERCISE_OBJECTIVES.keys()) == EXPECTED_NAMES

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_objective_is_correct_type(self, name: str):
        obj = EXERCISE_OBJECTIVES[name]
        assert isinstance(obj, ExerciseObjective)


class TestPhaseConstraints:
    """Phase fractions must be in [0, 1] and monotonically increasing."""

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_fractions_in_range(self, name: str):
        for phase in EXERCISE_OBJECTIVES[name].phases:
            assert 0.0 <= phase.fraction <= 1.0, (
                f"{name}/{phase.name}: fraction {phase.fraction} out of range"
            )

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_fractions_monotonic(self, name: str):
        fracs = [p.fraction for p in EXERCISE_OBJECTIVES[name].phases]
        for a, b in zip(fracs, fracs[1:], strict=False):
            assert a < b, f"{name}: fractions not strictly increasing ({a} >= {b})"

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_starts_at_zero_ends_at_one(self, name: str):
        phases = EXERCISE_OBJECTIVES[name].phases
        assert phases[0].fraction == 0.0
        assert phases[-1].fraction == 1.0


class TestGetExerciseObjective:
    """Lookup helper must work and raise on unknown names."""

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_valid_lookup(self, name: str):
        obj = get_exercise_objective(name)
        assert obj is EXERCISE_OBJECTIVES[name]

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown exercise"):
            get_exercise_objective("bicep_curl")


# Use distinct objectives (squat and back_squat alias the same object).
_INTERP_NAMES = [
    "deadlift",
    "bench_press",
    "snatch",
    "clean_and_jerk",
    "squat",
    "gait",
    "sit_to_stand",
]


class TestInterpolatePhases:
    """Keyframe interpolation: correct shape and boundary values."""

    @pytest.mark.parametrize("name", _INTERP_NAMES)
    def test_output_shape(self, name: str):
        obj = EXERCISE_OBJECTIVES[name]
        n_frames = 60
        result = interpolate_phases(obj, n_frames=n_frames)
        all_joints: set[str] = set()
        for phase in obj.phases:
            all_joints.update(phase.target_joints.keys())
        assert result.shape == (n_frames, len(all_joints))

    @pytest.mark.parametrize("name", _INTERP_NAMES)
    def test_start_matches_first_phase(self, name: str):
        obj = EXERCISE_OBJECTIVES[name]
        result = interpolate_phases(obj, n_frames=50)
        joint_names = sorted({jn for phase in obj.phases for jn in phase.target_joints})
        first_phase = obj.phases[0]
        for k, jn in enumerate(joint_names):
            expected = first_phase.target_joints.get(jn, 0.0)
            np.testing.assert_allclose(
                result[0, k],
                expected,
                atol=1e-8,
                err_msg=f"{name}: start mismatch for {jn}",
            )

    @pytest.mark.parametrize("name", _INTERP_NAMES)
    def test_end_matches_last_phase(self, name: str):
        obj = EXERCISE_OBJECTIVES[name]
        result = interpolate_phases(obj, n_frames=50)
        joint_names = sorted({jn for phase in obj.phases for jn in phase.target_joints})
        last_phase = obj.phases[-1]
        for k, jn in enumerate(joint_names):
            expected = last_phase.target_joints.get(jn, 0.0)
            np.testing.assert_allclose(
                result[-1, k],
                expected,
                atol=1e-8,
                err_msg=f"{name}: end mismatch for {jn}",
            )
