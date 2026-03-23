"""Tests for the exercise registry (#23)."""

from __future__ import annotations

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.exercises.base import ExerciseModelBuilder


class TestExerciseRegistry:
    def test_all_entries_are_builder_subclasses(self) -> None:
        for name, cls in EXERCISE_REGISTRY.items():
            assert issubclass(cls, ExerciseModelBuilder), (
                f"{name} -> {cls} is not an ExerciseModelBuilder"
            )

    def test_minimum_exercises(self) -> None:
        expected = {"squat", "bench_press", "deadlift", "snatch", "clean_and_jerk"}
        assert expected.issubset(set(EXERCISE_REGISTRY.keys()))

    def test_back_squat_alias(self) -> None:
        assert EXERCISE_REGISTRY["back_squat"] is EXERCISE_REGISTRY["squat"]

    def test_all_builders_instantiate(self) -> None:
        for cls in EXERCISE_REGISTRY.values():
            builder = cls()
            assert hasattr(builder, "build")
            assert hasattr(builder, "exercise_name")
