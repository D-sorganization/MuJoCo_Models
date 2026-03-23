"""Performance benchmarks for model generation (#14).

Uses pytest-benchmark to track model build times. These are informational;
they do not enforce performance thresholds but establish a baseline.

Run with: python3 -m pytest tests/unit/test_benchmarks.py --benchmark-only -p no:xdist
"""

from __future__ import annotations

import pytest

from mujoco_models.exercises.base import ExerciseConfig
from mujoco_models.exercises.bench_press.bench_press_model import BenchPressModelBuilder
from mujoco_models.exercises.clean_and_jerk.clean_and_jerk_model import (
    CleanAndJerkModelBuilder,
)
from mujoco_models.exercises.deadlift.deadlift_model import DeadliftModelBuilder
from mujoco_models.exercises.snatch.snatch_model import SnatchModelBuilder
from mujoco_models.exercises.squat.squat_model import SquatModelBuilder


@pytest.mark.benchmark
class TestModelBuildBenchmarks:
    """Benchmark the build time for each exercise model."""

    def test_squat_build(self, benchmark: object) -> None:
        config = ExerciseConfig()
        builder = SquatModelBuilder(config)
        benchmark(builder.build)

    def test_bench_press_build(self, benchmark: object) -> None:
        config = ExerciseConfig()
        builder = BenchPressModelBuilder(config)
        benchmark(builder.build)

    def test_deadlift_build(self, benchmark: object) -> None:
        config = ExerciseConfig()
        builder = DeadliftModelBuilder(config)
        benchmark(builder.build)

    def test_snatch_build(self, benchmark: object) -> None:
        config = ExerciseConfig()
        builder = SnatchModelBuilder(config)
        benchmark(builder.build)

    def test_clean_and_jerk_build(self, benchmark: object) -> None:
        config = ExerciseConfig()
        builder = CleanAndJerkModelBuilder(config)
        benchmark(builder.build)
