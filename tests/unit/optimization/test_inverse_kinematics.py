# SPDX-License-Identifier: MIT
"""Tests for the IK keyframe generator.

Covers ``solve_ik_keyframes`` for reachable targets, convergence through
phase interpolation, boundary values, and DbC failures on invalid input.
"""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_models.optimization.exercise_objectives import (
    EXERCISE_OBJECTIVES,
)
from mujoco_models.optimization.inverse_kinematics import (
    solve_ik_keyframes,
)

KNOWN_EXERCISES = [
    "squat",
    "deadlift",
    "bench_press",
    "snatch",
    "clean_and_jerk",
    "gait",
    "sit_to_stand",
]


class TestSolveIKKeyframesShape:
    """Output shape must match (n_frames, n_joints)."""

    @pytest.mark.parametrize("name", KNOWN_EXERCISES)
    def test_default_frame_count_shape(self, name: str) -> None:
        keyframes = solve_ik_keyframes(name)
        n_joints = len(EXERCISE_OBJECTIVES[name].joint_names)
        assert keyframes.shape == (50, n_joints)

    @pytest.mark.parametrize("n_frames", [2, 5, 50, 200])
    def test_custom_frame_count_shape(self, n_frames: int) -> None:
        keyframes = solve_ik_keyframes("squat", n_frames=n_frames)
        n_joints = len(EXERCISE_OBJECTIVES["squat"].joint_names)
        assert keyframes.shape == (n_frames, n_joints)

    def test_returns_numpy_array(self) -> None:
        keyframes = solve_ik_keyframes("squat", n_frames=3)
        assert isinstance(keyframes, np.ndarray)
        assert keyframes.dtype.kind == "f"


class TestSolveIKKeyframesInterpolation:
    """Endpoints must match the first/last phase targets."""

    def test_first_frame_matches_first_phase(self) -> None:
        objective = EXERCISE_OBJECTIVES["squat"]
        joint_names = objective.joint_names
        first_phase = objective.phases[0]

        keyframes = solve_ik_keyframes("squat", n_frames=10)

        expected = np.array(
            [first_phase.target_joints.get(name, 0.0) for name in joint_names]
        )
        assert keyframes[0] == pytest.approx(expected)

    def test_last_frame_matches_last_phase(self) -> None:
        objective = EXERCISE_OBJECTIVES["squat"]
        joint_names = objective.joint_names
        last_phase = objective.phases[-1]

        keyframes = solve_ik_keyframes("squat", n_frames=10)

        expected = np.array(
            [last_phase.target_joints.get(name, 0.0) for name in joint_names]
        )
        assert keyframes[-1] == pytest.approx(expected)

    def test_interpolation_is_smooth(self) -> None:
        """Adjacent keyframes should differ by a bounded step."""
        keyframes = solve_ik_keyframes("squat", n_frames=50)
        diffs = np.abs(np.diff(keyframes, axis=0))
        # No single step should exceed the full span of any joint.
        joint_spans = keyframes.max(axis=0) - keyframes.min(axis=0)
        per_joint_max_step = diffs.max(axis=0)
        # Each per-joint step is at most the joint's full span.
        assert np.all(per_joint_max_step <= joint_spans + 1e-9)

    def test_minimum_frame_count_returns_endpoints(self) -> None:
        """With n_frames=2 output is [first_phase, last_phase]."""
        objective = EXERCISE_OBJECTIVES["squat"]
        joint_names = objective.joint_names
        keyframes = solve_ik_keyframes("squat", n_frames=2)

        expected_first = np.array(
            [objective.phases[0].target_joints.get(n, 0.0) for n in joint_names]
        )
        expected_last = np.array(
            [objective.phases[-1].target_joints.get(n, 0.0) for n in joint_names]
        )
        assert keyframes[0] == pytest.approx(expected_first)
        assert keyframes[1] == pytest.approx(expected_last)

    def test_deterministic_output(self) -> None:
        """Calling twice must produce identical arrays."""
        a = solve_ik_keyframes("deadlift", n_frames=17)
        b = solve_ik_keyframes("deadlift", n_frames=17)
        assert np.array_equal(a, b)


class TestSolveIKKeyframesJointOrder:
    """Output columns must follow sorted joint_names order."""

    def test_joint_order_is_sorted(self) -> None:
        objective = EXERCISE_OBJECTIVES["squat"]
        joint_names = objective.joint_names
        assert joint_names == sorted(joint_names)

        keyframes = solve_ik_keyframes("squat", n_frames=3)
        assert keyframes.shape[1] == len(joint_names)


class TestSolveIKKeyframesDbC:
    """Design-by-contract: reject invalid inputs with ValueError."""

    def test_unknown_exercise_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown exercise"):
            solve_ik_keyframes("nonexistent_exercise")

    @pytest.mark.parametrize("n_frames", [0, 1, -5])
    def test_too_few_frames_raises(self, n_frames: int) -> None:
        with pytest.raises(ValueError, match="n_frames must be >= 2"):
            solve_ik_keyframes("squat", n_frames=n_frames)

    def test_empty_exercise_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown exercise"):
            solve_ik_keyframes("", n_frames=5)
