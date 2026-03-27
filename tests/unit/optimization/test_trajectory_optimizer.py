"""Tests for trajectory optimizer cost functions and dataclass validation."""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_models.optimization.trajectory_optimizer import (
    TrajectoryConfig,
    TrajectoryResult,
    compute_balance_cost,
    compute_bar_path_cost,
)

# ---------------------------------------------------------------------------
# TrajectoryConfig validation
# ---------------------------------------------------------------------------


class TestTrajectoryConfig:
    def test_defaults_valid(self) -> None:
        cfg = TrajectoryConfig()
        assert cfg.n_timesteps == 100
        assert cfg.dt > 0

    def test_total_time(self) -> None:
        cfg = TrajectoryConfig(n_timesteps=200, dt=0.005)
        assert cfg.total_time == pytest.approx(1.0)

    def test_rejects_zero_timesteps(self) -> None:
        with pytest.raises(ValueError, match="n_timesteps"):
            TrajectoryConfig(n_timesteps=0)

    def test_rejects_negative_dt(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            TrajectoryConfig(dt=-0.01)

    def test_rejects_zero_dt(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            TrajectoryConfig(dt=0.0)

    def test_rejects_zero_max_iterations(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            TrajectoryConfig(max_iterations=0)

    def test_rejects_zero_convergence_tol(self) -> None:
        with pytest.raises(ValueError, match="convergence_tol"):
            TrajectoryConfig(convergence_tol=0.0)

    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="control_weight"):
            TrajectoryConfig(control_weight=-1.0)

    def test_rejects_negative_balance_weight(self) -> None:
        with pytest.raises(ValueError, match="balance_weight"):
            TrajectoryConfig(balance_weight=-0.1)


# ---------------------------------------------------------------------------
# TrajectoryResult validation
# ---------------------------------------------------------------------------


class TestTrajectoryResult:
    def _make_result(self, n_t: int = 10, n_j: int = 3) -> TrajectoryResult:
        return TrajectoryResult(
            joint_positions=np.zeros((n_t, n_j)),
            joint_velocities=np.zeros((n_t, n_j)),
            joint_torques=np.zeros((n_t, n_j)),
            time=np.linspace(0, 1, n_t),
            cost=0.0,
            converged=True,
            iterations=1,
        )

    def test_valid_result(self) -> None:
        result = self._make_result()
        assert result.converged

    def test_position_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="joint_positions"):
            TrajectoryResult(
                joint_positions=np.zeros((5, 3)),
                joint_velocities=np.zeros((10, 3)),
                joint_torques=np.zeros((10, 3)),
                time=np.linspace(0, 1, 10),
                cost=0.0,
                converged=True,
                iterations=1,
            )

    def test_velocity_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="joint_velocities"):
            TrajectoryResult(
                joint_positions=np.zeros((10, 3)),
                joint_velocities=np.zeros((5, 3)),
                joint_torques=np.zeros((10, 3)),
                time=np.linspace(0, 1, 10),
                cost=0.0,
                converged=True,
                iterations=1,
            )

    def test_torque_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="joint_torques"):
            TrajectoryResult(
                joint_positions=np.zeros((10, 3)),
                joint_velocities=np.zeros((10, 3)),
                joint_torques=np.zeros((5, 3)),
                time=np.linspace(0, 1, 10),
                cost=0.0,
                converged=True,
                iterations=1,
            )


# ---------------------------------------------------------------------------
# compute_balance_cost
# ---------------------------------------------------------------------------


class TestComputeBalanceCost:
    @pytest.fixture()
    def triangle(self) -> np.ndarray:
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    def test_com_inside_returns_zero(self, triangle: np.ndarray) -> None:
        com = np.array([0.5, 0.3, 1.0])  # inside the triangle
        assert compute_balance_cost(com, triangle) == 0.0

    def test_com_outside_returns_positive(self, triangle: np.ndarray) -> None:
        com = np.array([5.0, 5.0, 1.0])  # far outside
        cost = compute_balance_cost(com, triangle)
        assert cost > 0.0

    def test_bad_com_shape_raises(self, triangle: np.ndarray) -> None:
        with pytest.raises(ValueError, match="com_position"):
            compute_balance_cost(np.array([1.0, 2.0]), triangle)

    def test_bad_bos_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="base_of_support"):
            compute_balance_cost(np.array([0, 0, 0.0]), np.array([1.0, 2.0]))

    def test_too_few_vertices_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            compute_balance_cost(
                np.array([0.0, 0.0, 0.0]),
                np.array([[0.0, 0.0], [1.0, 0.0]]),
            )

    def test_non_finite_com_raises(self, triangle: np.ndarray) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            compute_balance_cost(np.array([np.nan, 0.0, 0.0]), triangle)

    def test_non_finite_bos_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            compute_balance_cost(
                np.array([0.0, 0.0, 0.0]),
                np.array([[np.inf, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            )


# ---------------------------------------------------------------------------
# compute_bar_path_cost
# ---------------------------------------------------------------------------


class TestComputeBarPathCost:
    def test_identical_paths_zero_cost(self) -> None:
        path = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        assert compute_bar_path_cost(path, path) == pytest.approx(0.0)

    def test_horizontal_deviation(self) -> None:
        bar = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
        target = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        cost = compute_bar_path_cost(bar, target)
        assert cost == pytest.approx(1.0)

    def test_vertical_only_difference_zero_cost(self) -> None:
        bar = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 10.0]])
        target = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        assert compute_bar_path_cost(bar, target) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_bar_path_cost(np.zeros((5, 3)), np.zeros((3, 3)))

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="bar_position"):
            compute_bar_path_cost(np.zeros((5,)), np.zeros((5,)))

    def test_non_finite_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            compute_bar_path_cost(
                np.array([[np.nan, 0, 0]]),
                np.array([[0.0, 0, 0]]),
            )
