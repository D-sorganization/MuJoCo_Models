"""Tests for TrajectoryConfig and TrajectoryResult validation edge cases.

Covers issue #69: ensure all validation branches in __post_init__ are
exercised with explicit tests for boundary values and error messages.
"""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_models.optimization.exercise_objectives import (
    ExerciseObjective,
    Phase,
)
from mujoco_models.optimization.trajectory_optimizer import (
    TrajectoryConfig,
    TrajectoryResult,
)

# ---------------------------------------------------------------------------
# TrajectoryConfig boundary and edge-case validation
# ---------------------------------------------------------------------------


class TestTrajectoryConfigEdgeCases:
    """Edge-case validation for TrajectoryConfig."""

    def test_minimum_valid_timesteps(self) -> None:
        cfg = TrajectoryConfig(n_timesteps=1)
        assert cfg.n_timesteps == 1

    def test_negative_timesteps_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_timesteps must be >= 1"):
            TrajectoryConfig(n_timesteps=-5)

    def test_very_small_dt_accepted(self) -> None:
        cfg = TrajectoryConfig(dt=1e-10)
        assert cfg.dt == pytest.approx(1e-10)

    def test_negative_dt_rejected(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            TrajectoryConfig(dt=-1e-5)

    def test_negative_max_iterations_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            TrajectoryConfig(max_iterations=-1)

    def test_minimum_valid_max_iterations(self) -> None:
        cfg = TrajectoryConfig(max_iterations=1)
        assert cfg.max_iterations == 1

    def test_negative_convergence_tol_rejected(self) -> None:
        with pytest.raises(ValueError, match="convergence_tol must be positive"):
            TrajectoryConfig(convergence_tol=-0.001)

    def test_very_small_convergence_tol_accepted(self) -> None:
        cfg = TrajectoryConfig(convergence_tol=1e-15)
        assert cfg.convergence_tol == pytest.approx(1e-15)

    def test_zero_control_weight_accepted(self) -> None:
        cfg = TrajectoryConfig(control_weight=0.0)
        assert cfg.control_weight == 0.0

    def test_negative_state_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="state_weight"):
            TrajectoryConfig(state_weight=-0.5)

    def test_negative_terminal_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="terminal_weight"):
            TrajectoryConfig(terminal_weight=-1.0)

    def test_zero_state_weight_accepted(self) -> None:
        cfg = TrajectoryConfig(state_weight=0.0)
        assert cfg.state_weight == 0.0

    def test_zero_terminal_weight_accepted(self) -> None:
        cfg = TrajectoryConfig(terminal_weight=0.0)
        assert cfg.terminal_weight == 0.0

    def test_zero_balance_weight_accepted(self) -> None:
        cfg = TrajectoryConfig(balance_weight=0.0)
        assert cfg.balance_weight == 0.0

    def test_total_time_computation(self) -> None:
        cfg = TrajectoryConfig(n_timesteps=50, dt=0.02)
        assert cfg.total_time == pytest.approx(1.0)

    def test_all_valid_custom_values(self) -> None:
        cfg = TrajectoryConfig(
            n_timesteps=500,
            dt=0.005,
            max_iterations=1000,
            convergence_tol=1e-6,
            control_weight=0.01,
            state_weight=2.0,
            terminal_weight=20.0,
            balance_weight=10.0,
        )
        assert cfg.total_time == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# TrajectoryResult validation
# ---------------------------------------------------------------------------


class TestTrajectoryResultEdgeCases:
    """Edge-case validation for TrajectoryResult."""

    def test_single_timestep_valid(self) -> None:
        result = TrajectoryResult(
            joint_positions=np.zeros((1, 3)),
            joint_velocities=np.zeros((1, 3)),
            joint_torques=np.zeros((1, 3)),
            time=np.array([0.0]),
            cost=0.0,
            converged=True,
            iterations=1,
        )
        assert result.converged

    def test_different_joint_counts_valid(self) -> None:
        """positions and torques can have different column counts."""
        result = TrajectoryResult(
            joint_positions=np.zeros((5, 10)),
            joint_velocities=np.zeros((5, 10)),
            joint_torques=np.zeros((5, 4)),
            time=np.linspace(0, 1, 5),
            cost=1.5,
            converged=False,
            iterations=200,
        )
        assert not result.converged
        assert result.cost == pytest.approx(1.5)

    def test_positions_wrong_timesteps_rejected(self) -> None:
        with pytest.raises(ValueError, match="joint_positions has 3 timesteps"):
            TrajectoryResult(
                joint_positions=np.zeros((3, 2)),
                joint_velocities=np.zeros((5, 2)),
                joint_torques=np.zeros((5, 2)),
                time=np.linspace(0, 1, 5),
                cost=0.0,
                converged=True,
                iterations=1,
            )

    def test_velocities_wrong_timesteps_rejected(self) -> None:
        with pytest.raises(ValueError, match="joint_velocities has 7 timesteps"):
            TrajectoryResult(
                joint_positions=np.zeros((5, 2)),
                joint_velocities=np.zeros((7, 2)),
                joint_torques=np.zeros((5, 2)),
                time=np.linspace(0, 1, 5),
                cost=0.0,
                converged=True,
                iterations=1,
            )

    def test_torques_wrong_timesteps_rejected(self) -> None:
        with pytest.raises(ValueError, match="joint_torques has 2 timesteps"):
            TrajectoryResult(
                joint_positions=np.zeros((5, 2)),
                joint_velocities=np.zeros((5, 2)),
                joint_torques=np.zeros((2, 2)),
                time=np.linspace(0, 1, 5),
                cost=0.0,
                converged=True,
                iterations=1,
            )


# ---------------------------------------------------------------------------
# Phase and ExerciseObjective validation
# ---------------------------------------------------------------------------


class TestPhaseValidation:
    """Validate Phase dataclass __post_init__ constraints."""

    def test_fraction_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="Phase fraction must be in"):
            Phase("bad", -0.1)

    def test_fraction_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="Phase fraction must be in"):
            Phase("bad", 1.01)

    def test_fraction_zero_accepted(self) -> None:
        p = Phase("start", 0.0)
        assert p.fraction == 0.0

    def test_fraction_one_accepted(self) -> None:
        p = Phase("end", 1.0)
        assert p.fraction == 1.0

    def test_empty_target_joints_accepted(self) -> None:
        p = Phase("neutral", 0.5)
        assert p.target_joints == {}


class TestExerciseObjectiveValidation:
    """Validate ExerciseObjective dataclass __post_init__ constraints."""

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Exercise name must not be empty"):
            ExerciseObjective(name="")

    def test_invalid_bar_path_constraint_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid bar_path_constraint"):
            ExerciseObjective(name="test", bar_path_constraint="zigzag")

    def test_invalid_balance_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid balance_mode"):
            ExerciseObjective(name="test", balance_mode="handstand")

    def test_non_monotonic_phases_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            ExerciseObjective(
                name="test",
                phases=[Phase("a", 0.5), Phase("b", 0.3)],
            )

    def test_valid_objective_with_all_bar_paths(self) -> None:
        for constraint in ("vertical", "j_curve", "s_curve", "none"):
            obj = ExerciseObjective(name="test", bar_path_constraint=constraint)
            assert obj.bar_path_constraint == constraint

    def test_valid_objective_with_all_balance_modes(self) -> None:
        for mode in (
            "bilateral_stance",
            "split_stance",
            "supine",
            "alternating_stance",
        ):
            obj = ExerciseObjective(name="test", balance_mode=mode)
            assert obj.balance_mode == mode

    def test_get_phase_found(self) -> None:
        obj = ExerciseObjective(
            name="test",
            phases=[Phase("start", 0.0), Phase("end", 1.0)],
        )
        assert obj.get_phase("start").fraction == 0.0

    def test_get_phase_not_found_raises_key_error(self) -> None:
        obj = ExerciseObjective(
            name="test",
            phases=[Phase("start", 0.0)],
        )
        with pytest.raises(KeyError, match="No phase named 'missing'"):
            obj.get_phase("missing")

    def test_n_phases_property(self) -> None:
        obj = ExerciseObjective(
            name="test",
            phases=[Phase("a", 0.0), Phase("b", 0.5), Phase("c", 1.0)],
        )
        assert obj.n_phases == 3

    def test_joint_names_aggregated(self) -> None:
        obj = ExerciseObjective(
            name="test",
            start_pose={"hip_flex": 0.0},
            end_pose={"knee_flex": 0.0},
            phases=[Phase("mid", 0.5, {"ankle_flex": 0.1})],
        )
        assert obj.joint_names == ["ankle_flex", "hip_flex", "knee_flex"]
