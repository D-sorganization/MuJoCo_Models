"""Trajectory optimization for barbell exercise models using MuJoCo.

Implements cost functions and trajectory optimization configuration for
planning biomechanically optimal barbell exercise movements.

Design by Contract:
    - All array inputs are validated for shape and finite values.
    - Config values are validated for physical plausibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .exercise_objectives import ExerciseObjective

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectoryConfig:
    """Configuration for trajectory optimization.

    Attributes:
        n_timesteps: Number of discrete time steps in trajectory.
        dt: Time step duration in seconds.
        max_iterations: Maximum optimizer iterations before stopping.
        convergence_tol: Cost change threshold for convergence.
        control_weight: Penalty weight on joint torque magnitudes.
        state_weight: Weight for tracking target trajectory.
        terminal_weight: Weight for reaching the final target pose.
        balance_weight: Weight for keeping CoM over base of support.
    """

    n_timesteps: int = 100
    dt: float = 0.01
    max_iterations: int = 200
    convergence_tol: float = 1e-4
    control_weight: float = 1e-3
    state_weight: float = 1.0
    terminal_weight: float = 10.0
    balance_weight: float = 5.0

    def __post_init__(self) -> None:
        """Validate configuration values are physically plausible."""
        if self.n_timesteps < 1:
            msg = f"n_timesteps must be >= 1, got {self.n_timesteps}"
            raise ValueError(msg)
        if self.dt <= 0.0:
            msg = f"dt must be positive, got {self.dt}"
            raise ValueError(msg)
        if self.max_iterations < 1:
            msg = f"max_iterations must be >= 1, got {self.max_iterations}"
            raise ValueError(msg)
        if self.convergence_tol <= 0.0:
            msg = f"convergence_tol must be positive, got {self.convergence_tol}"
            raise ValueError(msg)
        _validate_non_negative_weight("control_weight", self.control_weight)
        _validate_non_negative_weight("state_weight", self.state_weight)
        _validate_non_negative_weight("terminal_weight", self.terminal_weight)
        _validate_non_negative_weight("balance_weight", self.balance_weight)

    @property
    def total_time(self) -> float:
        """Total trajectory duration in seconds."""
        return self.n_timesteps * self.dt


def _validate_non_negative_weight(name: str, value: float) -> None:
    """Validate that a weight is non-negative."""
    if value < 0.0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


@dataclass
class TrajectoryResult:
    """Result of trajectory optimization.

    Attributes:
        joint_positions: Joint angles at each timestep,
            shape (n_timesteps, n_joints).
        joint_velocities: Joint angular velocities,
            shape (n_timesteps, n_joints).
        joint_torques: Applied joint torques,
            shape (n_timesteps, n_actuators).
        time: Time values for each step, shape (n_timesteps,).
        cost: Final optimization cost value.
        converged: Whether the optimizer converged.
        iterations: Number of iterations performed.
    """

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    time: np.ndarray
    cost: float
    converged: bool
    iterations: int

    def __post_init__(self) -> None:
        """Validate result arrays have consistent shapes."""
        n_t = len(self.time)
        if self.joint_positions.shape[0] != n_t:
            msg = (
                f"joint_positions has {self.joint_positions.shape[0]}"
                f" timesteps, expected {n_t}"
            )
            raise ValueError(msg)
        if self.joint_velocities.shape[0] != n_t:
            msg = (
                f"joint_velocities has "
                f"{self.joint_velocities.shape[0]}"
                f" timesteps, expected {n_t}"
            )
            raise ValueError(msg)
        if self.joint_torques.shape[0] != n_t:
            msg = (
                f"joint_torques has {self.joint_torques.shape[0]}"
                f" timesteps, expected {n_t}"
            )
            raise ValueError(msg)


def interpolate_phases(
    objective: ExerciseObjective,
    n_frames: int = 50,
) -> np.ndarray:
    """Interpolate between phases to generate keyframes.

    Linearly interpolates joint angles between successive phases
    of an exercise objective, producing a dense keyframe array.

    Args:
        objective: The exercise objective whose phases will be
            interpolated.
        n_frames: Number of output frames (rows).

    Returns:
        Array of shape ``(n_frames, n_joints)`` with linearly
        interpolated joint angles.  Joint ordering is alphabetical.
    """
    all_joints: set[str] = set()
    for phase in objective.phases:
        all_joints.update(phase.target_joints.keys())
    joint_names = sorted(all_joints)

    fractions = np.linspace(0.0, 1.0, n_frames)
    keyframes = np.zeros((n_frames, len(joint_names)))

    for i, f in enumerate(fractions):
        prev_p = objective.phases[0]
        next_p = objective.phases[-1]
        for j in range(len(objective.phases) - 1):
            if objective.phases[j].fraction <= f <= objective.phases[j + 1].fraction:
                prev_p = objective.phases[j]
                next_p = objective.phases[j + 1]
                break

        denom = next_p.fraction - prev_p.fraction
        alpha = 0.0 if denom == 0 else (f - prev_p.fraction) / denom

        for k, jn in enumerate(joint_names):
            v0 = prev_p.target_joints.get(jn, 0.0)
            v1 = next_p.target_joints.get(jn, 0.0)
            keyframes[i, k] = v0 + alpha * (v1 - v0)

    return keyframes


def compute_balance_cost(
    com_position: np.ndarray,
    base_of_support: np.ndarray,
) -> float:
    """Penalize CoM projection outside the base of support polygon.

    Projects the center of mass onto the ground plane (x-y) and
    computes the squared distance to the nearest point inside the
    base of support polygon.

    Args:
        com_position: Center of mass position, shape (3,) as
            [x, y, z].
        base_of_support: Convex polygon vertices defining the base
            of support on the ground plane, shape (n_vertices, 2).

    Returns:
        Squared distance from CoM projection to the nearest polygon
        edge if outside, or 0.0 if inside.

    Raises:
        ValueError: If inputs have invalid shapes or non-finite
            values.
    """
    _validate_array_finite(com_position, "com_position")
    _validate_array_finite(base_of_support, "base_of_support")
    if com_position.shape != (3,):
        msg = f"com_position must have shape (3,), got {com_position.shape}"
        raise ValueError(msg)
    if base_of_support.ndim != 2 or base_of_support.shape[1] != 2:
        msg = f"base_of_support must have shape (n, 2), got {base_of_support.shape}"
        raise ValueError(msg)
    if base_of_support.shape[0] < 3:
        msg = (
            f"base_of_support needs at least 3 vertices, got {base_of_support.shape[0]}"
        )
        raise ValueError(msg)

    com_xy = com_position[:2]

    if _point_in_polygon(com_xy, base_of_support):
        return 0.0

    return _squared_distance_to_polygon(com_xy, base_of_support)


def compute_bar_path_cost(
    bar_position: np.ndarray,
    target_path: np.ndarray,
) -> float:
    """Penalize deviation from optimal bar path.

    Computes the mean squared horizontal deviation between the
    actual bar trajectory and the target (typically vertical over
    mid-foot).

    Args:
        bar_position: Actual bar positions over time,
            shape (n_timesteps, 3).
        target_path: Target bar positions over time,
            shape (n_timesteps, 3).

    Returns:
        Mean squared deviation in the horizontal plane (x-y).

    Raises:
        ValueError: If inputs have mismatched shapes or non-finite
            values.
    """
    _validate_array_finite(bar_position, "bar_position")
    _validate_array_finite(target_path, "target_path")
    if bar_position.shape != target_path.shape:
        msg = (
            f"Shape mismatch: bar_position {bar_position.shape} "
            f"vs target_path {target_path.shape}"
        )
        raise ValueError(msg)
    if bar_position.ndim != 2 or bar_position.shape[1] != 3:
        msg = f"bar_position must have shape (n, 3), got {bar_position.shape}"
        raise ValueError(msg)

    horizontal_diff = bar_position[:, :2] - target_path[:, :2]
    return float(np.mean(np.sum(horizontal_diff**2, axis=1)))


def _validate_array_finite(arr: np.ndarray, name: str) -> None:
    """Check that an array contains only finite values."""
    if not np.all(np.isfinite(arr)):
        msg = f"{name} contains non-finite values"
        raise ValueError(msg)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting test for point-in-polygon on the 2D plane.

    Args:
        point: 2D point, shape (2,).
        polygon: Convex polygon vertices, shape (n, 2).

    Returns:
        True if the point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    px, py = point[0], point[1]
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        j = i
    return inside


def _squared_distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
    """Minimum squared distance from a point to a polygon boundary.

    Args:
        point: 2D point, shape (2,).
        polygon: Polygon vertices, shape (n, 2).

    Returns:
        Minimum squared distance to any polygon edge.
    """
    min_dist_sq = float("inf")
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        dist_sq = _point_to_segment_sq(point, polygon[i], polygon[j])
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    return min_dist_sq


def _point_to_segment_sq(
    point: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> float:
    """Squared distance from a point to a line segment."""
    ab = seg_b - seg_a
    ap = point - seg_a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return float(np.dot(ap, ap))
    t = float(np.dot(ap, ab)) / ab_sq
    t = max(0.0, min(1.0, t))
    closest = seg_a + t * ab
    diff = point - closest
    return float(np.dot(diff, diff))
