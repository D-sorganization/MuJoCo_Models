"""Tests for Rust accelerator -- fallback/interface verification.

Since the Rust extension requires compilation with maturin, these tests
verify the Python-side interface expectations and provide fallback
pure-Python implementations that match the Rust signatures.
"""

from __future__ import annotations

import numpy as np


def _inverse_dynamics_batch_py(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    masses: np.ndarray,
    lengths: np.ndarray,
    gravity: float,
) -> np.ndarray:
    """Pure-Python reference implementation of batch inverse dynamics."""
    n, n_joints = q.shape
    torques = np.zeros((n, n_joints))
    for i in range(n):
        for j in range(n_joints - 1, -1, -1):
            inertia = masses[j] * lengths[j] ** 2 / 3.0
            torques[i, j] = inertia * qdd[i, j] + masses[j] * gravity * lengths[
                j
            ] * 0.5 * np.cos(q[i, j])
            if j + 1 < n_joints:
                torques[i, j] += torques[i, j + 1]
    return torques


def _com_batch_py(
    q: np.ndarray,
    masses: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Pure-Python reference implementation of batch CoM computation."""
    n, n_joints = q.shape
    total_mass = masses.sum()
    com = np.zeros((n, 2))
    for i in range(n):
        cum_x, cum_z = 0.0, 0.0
        x_sum, z_sum = 0.0, 0.0
        for j in range(n_joints):
            angle = q[i, j]
            seg_cx = cum_x + lengths[j] * 0.5 * np.sin(angle)
            seg_cz = cum_z + lengths[j] * 0.5 * np.cos(angle)
            x_sum += masses[j] * seg_cx
            z_sum += masses[j] * seg_cz
            cum_x += lengths[j] * np.sin(angle)
            cum_z += lengths[j] * np.cos(angle)
        com[i, 0] = x_sum / total_mass
        com[i, 1] = z_sum / total_mass
    return com


def _interpolate_phases_py(
    phase_fractions: np.ndarray,
    phase_angles: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Pure-Python reference implementation of phase interpolation."""
    n_joints = phase_angles.shape[1]
    t_out = np.linspace(0, 1, n_frames)
    result = np.zeros((n_frames, n_joints))
    for i, t in enumerate(t_out):
        idx = 0
        for k in range(len(phase_fractions) - 1):
            if phase_fractions[k] <= t <= phase_fractions[k + 1]:
                idx = k
                break
        denom = phase_fractions[idx + 1] - phase_fractions[idx]
        alpha = 0.0 if abs(denom) < 1e-12 else (t - phase_fractions[idx]) / denom
        for j in range(n_joints):
            result[i, j] = phase_angles[idx, j] + alpha * (
                phase_angles[idx + 1, j] - phase_angles[idx, j]
            )
    return result


class TestInverseDynamicsBatch:
    def test_output_shape(self) -> None:
        n, nj = 10, 3
        q = np.zeros((n, nj))
        qd = np.zeros((n, nj))
        qdd = np.ones((n, nj))
        masses = np.array([5.0, 3.0, 2.0])
        lengths = np.array([0.4, 0.3, 0.25])
        result = _inverse_dynamics_batch_py(q, qd, qdd, masses, lengths, 9.81)
        assert result.shape == (n, nj)

    def test_zero_acceleration_gravity_only(self) -> None:
        """With zero acceleration, torques should only reflect gravity."""
        q = np.zeros((1, 2))  # straight down
        qd = np.zeros((1, 2))
        qdd = np.zeros((1, 2))
        masses = np.array([5.0, 3.0])
        lengths = np.array([0.4, 0.3])
        result = _inverse_dynamics_batch_py(q, qd, qdd, masses, lengths, 9.81)
        # All torques should be non-zero due to gravity
        assert np.all(np.abs(result) > 0)

    def test_torque_propagation(self) -> None:
        """Parent joint should include child torque contribution."""
        q = np.zeros((1, 3))
        qd = np.zeros((1, 3))
        qdd = np.zeros((1, 3))
        masses = np.array([5.0, 3.0, 2.0])
        lengths = np.array([0.4, 0.3, 0.25])
        result = _inverse_dynamics_batch_py(q, qd, qdd, masses, lengths, 9.81)
        # Parent torque should be >= child torque (gravity cascades)
        assert abs(result[0, 0]) >= abs(result[0, 1])
        assert abs(result[0, 1]) >= abs(result[0, 2])


class TestComBatch:
    def test_output_shape(self) -> None:
        n, nj = 5, 3
        q = np.zeros((n, nj))
        masses = np.array([5.0, 3.0, 2.0])
        lengths = np.array([0.4, 0.3, 0.25])
        result = _com_batch_py(q, masses, lengths)
        assert result.shape == (n, 2)

    def test_straight_chain_com_on_z_axis(self) -> None:
        """With all angles zero, CoM x-coordinate should be zero."""
        q = np.zeros((1, 3))
        masses = np.array([5.0, 3.0, 2.0])
        lengths = np.array([0.4, 0.3, 0.25])
        result = _com_batch_py(q, masses, lengths)
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-10)
        assert result[0, 1] > 0  # CoM should be above origin


class TestInterpolatePhases:
    def test_output_shape(self) -> None:
        fracs = np.array([0.0, 0.5, 1.0])
        angles = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        result = _interpolate_phases_py(fracs, angles, 20)
        assert result.shape == (20, 2)

    def test_boundary_values(self) -> None:
        fracs = np.array([0.0, 0.5, 1.0])
        angles = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        result = _interpolate_phases_py(fracs, angles, 11)
        np.testing.assert_allclose(result[0], angles[0], atol=1e-10)
        np.testing.assert_allclose(result[-1], angles[-1], atol=1e-10)

    def test_midpoint_interpolation(self) -> None:
        fracs = np.array([0.0, 1.0])
        angles = np.array([[0.0], [2.0]])
        result = _interpolate_phases_py(fracs, angles, 3)
        np.testing.assert_allclose(result[1, 0], 1.0, atol=1e-10)
