"""Tests for precondition contract checks."""

import numpy as np
import pytest

from mujoco_models.shared.contracts.preconditions import (
    require_finite,
    require_in_range,
    require_non_negative,
    require_positive,
    require_shape,
    require_unit_vector,
)


class TestRequirePositive:
    def test_accepts_positive(self) -> None:
        require_positive(1.0, "x")

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            require_positive(0.0, "x")

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            require_positive(-1.0, "x")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            require_positive(value, "x")

    def test_includes_name_in_message(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            require_positive(-5.0, "mass")


class TestRequireNonNegative:
    def test_accepts_zero(self) -> None:
        require_non_negative(0.0, "x")

    def test_accepts_positive(self) -> None:
        require_non_negative(1.0, "x")

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            require_non_negative(-0.001, "x")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            require_non_negative(value, "x")


class TestRequireUnitVector:
    def test_accepts_unit_x(self) -> None:
        require_unit_vector([1, 0, 0], "axis")

    def test_accepts_unit_z(self) -> None:
        require_unit_vector([0, 0, 1], "axis")

    def test_accepts_normalized(self) -> None:
        v = np.array([1, 1, 1]) / np.sqrt(3)
        require_unit_vector(v, "axis")

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="3-vector"):
            require_unit_vector([1, 0], "axis")

    def test_rejects_non_unit(self) -> None:
        with pytest.raises(ValueError, match="unit-length"):
            require_unit_vector([2, 0, 0], "axis")

    def test_rejects_zero_vector(self) -> None:
        with pytest.raises(ValueError, match="unit-length"):
            require_unit_vector([0, 0, 0], "axis")


class TestRequireFinite:
    def test_accepts_finite(self) -> None:
        require_finite([1.0, 2.0, 3.0], "v")

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            require_finite([1.0, float("nan"), 3.0], "v")

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            require_finite([float("inf"), 0, 0], "v")


class TestRequireInRange:
    def test_accepts_in_range(self) -> None:
        require_in_range(5.0, 0.0, 10.0, "x")

    def test_accepts_at_low_bound(self) -> None:
        require_in_range(0.0, 0.0, 10.0, "x")

    def test_accepts_at_high_bound(self) -> None:
        require_in_range(10.0, 0.0, 10.0, "x")

    def test_rejects_below(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            require_in_range(-1.0, 0.0, 10.0, "x")

    def test_rejects_above(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            require_in_range(11.0, 0.0, 10.0, "x")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_value(self, value: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            require_in_range(value, 0.0, 10.0, "x")


class TestRequireShape:
    def test_accepts_matching_shape(self) -> None:
        require_shape([1, 2, 3], (3,), "v")

    def test_accepts_2d(self) -> None:
        require_shape(np.eye(3), (3, 3), "m")

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            require_shape([1, 2], (3,), "v")
