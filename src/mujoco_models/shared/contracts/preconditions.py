# SPDX-License-Identifier: MIT
"""Design-by-Contract precondition checks.

All public functions in this project validate inputs via these guards.
Violations raise ValidationError with descriptive messages -- never silently
accept invalid geometry or physics parameters.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from mujoco_models.exceptions import ValidationError


def require_positive(value: float, name: str) -> None:
    """Require *value* to be strictly positive."""
    # ⚡ Bolt Optimization:
    # Inlined math.isfinite to avoid function call overhead in tight loops.
    try:
        if not math.isfinite(value):
            raise ValidationError(f"{name} contains non-finite values")
    except TypeError as exc:
        raise ValidationError(f"{name} must be a finite scalar") from exc

    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def require_non_negative(value: float, name: str) -> None:
    """Require *value* >= 0."""
    # ⚡ Bolt Optimization:
    # Inlined math.isfinite to avoid function call overhead in tight loops.
    try:
        if not math.isfinite(value):
            raise ValidationError(f"{name} contains non-finite values")
    except TypeError as exc:
        raise ValidationError(f"{name} must be a finite scalar") from exc

    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def require_unit_vector(vec: ArrayLike, name: str, tol: float = 1e-6) -> None:
    """Require *vec* to have unit norm within *tol*."""
    # ⚡ Bolt Optimization: Fast path for lists and tuples without coercing to ndarray
    if isinstance(vec, (tuple, list)):
        if len(vec) != 3:
            raise ValidationError(f"{name} must be a 3-vector, got shape ({len(vec)},)")
        try:
            vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be a 3-vector of numbers") from None
    elif getattr(vec, "shape", None) == (3,):
        arr = cast("np.ndarray", vec)
        vx, vy, vz = float(arr[0]), float(arr[1]), float(arr[2])
    else:
        arr = np.asarray(vec, dtype=float)
        if arr.shape != (3,):
            raise ValidationError(f"{name} must be a 3-vector, got shape {arr.shape}")
        vx, vy, vz = float(arr[0]), float(arr[1]), float(arr[2])

    # OPTIMIZATION: Unrolled scalar math instead of np.linalg.norm
    # to avoid allocation and dispatch overhead.
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if abs(norm - 1.0) > tol:
        raise ValidationError(f"{name} must be unit-length (norm={norm:.6f})")


def require_finite(arr: ArrayLike, name: str) -> None:  # noqa: C901
    """Require all elements of *arr* to be finite (no NaN/Inf)."""
    # ⚡ Bolt Optimization: Fast path for scalars.
    if isinstance(arr, (int, float)):
        if not math.isfinite(arr):
            raise ValidationError(f"{name} contains non-finite values")
        return

    # ⚡ Bolt Optimization: Fast path for basic iterables like lists and tuples
    if isinstance(arr, (list, tuple)):
        try:
            for x in arr:
                if not math.isfinite(x):  # type: ignore
                    raise ValidationError(f"{name} contains non-finite values")
            return
        except TypeError:
            pass  # Fallback if list contains non-scalars

    try:
        # Try to use array's fast methods first if it's already a numpy array
        if not np.isfinite(arr).all():
            raise ValidationError(f"{name} contains non-finite values")
    except TypeError as exc:
        try:
            a = np.asarray(arr, dtype=float)
            if not np.isfinite(a).all():
                raise ValidationError(f"{name} contains non-finite values") from None
        except (TypeError, ValueError):
            raise ValidationError(f"{name} contains non-finite values") from exc


def require_in_range(value: float, low: float, high: float, name: str) -> None:
    """Require *low* <= *value* <= *high*."""
    require_finite(value, name)
    require_finite(low, name)
    require_finite(high, name)
    if not (low <= value <= high):
        raise ValidationError(f"{name} must be in [{low}, {high}], got {value}")


def require_shape(arr: ArrayLike, expected: tuple[int, ...], name: str) -> None:
    """Require *arr* to have the given shape."""
    # ⚡ Bolt Optimization: Fast path to avoid `np.asarray`
    shape = getattr(arr, "shape", None)
    if shape is not None:
        if shape != expected:
            raise ValidationError(f"{name} must have shape {expected}, got {shape}")
        return

    # Fast path for 1D lists/tuples using len() directly
    if isinstance(arr, (list, tuple)) and len(expected) == 1:
        # Check if it's truly a 1D list (i.e. first element is not another iterable)
        if len(arr) > 0 and isinstance(arr[0], (list, tuple, np.ndarray)):
            pass
        elif len(arr) != expected[0]:
            msg = f"{name} must have shape {expected}, got ({len(arr)},)"
            raise ValidationError(msg)
        else:
            return

    a = np.asarray(arr)
    if a.shape != expected:
        raise ValidationError(f"{name} must have shape {expected}, got {a.shape}")
