# SPDX-License-Identifier: MIT
"""Unit tests for the custom exception hierarchy."""

import pytest

from mujoco_models.exceptions import (
    ModelBuildError,
    MuJoCoModelError,
    PreconditionError,
    ValidationError,
)


def test_exception_inheritance() -> None:
    """All domain exceptions inherit from MuJoCoModelError."""
    assert issubclass(ValidationError, MuJoCoModelError)
    assert issubclass(ModelBuildError, MuJoCoModelError)
    assert issubclass(PreconditionError, MuJoCoModelError)
    assert issubclass(MuJoCoModelError, Exception)


def test_validation_error_is_value_error() -> None:
    """ValidationError inherits from ValueError for backward compatibility."""
    assert issubclass(ValidationError, ValueError)


def test_can_catch_with_base_class() -> None:
    """Catching MuJoCoModelError catches all domain exceptions."""
    with pytest.raises(MuJoCoModelError):
        raise ValidationError("test")

    with pytest.raises(MuJoCoModelError):
        raise ModelBuildError("test")

    with pytest.raises(MuJoCoModelError):
        raise PreconditionError("test")


def test_can_catch_with_value_error() -> None:
    """Catching ValueError catches ValidationError for backward compatibility."""
    with pytest.raises(ValueError):
        raise ValidationError("test")
