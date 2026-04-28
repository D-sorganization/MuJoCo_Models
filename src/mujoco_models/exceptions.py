# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Domain-specific exception hierarchy for MuJoCo Models.

All public exceptions inherit from :class:`MuJoCoModelError` so callers can
catch every model-related failure with a single ``except`` clause.
"""

from __future__ import annotations


class MuJoCoModelError(Exception):
    """Base exception for all MuJoCo model domain errors."""


class ModelBuildError(MuJoCoModelError):
    """Raised when a model cannot be constructed from its specification."""


class ValidationError(MuJoCoModelError, ValueError):
    """Raised when input data fails domain validation."""


class PreconditionError(MuJoCoModelError):
    """Raised when a runtime precondition is violated."""
