# SPDX-License-Identifier: MIT
"""Domain-specific exception hierarchy for MuJoCo Models.

All public exceptions inherit from :class:`MuJoCoModelError` so callers can
catch every model-related failure with a single ``except`` clause.
"""

from __future__ import annotations


class MuJoCoModelError(Exception):
    """Base exception for all model-related errors."""


class ModelBuildError(MuJoCoModelError):
    """Raised when an MJCF model cannot be constructed."""


class ValidationError(MuJoCoModelError):
    """Raised when input parameters fail domain validation."""


class PreconditionError(MuJoCoModelError):
    """Raised when a runtime precondition is violated."""
