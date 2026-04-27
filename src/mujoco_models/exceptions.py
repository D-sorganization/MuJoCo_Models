# SPDX-License-Identifier: MIT
"""Domain-specific exceptions for MuJoCo model operations.

All errors raised by this package should use (or inherit from)
:class:`MuJoCoModelError` so that consumers can catch domain-specific
problems separately from generic Python built-in exceptions.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization



class MuJoCoModelError(Exception):
    """Base exception for all MuJoCo model domain errors."""

    pass


class ModelBuildError(MuJoCoModelError):
    """Raised when a model cannot be constructed from its specification."""

    pass


class ValidationError(MuJoCoModelError, ValueError):
    """Raised when input data fails domain validation."""

    pass


class PreconditionError(MuJoCoModelError):
    """Raised when a runtime precondition is violated."""

    pass
