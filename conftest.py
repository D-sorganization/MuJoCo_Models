"""Root pytest configuration for MuJoCo Models test suite.

This file is intentionally minimal. Shared fixtures and hooks live here
so pytest can discover them from any test subdirectory.

Fleet Testing Standards §5: thread-safety and headless env vars must be
set before any heavy import (numpy/MKL, matplotlib, Qt, MuJoCo's GLFW).
See: docs/FLEET_TESTING_STANDARDS.md in Repository_Management.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

logger = logging.getLogger(__name__)

# C-extension thread safety. Many "xdist worker crashed" failures come
# from MKL/OpenBLAS forking under xdist. Pin to single-threaded for tests;
# production code can re-thread itself if it needs to.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# matplotlib headless backend, set before any matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")

# Qt headless backend. Matters for MuJoCo's GLFW viewer paths and any
# indirect PyQt/PySide imports during test collection.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --- MuJoCo GL backend probe (issue #399) ---------------------------------
# Importing ``mujoco`` under ``MUJOCO_GL=egl`` loads PyOpenGL's EGL platform
# eagerly; without EGL that raises AttributeError (or RuntimeError / OSError /
# ImportError for other broken backends) and aborts collection. Probe the
# import here, fall back to osmesa and then to an unset MUJOCO_GL so physics
# tests still collect, and skip tests marked ``rendering`` with a clear reason.
_GL_IMPORT_ERRORS = (ImportError, AttributeError, OSError, RuntimeError)


def _import_mujoco() -> None:
    import mujoco  # noqa: F401


def _evict_gl_modules() -> None:
    """Drop partially imported mujoco/OpenGL modules so the import can be retried."""
    for name in list(sys.modules):
        if name == "mujoco" or name.startswith(("mujoco.", "OpenGL")):
            sys.modules.pop(name, None)


def probe_mujoco_gl() -> str | None:
    """Import mujoco under the configured MUJOCO_GL, falling back if it fails.

    Returns None when the configured backend imports cleanly (or none is
    configured), else a reason string describing the failure and fallback.
    Fallback order: osmesa, then MUJOCO_GL unset. If mujoco cannot be imported
    at all, MUJOCO_GL is restored and the reason says so.
    """
    configured = os.environ.get("MUJOCO_GL")
    if configured is None:
        return None
    try:
        _import_mujoco()
        return None
    except _GL_IMPORT_ERRORS as exc:
        reason = f"MUJOCO_GL={configured!r} unusable ({type(exc).__name__}: {exc})"
    logger.warning("%s; falling back", reason)
    for fallback in ("osmesa", None):
        _evict_gl_modules()
        if fallback is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = fallback
        try:
            _import_mujoco()
        except _GL_IMPORT_ERRORS:
            continue
        return f"{reason}; fell back to MUJOCO_GL={fallback or '<unset>'}"
    _evict_gl_modules()
    os.environ["MUJOCO_GL"] = configured
    return f"{reason}; mujoco unavailable under any GL backend"


GL_PROBE_REASON: str | None = probe_mujoco_gl()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "rendering: needs a working MuJoCo GL rendering backend"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``rendering`` tests when the configured GL backend was unusable."""
    if GL_PROBE_REASON is None:
        return
    skip = pytest.mark.skip(reason=f"MuJoCo rendering unavailable: {GL_PROBE_REASON}")
    for item in items:
        if "rendering" in item.keywords:
            item.add_marker(skip)
