"""Root pytest configuration for MuJoCo Models test suite.

This file is intentionally minimal. Shared fixtures and hooks live here
so pytest can discover them from any test subdirectory.

Fleet Testing Standards §5: thread-safety and headless env vars must be
set before any heavy import (numpy/MKL, matplotlib, Qt, MuJoCo's GLFW).
See: docs/FLEET_TESTING_STANDARDS.md in Repository_Management.
"""
from __future__ import annotations

import os

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

