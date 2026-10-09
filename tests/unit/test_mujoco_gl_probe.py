# SPDX-License-Identifier: MIT
"""Tests for the MuJoCo GL backend probe in the root conftest.py (issue #399)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import conftest as root_conftest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fake_import(monkeypatch: pytest.MonkeyPatch, working: set[str | None]) -> None:
    """Make _import_mujoco succeed only for MUJOCO_GL values in ``working``."""

    def fake() -> None:
        if os.environ.get("MUJOCO_GL") not in working:
            raise AttributeError("'NoneType' object has no attribute 'eglQueryString'")

    monkeypatch.setattr(root_conftest, "_import_mujoco", fake)


@pytest.mark.unit
def test_probe_noop_when_mujoco_gl_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    _fake_import(monkeypatch, set())  # would raise if the probe imported
    assert root_conftest.probe_mujoco_gl() is None


@pytest.mark.unit
def test_probe_keeps_working_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    _fake_import(monkeypatch, {"egl"})
    assert root_conftest.probe_mujoco_gl() is None
    assert os.environ["MUJOCO_GL"] == "egl"


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc",
    [
        AttributeError("'NoneType' object has no attribute 'eglQueryString'"),
        ImportError("Unable to load EGL library"),
        OSError("libEGL.so.1: cannot open shared object file"),
        RuntimeError("invalid value for environment variable MUJOCO_GL: egl"),
    ],
)
def test_probe_falls_back_to_osmesa_on_gl_errors(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")

    def fake() -> None:
        if os.environ.get("MUJOCO_GL") == "egl":
            raise exc

    monkeypatch.setattr(root_conftest, "_import_mujoco", fake)
    reason = root_conftest.probe_mujoco_gl()
    assert reason is not None
    assert "'egl'" in reason
    assert type(exc).__name__ in reason
    assert "fell back to MUJOCO_GL=osmesa" in reason
    assert os.environ["MUJOCO_GL"] == "osmesa"


@pytest.mark.unit
def test_probe_unsets_when_osmesa_also_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    _fake_import(monkeypatch, {None})
    reason = root_conftest.probe_mujoco_gl()
    assert reason is not None
    assert "<unset>" in reason
    assert "MUJOCO_GL" not in os.environ


@pytest.mark.unit
def test_probe_restores_env_when_mujoco_unimportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    _fake_import(monkeypatch, set())
    reason = root_conftest.probe_mujoco_gl()
    assert reason is not None
    assert "mujoco unavailable" in reason
    assert os.environ["MUJOCO_GL"] == "egl"


@pytest.mark.unit
def test_collection_hook_skips_only_rendering_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(root_conftest, "GL_PROBE_REASON", "EGL missing")
    render, physics = MagicMock(), MagicMock()
    render.keywords = {"rendering": True}
    physics.keywords = {"unit": True}
    root_conftest.pytest_collection_modifyitems(MagicMock(), [render, physics])
    marker = render.add_marker.call_args.args[0]
    assert marker.name == "skip"
    assert "EGL missing" in marker.kwargs["reason"]
    physics.add_marker.assert_not_called()


@pytest.mark.unit
def test_collection_hook_noop_when_gl_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(root_conftest, "GL_PROBE_REASON", None)
    render = MagicMock()
    render.keywords = {"rendering": True}
    root_conftest.pytest_collection_modifyitems(MagicMock(), [render])
    render.add_marker.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "test_file", ["tests/test_model_pack.py", "tests/parity/test_engine_physics.py"]
)
def test_collection_survives_unusable_mujoco_gl(test_file: str) -> None:
    """A forced-bad MUJOCO_GL must not abort collection of physics-only modules."""
    env = {
        **os.environ,
        "MUJOCO_GL": "bogus_headless_backend",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "--collect-only", "-q"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "error" not in proc.stdout.lower().split("collected")[-1]
