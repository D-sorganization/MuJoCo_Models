# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Tests for the ``model_pack`` discovery surface (issue #266).

These verify the contract UpstreamDrift (#5179) consumes:

* ``resolve()`` returns an existing models-root directory.
* ``manifest()`` matches the YAML on disk.
* ``list_exercises()`` returns the declared IDs.
* The ``mujoco-models`` CLI exports a valid MJCF for each declared exercise
  and the result parses with ``mujoco.MjModel.from_xml_path``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mujoco_models import model_pack as model_pack_mod
from mujoco_models.__main__ import main as cli_main
from mujoco_models.model_pack import list_exercises, manifest, resolve

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "model_pack.yaml"

EXPECTED_EXERCISES = [
    "squat",
    "deadlift",
    "bench_press",
    "snatch",
    "clean_and_jerk",
    "gait",
    "sit_to_stand",
]


@pytest.fixture(autouse=True)
def _reset_manifest_cache() -> None:
    """Drop the ``functools.cache`` so per-test mutation cannot leak."""
    model_pack_mod.manifest.cache_clear()


@pytest.mark.unit
class TestManifest:
    def test_manifest_matches_yaml_on_disk(self) -> None:
        on_disk = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
        assert manifest() == on_disk

    def test_manifest_declares_seven_exercises(self) -> None:
        ids = [e["id"] for e in manifest()["exercises"]]
        assert ids == EXPECTED_EXERCISES

    def test_manifest_has_required_top_level_fields(self) -> None:
        m = manifest()
        assert m["schema"] == "model_pack/v1"
        assert m["package"] == "mujoco_models"
        assert m["engine"] == "mujoco"
        assert m["engine_version"] == ">=3.0,<4"
        assert m["format"] == "mjcf"
        assert m["anthropometrics"] == "winter_2009"
        assert m["models_root"] == "src/mujoco_models/exercises"

    def test_list_exercises(self) -> None:
        assert list_exercises() == EXPECTED_EXERCISES


@pytest.mark.unit
class TestResolve:
    def test_resolve_returns_existing_directory(self) -> None:
        root = resolve()
        assert root.is_dir()

    def test_resolve_contains_seven_exercise_subdirs(self) -> None:
        root = resolve()
        for exercise_id in EXPECTED_EXERCISES:
            assert (root / exercise_id).is_dir(), (
                f"missing exercise subdir: {exercise_id}"
            )


@pytest.mark.unit
def test_cli_list_exercises_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(["--list-exercises"])
    captured = capsys.readouterr()
    assert exit_code == 0
    listed = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert listed == EXPECTED_EXERCISES


@pytest.mark.unit
def test_cli_export_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "squat.xml"
    exit_code = cli_main(["--exercise", "squat", "--export", str(out)])
    assert exit_code == 0
    assert out.is_file()
    assert "<mujoco" in out.read_text(encoding="utf-8")


@pytest.mark.unit
def test_cli_console_script_smoke(tmp_path: Path) -> None:
    """The ``mujoco-models`` console script must resolve to our ``main``."""
    script = shutil.which("mujoco-models")
    if script is None:
        # Not installed in this environment — fall back to module form, which
        # exercises the same dispatch path.
        result = subprocess.run(
            [sys.executable, "-m", "mujoco_models", "--list-exercises"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        result = subprocess.run(
            [script, "--list-exercises"],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    listed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert set(listed) >= set(EXPECTED_EXERCISES)


try:
    import mujoco

    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False


@pytest.mark.live_simulation
@pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="mujoco not installed")
@pytest.mark.parametrize("exercise", EXPECTED_EXERCISES)
def test_cli_export_parses_with_mujoco(
    exercise: str,
    tmp_path: Path,
) -> None:
    """Each declared exercise must export an MJCF that MuJoCo can parse."""
    out = tmp_path / f"{exercise}.xml"
    exit_code = cli_main(["--exercise", exercise, "--export", str(out)])
    assert exit_code == 0
    model = mujoco.MjModel.from_xml_path(str(out))
    assert model.nbody > 1
