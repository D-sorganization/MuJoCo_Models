# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Model-pack manifest discovery for the UpstreamDrift launcher integration.

This module exposes the stable entry-point surface that UpstreamDrift uses
to find this repo's exercise content. See issue
https://github.com/D-sorganization/MuJoCo_Models/issues/266 and umbrella
https://github.com/D-sorganization/UpstreamDrift/issues/5179.

The manifest is loaded from ``model_pack.yaml`` which is shipped both at
the repo root (for source checkouts) and inside the installed package
(for ``pip install`` consumers) via ``importlib.resources``.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import yaml

_MANIFEST_FILENAME = "model_pack.yaml"


def _load_manifest_text() -> str:
    """Return the raw contents of ``model_pack.yaml``.

    Looks first inside the installed package (the wheel ships the file via
    ``force-include``). Falls back to the source-tree repo root so that
    editable installs / source checkouts continue to work even when the
    package-data copy is absent.
    """
    package_resource = files("mujoco_models").joinpath(_MANIFEST_FILENAME)
    if package_resource.is_file():
        return package_resource.read_text(encoding="utf-8")

    repo_root_candidate = Path(__file__).resolve().parents[2] / _MANIFEST_FILENAME
    if repo_root_candidate.is_file():
        return repo_root_candidate.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"{_MANIFEST_FILENAME} not found in package data or repo root",
    )


@cache
def manifest() -> dict[str, Any]:
    """Load and return the ``model_pack.yaml`` contents as a dict.

    The result is cached because the manifest is immutable for the lifetime
    of the process. Returns a fresh shallow-copyable dict on each access by
    way of ``yaml.safe_load``.
    """
    parsed = yaml.safe_load(_load_manifest_text())
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{_MANIFEST_FILENAME} must parse to a mapping, "
            f"got {type(parsed).__name__}",
        )
    return cast("dict[str, Any]", parsed)


def resolve() -> Path:
    """Return the absolute path to this package's models root directory.

    Postcondition: returned path exists and is a directory.
    """
    root = Path(str(files("mujoco_models").joinpath("exercises"))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"models root not found: {root}")
    return root


def list_exercises() -> list[str]:
    """Return the list of exercise IDs declared by the manifest."""
    exercises = manifest().get("exercises", [])
    return [str(entry["id"]) for entry in exercises]
