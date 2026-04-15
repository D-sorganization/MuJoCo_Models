"""Unit tests for the ``examples/generate_all_models.py`` helpers.

Loaded via ``importlib`` because ``examples/`` is not a package in the
source tree.  Covers the decomposition done during A-N Refresh
2026-04-14 (issue #129).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from mujoco_models.exercises.base import ExerciseConfig


def _load_module() -> object:
    """Load ``examples/generate_all_models.py`` as a module object."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples" / "generate_all_models.py"
    spec = importlib.util.spec_from_file_location("generate_all_models", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_parser_defaults() -> None:
    """Default parse yields ``output`` dir and 80 kg / 1.75 m anthropometry."""
    mod = _load_module()
    parser = mod._build_parser()  # type: ignore[attr-defined]
    ns = parser.parse_args([])
    assert ns.output_dir == "output"
    assert ns.body_mass == pytest.approx(80.0)
    assert ns.height == pytest.approx(1.75)


def test_build_config_produces_exercise_config() -> None:
    """``_build_config`` wraps user anthropometry in an ExerciseConfig."""
    mod = _load_module()
    cfg = mod._build_config(72.0, 1.70)  # type: ignore[attr-defined]
    assert isinstance(cfg, ExerciseConfig)
    assert cfg.body_spec.total_mass == pytest.approx(72.0)
    assert cfg.body_spec.height == pytest.approx(1.70)


def test_build_config_enforces_positive_mass() -> None:
    """Preconditions are inherited from BodyModelSpec: zero mass is rejected."""
    mod = _load_module()
    with pytest.raises(ValueError):
        mod._build_config(0.0, 1.75)  # type: ignore[attr-defined]


def test_iter_unique_builders_deduplicates_by_class() -> None:
    """Each unique builder class appears exactly once."""
    mod = _load_module()
    unique = mod._iter_unique_builders()  # type: ignore[attr-defined]
    classes = [cls.__name__ for _, cls in unique]
    assert len(classes) == len(set(classes))


def test_write_model_writes_xml_file(tmp_path) -> None:
    """``_write_model`` produces an MJCF XML file at the expected path."""
    mod = _load_module()
    unique = mod._iter_unique_builders()  # type: ignore[attr-defined]
    name, builder_cls = unique[0]
    cfg = ExerciseConfig()
    out_path = mod._write_model(  # type: ignore[attr-defined]
        name, builder_cls, cfg, str(tmp_path)
    )
    assert os.path.exists(out_path)
    with open(out_path, encoding="utf-8") as fh:
        head = fh.read(200)
    assert "<mujoco" in head
