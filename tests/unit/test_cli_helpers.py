"""Unit tests for the argument-parser decomposition in ``__main__``.

Covers the helpers introduced during A-N Refresh 2026-04-14 (issue #129)
that split ``_build_parser`` into three focused functions, plus the
``main()`` decomposition helpers added in issue #122 (A-N 2026-04-10).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pytest

from mujoco_models import __main__ as cli
from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.exercises.base import ExerciseConfig


def test_add_exercise_argument_restricts_to_known_exercises() -> None:
    """Invalid exercise names are rejected by argparse."""
    p = argparse.ArgumentParser()
    cli._add_exercise_argument(p)
    # Valid exercise succeeds
    known = sorted(EXERCISE_REGISTRY.keys())[0]
    ns = p.parse_args([known])
    assert ns.exercise == known
    # Unknown exercise raises SystemExit (argparse error)
    with pytest.raises(SystemExit):
        p.parse_args(["not_a_real_exercise"])


def test_add_anthropometry_arguments_supplies_defaults() -> None:
    """Anthropometry args default to the documented 80 kg / 1.75 m / 0 kg."""
    p = argparse.ArgumentParser()
    cli._add_anthropometry_arguments(p)
    ns = p.parse_args([])
    assert ns.body_mass == pytest.approx(80.0)
    assert ns.height == pytest.approx(1.75)
    assert ns.plate_mass == pytest.approx(0.0)


def test_add_output_arguments_toggles_verbose_flag() -> None:
    """``-v`` flips ``verbose`` to True; output path optional."""
    p = argparse.ArgumentParser()
    cli._add_output_arguments(p)
    ns = p.parse_args(["-v"])
    assert ns.verbose is True
    assert ns.output is None
    ns2 = p.parse_args(["-o", "/tmp/out.xml"])
    assert ns2.verbose is False
    assert ns2.output == "/tmp/out.xml"


def test_build_parser_composes_all_argument_groups() -> None:
    """End-to-end parse exercises all three helper groups."""
    p = cli._build_parser()
    known = sorted(EXERCISE_REGISTRY.keys())[0]
    ns = p.parse_args(
        [known, "--body-mass", "70", "--height", "1.8", "--plate-mass", "20", "-v"]
    )
    assert ns.exercise == known
    assert ns.body_mass == pytest.approx(70.0)
    assert ns.height == pytest.approx(1.8)
    assert ns.plate_mass == pytest.approx(20.0)
    assert ns.verbose is True


# ---------------------------------------------------------------------------
# Helpers extracted from ``main`` for issue #122 (A-N 2026-04-10 refresh)
# ---------------------------------------------------------------------------


def _default_args(**overrides: Any) -> argparse.Namespace:
    """Build a parsed ``Namespace`` matching CLI defaults, with overrides."""
    defaults = {
        "exercise": sorted(EXERCISE_REGISTRY.keys())[0],
        "body_mass": 80.0,
        "height": 1.75,
        "plate_mass": 0.0,
        "output": None,
        "verbose": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_configure_logging_forwards_verbose_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verbose flag forwards ``DEBUG`` to ``logging.basicConfig``."""
    captured: dict[str, Any] = {}

    def _fake_basic_config(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", _fake_basic_config)
    cli._configure_logging(verbose=True)
    assert captured.get("level") == logging.DEBUG


def test_configure_logging_defaults_to_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-verbose invocation forwards ``WARNING`` to ``logging.basicConfig``."""
    captured: dict[str, Any] = {}

    def _fake_basic_config(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", _fake_basic_config)
    cli._configure_logging(verbose=False)
    assert captured.get("level") == logging.WARNING
    assert "format" in captured


def test_build_config_from_args_returns_valid_config() -> None:
    """Valid anthropometry yields an ``ExerciseConfig`` instance."""
    cfg = cli._build_config_from_args(_default_args())
    assert isinstance(cfg, ExerciseConfig)
    assert cfg.body_spec.total_mass == pytest.approx(80.0)


def test_build_config_from_args_returns_none_on_invalid_mass(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative body mass is rejected and logged, returning ``None``."""
    with caplog.at_level(logging.ERROR, logger=cli.logger.name):
        result = cli._build_config_from_args(_default_args(body_mass=-1.0))
    assert result is None
    assert any("Invalid configuration" in rec.message for rec in caplog.records)


def test_build_model_xml_returns_xml_for_known_exercise() -> None:
    """A successful build produces a non-empty ``<mujoco>`` XML string."""
    cfg = cli._build_config_from_args(_default_args())
    assert cfg is not None
    xml_str = cli._build_model_xml(sorted(EXERCISE_REGISTRY.keys())[0], cfg)
    assert xml_str is not None
    assert xml_str.lstrip().startswith("<mujoco") or "<mujoco" in xml_str


def test_build_model_xml_returns_none_on_builder_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Builder-raised ``ValueError``/``RuntimeError`` is swallowed -> ``None``."""

    class _BoomBuilder:
        def __init__(self, _cfg: Any) -> None: ...

        def build(self) -> str:
            raise RuntimeError("boom")

    target = sorted(EXERCISE_REGISTRY.keys())[0]
    monkeypatch.setitem(EXERCISE_REGISTRY, target, _BoomBuilder)
    cfg = ExerciseConfig()
    with caplog.at_level(logging.ERROR, logger=cli.logger.name):
        result = cli._build_model_xml(target, cfg)
    assert result is None
    assert any("Model build failed" in rec.message for rec in caplog.records)


def test_emit_xml_writes_to_stdout_when_output_is_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No output path means the XML is written to stdout and exit code is 0."""
    code = cli._emit_xml("<mujoco/>", output=None)
    assert code == 0
    assert capsys.readouterr().out == "<mujoco/>"


def test_emit_xml_writes_to_file(tmp_path: Path) -> None:
    """A provided output path writes the XML to that file."""
    out = tmp_path / "model.xml"
    code = cli._emit_xml("<mujoco/>", output=str(out))
    assert code == 0
    assert out.read_text(encoding="utf-8") == "<mujoco/>"


def test_emit_xml_returns_error_on_unwritable_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OSError on write yields exit code 1 and is logged."""
    bogus = "/this/path/does/not/exist/model.xml"
    with caplog.at_level(logging.ERROR, logger=cli.logger.name):
        code = cli._emit_xml("<mujoco/>", output=bogus)
    assert code == 1
    assert any("Failed to write output file" in rec.message for rec in caplog.records)
