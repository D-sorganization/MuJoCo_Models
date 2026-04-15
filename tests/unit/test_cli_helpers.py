"""Unit tests for the argument-parser decomposition in ``__main__``.

Covers the helpers introduced during A-N Refresh 2026-04-14 (issue #129)
that split ``_build_parser`` into three focused functions.
"""

from __future__ import annotations

import argparse

import pytest

from mujoco_models import __main__ as cli
from mujoco_models.exercises import EXERCISE_REGISTRY


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
