"""Tests for the CLI entry point (__main__.py)."""

from __future__ import annotations

from typing import Any

from mujoco_models.__main__ import _build_parser, main


class TestCLI:
    def test_parser_accepts_squat(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["squat"])
        assert args.exercise == "squat"

    def test_parser_default_body_mass(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["deadlift"])
        assert args.body_mass == 80.0

    def test_parser_default_height(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["bench_press"])
        assert args.height == 1.75

    def test_parser_custom_params(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["snatch", "--body-mass", "90", "--height", "1.85", "--plate-mass", "40"]
        )
        assert args.body_mass == 90.0
        assert args.height == 1.85
        assert args.plate_mass == 40.0

    def test_main_returns_zero(self) -> None:
        result = main(["squat"])
        assert result == 0

    def test_main_with_output(self, tmp_path: Any) -> None:
        out_file = str(tmp_path / "test_model.xml")
        result = main(["deadlift", "-o", out_file])
        assert result == 0
        with open(out_file) as fh:
            content = fh.read()
        assert "<mujoco" in content

    def test_main_verbose(self) -> None:
        result = main(["bench_press", "-v"])
        assert result == 0
