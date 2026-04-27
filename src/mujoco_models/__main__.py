# SPDX-License-Identifier: MIT
"""CLI entry point for ``python -m mujoco_models <exercise> [options]``."""

from __future__ import annotations

import argparse
import logging
import sys

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.exercises.base import ExerciseConfig
from mujoco_models.shared.barbell import BarbellSpec
from mujoco_models.shared.body import BodyModelSpec

logger = logging.getLogger(__name__)


def _add_exercise_argument(parser: argparse.ArgumentParser) -> None:
    """Add the positional ``exercise`` argument."""
    parser.add_argument(
        "exercise",
        choices=sorted(set(EXERCISE_REGISTRY.keys())),
        help="Exercise to generate a model for.",
    )


def _add_anthropometry_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--body-mass``, ``--height`` and ``--plate-mass`` options."""
    parser.add_argument(
        "--body-mass",
        type=float,
        default=80.0,
        help="Body mass in kg (default: 80.0).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1.75,
        help="Body height in meters (default: 1.75).",
    )
    parser.add_argument(
        "--plate-mass",
        type=float,
        default=0.0,
        help="Plate mass per side in kg (default: 0.0).",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add output-file and verbosity options."""
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m mujoco_models",
        description="Generate MuJoCo MJCF models for barbell exercises.",
    )
    _add_exercise_argument(parser)
    _add_anthropometry_arguments(parser)
    _add_output_arguments(parser)
    return parser


def _configure_logging(verbose: bool) -> None:
    """Initialise the root logger for the CLI run.

    Precondition: called exactly once per CLI invocation.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s",
    )


def _build_config_from_args(args: argparse.Namespace) -> ExerciseConfig | None:
    """Build an :class:`ExerciseConfig` from parsed CLI args.

    Returns ``None`` and logs an error on invalid anthropometry, so the
    caller can exit with a non-zero status without catching exceptions.
    """
    try:
        return ExerciseConfig(
            body_spec=BodyModelSpec(total_mass=args.body_mass, height=args.height),
            barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=args.plate_mass),
        )
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        return None


def _build_model_xml(exercise: str, config: ExerciseConfig) -> str | None:
    """Instantiate the registered builder for *exercise* and call ``build()``.

    Returns the MJCF XML string, or ``None`` when the builder raises a
    known error (``ValueError`` or ``RuntimeError``).
    """
    builder_cls = EXERCISE_REGISTRY[exercise]
    try:
        return builder_cls(config).build()
    except (ValueError, RuntimeError) as exc:
        logger.error("Model build failed for '%s': %s", exercise, exc)
        return None


def _emit_xml(xml_str: str, output: str | None) -> int:
    """Write *xml_str* to *output* (or stdout) and return an exit code."""
    if output is None:
        sys.stdout.write(xml_str)
        return 0
    try:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(xml_str)
    except OSError as exc:
        logger.error("Failed to write output file '%s': %s", output, exc)
        return 1
    logger.info("Wrote %s", output)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI main function. Returns 0 on success, 1 on error."""
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    config = _build_config_from_args(args)
    if config is None:
        return 1

    xml_str = _build_model_xml(args.exercise, config)
    if xml_str is None:
        return 1

    return _emit_xml(xml_str, args.output)


if __name__ == "__main__":
    sys.exit(main())
