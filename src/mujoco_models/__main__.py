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


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m mujoco_models",
        description="Generate MuJoCo MJCF models for barbell exercises.",
    )
    parser.add_argument(
        "exercise",
        choices=sorted(set(EXERCISE_REGISTRY.keys())),
        help="Exercise to generate a model for.",
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main function. Returns 0 on success, 1 on error."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s %(levelname)s: %(message)s",
    )

    try:
        config = ExerciseConfig(
            body_spec=BodyModelSpec(total_mass=args.body_mass, height=args.height),
            barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=args.plate_mass),
        )
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        print(f"Error: invalid configuration -- {exc}")
        return 1

    builder_cls = EXERCISE_REGISTRY[args.exercise]

    try:
        xml_str = builder_cls(config).build()
    except (ValueError, RuntimeError) as exc:
        logger.error("Model build failed for '%s': %s", args.exercise, exc)
        print(f"Error: model build failed -- {exc}")
        return 1

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(xml_str)
        except OSError as exc:
            logger.error("Failed to write output file '%s': %s", args.output, exc)
            print(f"Error: cannot write to '{args.output}' -- {exc}")
            return 1
        print(f"Wrote {args.output}")
    else:
        print(xml_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
