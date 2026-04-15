"""Generate MJCF XML files for all five exercise models.

Usage:
    python3 examples/generate_all_models.py [--output-dir OUTPUT_DIR]

Produces one XML file per exercise in the output directory (default: ./output/).
"""

from __future__ import annotations

import argparse
import os

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.exercises.base import ExerciseConfig
from mujoco_models.shared.barbell import BarbellSpec
from mujoco_models.shared.body import BodyModelSpec


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ``generate_all_models``."""
    parser = argparse.ArgumentParser(description="Generate all exercise models.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write XML files (default: output/).",
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
    return parser


def _build_config(body_mass: float, height: float) -> ExerciseConfig:
    """Build an :class:`ExerciseConfig` with validated anthropometry.

    Preconditions:
        ``body_mass > 0`` and ``height > 0``.  :class:`BodyModelSpec`
        enforces the contract and raises :class:`ValueError` otherwise.
    """
    return ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=body_mass, height=height),
        barbell_spec=BarbellSpec.mens_olympic(),
    )


def _write_model(
    name: str, builder_cls: type, config: ExerciseConfig, output_dir: str
) -> str:
    """Build one exercise model and write it to ``<output_dir>/<name>.xml``.

    Returns the output path so the caller can report progress.
    """
    xml_str = builder_cls(config).build()
    out_path = os.path.join(output_dir, f"{name}.xml")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(xml_str)
    return out_path


def _iter_unique_builders() -> list[tuple[str, type]]:
    """Yield ``(name, builder_cls)`` pairs with duplicate classes removed.

    Some exercises share a builder (e.g. squat/back_squat).  We emit each
    unique class once using the alphabetically-first alias.
    """
    seen: set[str] = set()
    unique: list[tuple[str, type]] = []
    for name, builder_cls in sorted(EXERCISE_REGISTRY.items()):
        cls_name = builder_cls.__name__
        if cls_name in seen:
            continue
        seen.add(cls_name)
        unique.append((name, builder_cls))
    return unique


def _generate_models(config: ExerciseConfig, output_dir: str) -> list[str]:
    """Generate each unique exercise model and return written file paths."""
    return [
        _write_model(name, builder_cls, config, output_dir)
        for name, builder_cls in _iter_unique_builders()
    ]


def _report_generation(paths: list[str], output_dir: str) -> None:
    """Print generation progress for CLI users."""
    for out_path in paths:
        print(f"Wrote {out_path}")
    print(f"Generated {len(paths)} models in {output_dir}/")


def main(argv: list[str] | None = None) -> None:
    """Generate MJCF XML files for all registered exercise models.

    Parses command-line arguments and writes one XML file per exercise
    to the configured output directory.
    """
    args = _build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    config = _build_config(args.body_mass, args.height)
    paths = _generate_models(config, args.output_dir)
    _report_generation(paths, args.output_dir)


if __name__ == "__main__":
    main()
