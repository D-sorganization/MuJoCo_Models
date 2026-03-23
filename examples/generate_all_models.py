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


def main() -> None:
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = ExerciseConfig(
        body_spec=BodyModelSpec(total_mass=args.body_mass, height=args.height),
        barbell_spec=BarbellSpec.mens_olympic(),
    )

    # Deduplicate builders (some exercises share a builder, e.g. squat/back_squat)
    seen: set[str] = set()
    for name, builder_cls in sorted(EXERCISE_REGISTRY.items()):
        cls_name = builder_cls.__name__
        if cls_name in seen:
            continue
        seen.add(cls_name)

        xml_str = builder_cls(config).build()
        out_path = os.path.join(args.output_dir, f"{name}.xml")
        with open(out_path, "w") as fh:
            fh.write(xml_str)
        print(f"Wrote {out_path}")

    print(f"Generated {len(seen)} models in {args.output_dir}/")


if __name__ == "__main__":
    main()
