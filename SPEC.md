# SPEC.md - MuJoCo_Models Repository Specification

## 1. Identity

| Field | Value |
| --- | --- |
| Repository | `MuJoCo_Models` |
| GitHub | `https://github.com/D-sorganization/MuJoCo_Models` |
| Primary language | Python 3.10+ |
| Package name | `mujoco_models` |
| License | MIT |

## 2. Purpose

MuJoCo_Models provides MuJoCo MJCF model builders for classical barbell exercises and related movement patterns. The repository focuses on generating well-formed XML models, validating model structure, and keeping shared anthropometric and MJCF assembly logic reusable across exercises.

## 3. Current Layout

The maintained source lives under `src/mujoco_models/` and is organized around three layers:

- `src/mujoco_models/exercises/` contains the exercise-specific model builders and the registry used by the CLI and tests.
- `src/mujoco_models/shared/` contains reusable body, barbell, contract, geometry, parity, and MJCF helper code.
- `src/mujoco_models/optimization/` contains objective data and optimization helpers used by model-generation and validation workflows.

The shared body layer is split into focused modules:

- `shared/body/body_model.py` is a thin facade that re-exports `BodyModelSpec`, `_LEG_HEIGHT_FRACTION`, and `create_full_body()`, delegating to the sub-modules below.
- `shared/body/axial_skeleton.py` builds the pelvis, torso, and head (the axial skeleton).
- `shared/body/upper_limbs.py` attaches bilateral upper-arm, forearm, and hand segments.
- `shared/body/lower_limbs.py` attaches bilateral thigh, shank, and foot segments with contact geometry.
- `shared/body/body_helpers.py` contains shared limb assembly and foot-contact helpers.
- `shared/body/segment_data.py` contains anthropometric tables, segment property helpers, and shared joint limits.

The shared barbell layer lives in `shared/barbell/barbell_model.py` and builds the three-body barbell assembly plus weld constraints.

## 4. Public Interface

The supported entrypoint is module execution:

- `python -m mujoco_models <exercise> [options]`

The CLI is implemented in `src/mujoco_models/__main__.py` and builds models from:

- `ExerciseConfig`
- `BodyModelSpec`
- `BarbellSpec`
- `EXERCISE_REGISTRY`

The registry in `src/mujoco_models/exercises/__init__.py` exposes the supported builders and aliases:

- `squat` and `back_squat`
- `bench_press`
- `deadlift`
- `snatch`
- `clean_and_jerk`
- `gait`
- `sit_to_stand`

Each exercise module also exposes a convenience function named `build_*_model()` that returns an MJCF XML string.

## 5. Model Generation Flow

Model generation follows the same pipeline across exercises:

1. Build an `ExerciseConfig` from body and barbell specs.
2. Instantiate the exercise-specific `ExerciseModelBuilder`.
3. Create the `<mujoco>` root with options, compiler settings, and defaults.
4. Build the `worldbody` and `equality` sections.
5. Assemble the full body through `create_full_body()`.
6. Assemble the barbell through `create_barbell_bodies()` when the exercise builder's `uses_barbell` property is `True` (gait and sit-to-stand opt out).
7. Apply the exercise-specific barbell attachment and pose hooks.
8. Add contact exclusions, actuators, sensors, and keyframes.
9. Serialize and validate the MJCF XML.

Exercise subclasses only customize attachment strategy, optional worldbody hooks, and the initial pose. The shared base class in `src/mujoco_models/exercises/base.py` owns the common build pipeline.

## 6. Data And Configuration

The model-building APIs accept plain dataclass configuration rather than implicit global state.

- `BodyModelSpec` controls total mass and height for anthropometric scaling.
- `BarbellSpec` controls bar dimensions, mass, and optional plate mass per side.
- All lengths are expressed in meters and all masses in kilograms.
- MuJoCo uses a Z-up convention in this repository: gravity is `(0.0, 0.0, -9.80665)`.

The repo does not currently define a package console script. `python -m mujoco_models` is the stable runtime entrypoint.

## 7. Testing And CI

Testing is organized around unit, integration, and parity coverage under `tests/`.

- `tests/unit/` covers dataclasses, helpers, the CLI, the registry, and shared body/barbell behavior.
- `tests/integration/` validates that every exercise builds end-to-end into valid MJCF.
- `tests/parity/` checks parity and contract expectations.

The CI contract is defined in `.github/workflows/ci-standard.yml` and currently includes:

- `ruff check src scripts tests` (line length enforced at 88 characters)
- `ruff format --check src scripts tests`
- `mypy src --config-file pyproject.toml`
- `pytest` with coverage enforcement at 80% on the source tree
- a placeholder scan for `TODO` and `FIXME`
- `pip-audit`
- `bandit -r src/`

Local validation should follow the same shape as CI when changes affect source behavior. Documentation-only changes should still keep the spec truthful and aligned with the current tree.

## 8. Maintenance Rules

- Keep `SPEC.md` aligned with the actual package layout and supported entrypoints.
- Update the spec whenever the public model-generation surface changes.
- Preserve the shared body and barbell layers as the single source of truth for common MJCF assembly logic.
- Prefer adding new exercise behavior through subclasses and shared helpers rather than duplicating body assembly logic.

