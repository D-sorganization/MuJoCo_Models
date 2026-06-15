# SPEC.md - MuJoCo_Models Repository Specification

## 1. Identity

| Field            | Value                                              |
| ---------------- | -------------------------------------------------- |
| Repository       | `MuJoCo_Models`                                    |
| GitHub           | `https://github.com/D-sorganization/MuJoCo_Models` |
| Primary language | Python 3.10+                                       |
| Package name     | `mujoco_models`                                    |
| License          | MIT                                                |

## 2. Purpose

MuJoCo_Models provides MuJoCo MJCF model builders for classical barbell exercises and related movement patterns. The repository focuses on generating well-formed XML models, validating model structure, and keeping shared anthropometric and MJCF assembly logic reusable across exercises.

## 3. Current Layout

The maintained source lives under `src/mujoco_models/` and is organized around three layers:

- `src/mujoco_models/exercises/` contains the exercise-specific model builders and the registry used by the CLI and tests.
- `src/mujoco_models/shared/` contains reusable body, barbell, contract, geometry, parity, and MJCF helper code.
- `src/mujoco_models/optimization/` contains objective data and optimization helpers used by model-generation and validation workflows.
  - `trajectory_optimizer.py` exposes cost functions (`compute_balance_cost`, `compute_bar_path_cost`, etc.) with input validation extracted to dedicated `_validate_*` helpers to keep per-function cyclomatic complexity low.

The shared body layer is split into focused modules:

- `shared/body/body_model.py` is a thin facade that re-exports `BodyModelSpec`, `_LEG_HEIGHT_FRACTION`, and `create_full_body()`, delegating to the sub-modules below.
- `shared/body/axial_skeleton.py` builds the pelvis, torso, and head (the axial skeleton).
- `shared/body/upper_limbs.py` attaches bilateral upper-arm, forearm, and hand segments.
- `shared/body/lower_limbs.py` attaches bilateral thigh, shank, and foot segments with contact geometry.
- `shared/body/body_helpers.py` contains shared limb assembly and foot-contact helpers.
- `shared/body/segment_data.py` contains anthropometric tables, segment property helpers, and shared joint limits.

The shared barbell layer lives in `shared/barbell/barbell_model.py` and builds the three-body barbell assembly plus weld constraints.

Repository documentation for generated API references lives under `docs/` and is
built with Sphinx using `docs/conf.py`, `docs/index.rst`, and `docs/api.rst`.

## 4. Public Interface

The supported entrypoints are module execution and the package console script:

- `python -m mujoco_models <exercise> [options]`
- `mujoco-models <exercise> [options]`

The CLI is implemented in `src/mujoco_models/__main__.py` and builds models from:

- `ExerciseConfig`
- `BodyModelSpec`
- `BarbellSpec`
- `EXERCISE_REGISTRY`

The package root also re-exports the public exception hierarchy for callers that
need stable failure handling without importing internal modules:

- `MuJoCoModelError`
- `ModelBuildError`
- `ValidationError`
- `PreconditionError`

The registry in `src/mujoco_models/exercises/__init__.py` exposes the supported builders and aliases:

- `squat` and `back_squat`
- `bench_press`
- `deadlift`
- `snatch`
- `clean_and_jerk`
- `gait`
- `sit_to_stand`

Each exercise module also exposes a convenience function named `build_*_model()` that returns an MJCF XML string.

The package-level `__init__.py` also re-exports the exception hierarchy for downstream consumers:

- `MuJoCoModelError` (base exception)
- `ModelBuildError`
- `ValidationError`
- `PreconditionError`

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
The keyframe builder now walks `<body>` elements once and uses `find("freejoint")` to prepend free-joint pose data without the prior nested `ElementTree.iter()` traversal.

## 6. Data And Configuration

The model-building APIs accept plain dataclass configuration rather than implicit global state.

- `BodyModelSpec` controls total mass and height for anthropometric scaling.
- `BarbellSpec` controls bar dimensions, mass, and optional plate mass per side.
- All lengths are expressed in meters and all masses in kilograms.
- MuJoCo uses a Z-up convention in this repository: gravity is `(0.0, 0.0, -9.80665)`.

The package also exposes `mujoco-models` as a console script via `[project.scripts]` in `pyproject.toml`. Both entrypoints are functionally equivalent.

## 7. Testing And CI

Testing is organized around unit, integration, and parity coverage under `tests/`.

- `tests/unit/` covers dataclasses, helpers, the CLI, the registry, and shared body/barbell behavior.
- `tests/integration/` validates that every exercise builds end-to-end into valid MJCF.
- `tests/parity/` checks parity and contract expectations.

The CI contract is defined in `.github/workflows/ci-standard.yml` and currently includes:

- `ruff check src scripts tests` (line length enforced at 88 characters)
- cyclomatic complexity (McCabe) enforcement via `ruff` rule `C90` with a max complexity of 10
- `ruff format --check src scripts tests`
- `mypy src --config-file pyproject.toml` (aligned with the fleet-standard pin)
- `pytest` with coverage enforcement at 80% on the source tree
- the `dev` optional dependency set in `pyproject.toml`, including PyYAML for
  repository governance tests that parse tracked YAML metadata
- a placeholder scan for `TODO` and `FIXME`
- `pip-audit`
- `bandit -r src/`

Pip-audit CVE suppressions are tracked in
`docs/security/pip_audit_ignores.yml`. Each exception records the CVE, package,
reason, expiration date, remediation status, and tracking issue, and
`tests/unit/test_pip_audit_ignores.py` validates that structure.
The scheduled CVE exception monitoring workflow checks that suppression
expiration dates remain current and uploads an unsuppressed pip-audit report
from the same local self-hosted runner pool used by CI.

Documentation changes that affect the Sphinx configuration or API reference
surface should keep the generated-docs layout in `docs/` and the spec aligned
with that documentation entrypoint.

Local validation should follow the same shape as CI when changes affect source behavior. Documentation-only changes should still keep the spec truthful and aligned with the current tree.

Validation-related tests should assert the repository exception hierarchy. For
example, objective phase lookup failures are expected to raise
`ValidationError`, and CLI logging helper tests should patch the repository's
structured logging hook rather than the standard-library `logging.basicConfig`
implementation detail.

Contributor governance is documented in `CONTRIBUTING.md` and reinforced by
`.github/PULL_REQUEST_TEMPLATE.md`. The local pre-commit configuration includes
a `commit-msg` hook implemented by `scripts/check_commit_message.py` to validate
commit subjects against the repository's conventional commit types:
`feat`, `fix`, `refactor`, `test`, `docs`, `ci`, and `chore`.

## 8. Licensing

All Python source files in `src/` include an SPDX license header:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
```

This header is present in every module-level `.py` file as of the SPDX header update.

## 9. Maintenance Rules

- Keep `SPEC.md` aligned with the actual package layout and supported entrypoints.
- Update the spec whenever the public model-generation surface changes.
- Preserve the shared body and barbell layers as the single source of truth for common MJCF assembly logic.
- Prefer adding new exercise behavior through subclasses and shared helpers rather than duplicating body assembly logic.

### Performance Optimization History

- Performance optimizations using unrolled scalar operations over numpy methods for small arrays and inlined Python math scalar checks (e.g., `math.isfinite`) are encouraged for hot-path calculations, such as in `src/mujoco_models/shared/contracts/preconditions.py` and `src/mujoco_models/shared/utils/geometry.py`.
- Inlining basic math operations (like `math.isfinite`) directly into precondition checks reduces Python function stack frame overhead and is encouraged for methods called repeatedly in tight loops.
- ElementTree (`ET`) loop traversals during XML generation are optimized by avoiding nested loop passes and preferring `.find()` and combined iterators where applicable, as demonstrated in `ExerciseModelBuilder`.
- Unrolled 1D slice operations (`dx*dx + dy*dy`) are preferred over intermediate 2D array allocations and `np.sum(..., axis=1)` for small matrix reductions, as demonstrated in `src/mujoco_models/optimization/trajectory_optimizer.py`.
- Hardcoded string formatting using `%` operators (e.g. `"%.6f %.6f %.6f" % (x, y, z)`) is preferred over `"".join(f"{v:.6f}" for v in ...)` generator expressions and Python 3 f-strings for known small fixed-size tuples (e.g. 1D, 2D, 3D, or 7D arrays) during MJCF generation to avoid generator allocation and expression compilation overhead.
- Vectorized array operations like `np.interp` over all frames at once are preferred over Python `for` loops evaluating piecewise functions frame-by-frame during trajectory generation.
- Checking for finiteness of NumPy arrays using `np.isfinite(arr).all()` is preferred over `np.all(np.isfinite(arr))` to avoid Python function dispatch overhead in tight loops.
- Inlining array finiteness checks directly inside validation guards using `np.isfinite(arr).all()` rather than delegating them to helper functions avoids function call overhead in tight optimization loops.

### CI Runner Routing

- CI workflows must route to local self-hosted runners. The standard CI runner
  selector retries transient runner inventory failures, but it must keep
  `d-sorg-fleet` as the selected runner instead of falling back to hosted
  GitHub runners.

<!-- Update trigger for CI freshness check -->

## Performance Notes

- 2026-06-11: Optimized builtin min/max in tight geometry loops. Replaced `max(0, min(1, t))` with faster explicit `if/elif` logic in `_point_to_segment_sq`.
- 2026-06-11: Added constraint on matplotlib `<3.10` to avoid fonttools dependency conflict during CI tests with Python 3.10.
- 2026-06-14: Optimized array indexing in trajectory optimizer loops. Converted `polygon` NumPy arrays to native Python lists using `.tolist()` before iterating in `_point_in_polygon` and `_squared_distance_to_polygon` to bypass NumPy's C-API dispatch and scalar casting overhead.
- 2026-06-15: Optimized point_in_polygon and squared_distance_to_polygon in polygon_geometry.py by converting 2D NumPy arrays to nested Python lists using `.tolist()` before loop iteration, eliminating expensive C-API array dispatch operations.
- 2026-06-15: Expanded the matplotlib dependency support window to `<3.11` so Python dependency automation can test the current 3.10.x microrelease series while keeping the existing lower bound and NumPy/MuJoCo contracts unchanged.
- 2026-06-15: The CI test matrix now installs into a per-job virtual environment and invokes pip through the selected interpreter so self-hosted runner site-packages cannot shadow the editable checkout.
