# CLAUDE.md -- MuJoCo_Models

## What This Is

MuJoCo musculoskeletal simulation models for classical barbell exercises
(squat, deadlift, bench press, snatch, clean & jerk) plus gait and
sit-to-stand. Python package that generates MJCF XML files.

## Key Directories

- `src/mujoco_models/` -- importable package (source of truth)
- `src/mujoco_models/exercises/` -- one sub-package per exercise
- `src/mujoco_models/optimization/` -- trajectory optimiser, IK solver, objectives
- `src/mujoco_models/shared/` -- barbell, body model, contracts, geometry, MJCF helpers
- `tests/` -- pytest suite (unit, integration, parity)
- `rust_core/` -- optional Rust accelerator (PyO3)
- `.github/workflows/` -- CI (lint, format, mypy, tests, security)

## Python and Tooling

- **Python >=3.10**. Use `python3`.
- **Formatter:** ruff format (line-length 88).
- **Linter:** ruff check.
- **Type checker:** mypy (strict on src/).

## Development Commands

```bash
pip install -e ".[dev]"                         # install with dev deps
ruff check src tests scripts                    # lint
ruff format --check src tests scripts           # format check
mypy src --config-file pyproject.toml           # type check
python3 -m pytest -m "not slow and not requires_mujoco" --cov=src --cov-fail-under=80
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on every commit: ruff lint+fix, ruff format,
no-wildcard-imports, no-debug-statements, no-print-in-src, prettier (YAML/JSON/MD).

### Test Markers

- `@pytest.mark.slow` -- long-running tests (excluded from default CI run)
- `@pytest.mark.requires_mujoco` -- needs MuJoCo installed (excluded from default CI run)
- `@pytest.mark.integration` -- integration tests
- `@pytest.mark.unit` -- unit tests
- `@pytest.mark.benchmark` -- performance benchmarks

Tests run in parallel by default (`-n auto --dist loadscope`).

### Docker

```bash
docker build -t mujoco-models .                 # build runtime image
docker build --target training -t mujoco-train . # build training image (adds gymnasium, stable-baselines3)
docker run -it mujoco-models bash                # interactive shell
```

## CI Requirements (All Must Pass)

1. `ruff check` -- zero violations
2. `ruff format --check` -- zero diffs
3. `mypy` -- no errors on src/
4. No TODO/FIXME without tracked GitHub issue
5. `bandit` security scan -- no high-severity findings
6. `pip-audit` -- no known vulnerabilities
7. pytest with **80% coverage minimum**
8. Multi-version matrix: Python 3.10, 3.11, 3.12

## Coding Standards

- **No print() in src/** -- use `logging` module
- **No bare except** -- catch specific exceptions
- **No wildcard imports**
- **Type hints** on all public functions
- **Docstrings** on all public functions and classes
- **Module budget:** max 300 lines per file (split data modules as needed)
- **DbC:** preconditions validate inputs with ValueError, postconditions verify outputs
- **DRY:** shared base class, shared barbell/body models, shared MJCF helpers
- **Law of Demeter:** no method chains >2 levels

## Architecture Notes

- All exercises inherit from `ExerciseModelBuilder` (base.py)
- Barbell attachment and initial pose are exercise-specific
- Coordinate convention: Z-up, gravity (0, 0, -9.80665)
- Output format: MJCF XML (`<mujoco>` root element)
- Exercise objective data is split across `exercise_objective_data.py` (strength)
  and `exercise_objective_data_aux.py` (Olympic lifts, gait, sit-to-stand)

## Git Workflow

- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- Branch naming: `feat/description`, `fix/description`
- PRs must pass CI before merge
