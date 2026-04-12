# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-04-11

### Refactored (issue #118)

- **Reusability**: Added `uses_barbell` property to `ExerciseModelBuilder` (default `True`) so `build()` can skip barbell construction and attachment when an exercise has no barbell.
- **Code smells**: Removed the zero-mass `BarbellSpec.mens_olympic(plate_mass_per_side=0.0)` hack from `build_gait_model()` and `build_sit_to_stand_model()`. Their builders now override `uses_barbell = False`, and the barbell bodies/welds are omitted entirely rather than materialised with zero plate mass.

## [Unreleased] - 2026-03-30

### Fixed (A-N Assessment Remediation — issue #90)

- **DbC**: Added precondition guards (`n_frames >= 2`, non-empty phases) to `interpolate_phases()` in `trajectory_optimizer.py`.
- **DbC**: Added shape and size validation to `point_in_polygon()` in `polygon_geometry.py`.
- **LoD**: Broke `Path(__file__).resolve().parent.parent` chain in `setup_dev.py` into named `_SCRIPT_DIR` and `PROJECT_ROOT` intermediates.
- **Documentation**: Added missing docstrings to `main()` in `setup_dev.py` and `generate_all_models.py`.
- **Documentation**: Added docstrings to `ExerciseModelBuilder.__init__()`, `BarbellSpec.__post_init__()`, and `BodyModelSpec.__post_init__()`.
- **Documentation**: Added docstrings to `shaft_radius` and `sleeve_radius` properties in `BarbellSpec`.
- **Documentation**: Added docstrings to `exercise_name` property in all 7 exercise model builders.
- **Documentation**: Added docstring to `_rad()` helper in `shared/parity/standard.py`.

## [0.1.0] - 2026-03-22

### Added

- Full-body musculoskeletal model with 15 segments (Winter 2009 anthropometrics).
- Olympic barbell model with shaft and bilateral sleeves.
- Five exercise builders: squat, bench press, deadlift, snatch, clean and jerk.
- Bilateral barbell attachment (both hands) for grip exercises.
- CLI entry point: `python -m mujoco_models <exercise>`.
- Exercise registry for programmatic discovery of builders.
- Design-by-Contract precondition and postcondition checks.
- Geometry utilities: cylinder, box, sphere inertia; parallel axis theorem.
- Hypothesis property-based tests and segment data integrity tests.
- Edge-case anthropometric tests.
- ExerciseConfig validation tests.
- CI pipeline with Python 3.10-3.12 matrix, ruff, mypy, bandit, pip-audit.
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md.
- GitHub issue/PR templates and dependabot configuration.
- Benchmarks for model build performance.
- `.dockerignore` for lean container images.
