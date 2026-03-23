# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
