# Contributing to MuJoCo Models

Thank you for your interest in contributing! This document provides guidelines
for contributing to the MuJoCo Models project.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install in development mode:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Quality

All code must pass the following before merge:

- **Linting**: `ruff check src scripts tests examples`
- **Formatting**: `ruff format --check src scripts tests examples`
- **Type checking**: `mypy src --config-file pyproject.toml`
- **Tests**: `python3 -m pytest tests/ -v`

### Writing Tests

- Place unit tests in `tests/unit/` mirroring the `src/` structure.
- Place integration tests in `tests/integration/`.
- All new features must include tests with at least 80% coverage.
- Use `hypothesis` for property-based tests where appropriate.

### Commit Messages

Use conventional commit format:

```
type: concise description

Longer explanation if needed.
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `ci`, `chore`.

## Pull Request Process

1. Ensure all CI checks pass.
2. Update documentation if you changed public APIs.
3. Request review from at least one maintainer.
4. Squash merge when approved.

## Code Style

- Follow PEP 8 (enforced by ruff).
- Type-annotate all public functions (`disallow_untyped_defs = true`).
- Use Design-by-Contract: validate inputs with preconditions, outputs with postconditions.
- Keep functions small and single-purpose.
- No `TODO` or `FIXME` in code -- create GitHub issues instead.

## Reporting Issues

Use the GitHub issue tracker. Include:

- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS
