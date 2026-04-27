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

## Developer Certificate of Origin (DCO)

By contributing to this project, you agree to the following terms:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### Sign-off Requirement

All commits must be signed off using `git commit -s`. This adds a
`Signed-off-by:` line to your commit message. For example:

```bash
git commit -s -m "feat: add new exercise model"
```

If you have already made commits without sign-off, amend them:

```bash
git rebase --signoff HEAD~N
```

Where `N` is the number of commits to amend. Then force-push to your
fork (never to `main`).

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
