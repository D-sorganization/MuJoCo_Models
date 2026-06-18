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
- `exercises/`: Core entry points (`SquatModelBuilder`, `DeadliftModelBuilder`, etc.)
- `shared/`: Model parts (body, barbell, basic contracts, MJCF helpers)
- `optimization/`: Routines to build kinematic cost functions and balance limits for trajectory tracking.

## 4. Key Performance Characteristics

- **Generation:** XML `serialize_model` is custom logic via `_fast_serialize_node` (no `ElementTree.tostring` overhead) and builds strings using direct append of `%` formatting output.
- **Cost evaluation:** Tight Python math loops unroll 3-vectors and use scalar math and `math.isfinite()`. Array dispatching is eliminated dynamically.

## 5. Development

Code complies with standard Python formatters (`ruff`), runs tests (`pytest`), and implements strict typing constraints (`mypy`). Pre-commit hooks enforce these bounds.

*(Last updated: 2026-06-18 to reflect optimization improvements in MJCF text generation.)*
