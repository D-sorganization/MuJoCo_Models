# MuJoCo Models

MuJoCo musculoskeletal models for classical barbell exercises.

## Exercises

| Exercise       | Module                     | Description                                  |
| -------------- | -------------------------- | -------------------------------------------- |
| Back Squat     | `exercises.squat`          | High-bar back squat with barbell on trapezius |
| Bench Press    | `exercises.bench_press`    | Supine press with barbell gripped at chest    |
| Deadlift       | `exercises.deadlift`       | Conventional deadlift from floor to lockout   |
| Snatch         | `exercises.snatch`         | Wide-grip floor-to-overhead in one motion     |
| Clean and Jerk | `exercises.clean_and_jerk` | Floor to shoulders (clean) + overhead (jerk)  |
| Gait           | `exercises.gait`           | Walking gait cycle analysis                   |
| Sit to Stand   | `exercises.sit_to_stand`   | Chair rise movement pattern                   |

## Quick Start

```bash
pip install -e ".[dev]"
python3 -m pytest
```

### Generate a model

```python
from mujoco_models.exercises.squat.squat_model import build_squat_model

xml = build_squat_model(body_mass=80, height=1.75, plate_mass_per_side=60)
with open("squat.xml", "w") as f:
    f.write(xml)
```

## Architecture

- **`shared/`** -- Reusable components (DRY)
  - `barbell/` -- Olympic barbell model (IWF/IPF spec)
  - `body/` -- Full-body musculoskeletal model (Winter 2009 anthropometrics)
  - `contracts/` -- Design-by-Contract preconditions and postconditions
  - `utils/` -- MJCF generation helpers and geometry/inertia calculations
- **`exercises/`** -- Exercise-specific model builders
  - Each exercise inherits from `ExerciseModelBuilder` (base class)
  - Customizes barbell attachment and initial pose

## MuJoCo Conventions

- **Z-up**: Gravity is `(0, 0, -9.80665)`, vertical axis is Z
- **MJCF format**: Models output `<mujoco>` XML, not `.osim`
- **Weld constraints**: Barbell-to-body attachment via `<equality><weld>`
- **Hinge joints**: Sagittal-plane flexion/extension about X-axis

## Design Principles

- **TDD** -- Tests written alongside models; CI enforces 80% coverage
- **Design by Contract** -- All inputs validated via preconditions; outputs checked via postconditions
- **DRY** -- Shared base class, shared barbell/body models, shared MJCF helpers
- **Law of Demeter** -- Exercise builders interact only with public APIs of shared components

## Used by UpstreamDrift

This repo publishes a `model_pack/v1` manifest (`model_pack.yaml` at repo
root) and a `biomech.model_pack` entry point (`mujoco_models.model_pack`)
so that the UpstreamDrift launcher can discover and host these exercises.
See the umbrella tracking issue:
[UpstreamDrift#5179](https://github.com/D-sorganization/UpstreamDrift/issues/5179).

Discovery API:

```python
from mujoco_models.model_pack import resolve, manifest, list_exercises

resolve()         # absolute path to the installed exercises directory
manifest()        # parsed model_pack.yaml as a dict
list_exercises()  # ['squat', 'deadlift', ...]
```

CLI export contract:

```bash
python -m mujoco_models --exercise gait --export /tmp/gait.xml
mujoco-models --list-exercises
```

## License

MIT
