# Project Charter

> Drafted 2026-09-25 by the fleet charter sweep (Gemini) from README, git history, and open issues/PRs.
> The project-steward role keeps this current; owners should correct feature statuses.

## End Goal

MuJoCo_Models provides biomechanically accurate, computationally efficient MuJoCo MJCF musculoskeletal model builders for classical barbell exercises and foundational human movement patterns (back squat, bench press, deadlift, snatch, clean and jerk, gait, and sit-to-stand), adhering to Winter 2009 anthropometrics and IWF/IPF barbell specifications. The project is done when all supported barbell exercises and movements generate valid, self-contained MJCF XML configurations with kinematic constraints, integrate seamlessly with the UpstreamDrift model pack ecosystem, and maintain sub-millisecond generation and trajectory optimization performance backed by 80%+ test coverage.

## Non-Goals

- Simulating sports movements or equipment outside barbell weightlifting and foundational locomotion.
- Physics engine runtime or interactive graphical simulation viewer development (delegated to MuJoCo runtime).
- Exporting to OpenSim `.osim` format (MJCF XML is the sole target).
- Muscle-tendon finite element modeling or soft tissue deformation simulation.
- Real-time motion capture hardware integration or data acquisition.

## Features

| ID | Feature | Status | Tracking | Notes |
| --- | --- | --- | --- | --- |
| F1 | Back Squat Model Builder | shipped | - | High-bar back squat generator with trapezius barbell weld |
| F2 | Bench Press Model Builder | shipped | - | Supine press model with bilateral hand barbell grips |
| F3 | Deadlift Model Builder | shipped | - | Conventional floor-to-lockout deadlift model builder |
| F4 | Snatch Model Builder | shipped | - | Wide-grip Olympic floor-to-overhead snatch model generator |
| F5 | Clean and Jerk Model Builder | shipped | - | Floor-to-shoulders clean and overhead jerk model builder |
| F6 | Gait Movement Model Builder | shipped | #118 | Locomotion kinematics model without barbell requirement |
| F7 | Sit to Stand Model Builder | shipped | #118 | Chair rise movement pattern model without barbell requirement |
| F8 | Shared Anthropometric Body Model | shipped | #110 | Winter 2009 segmented axial skeleton and upper/lower limbs |
| F9 | Olympic Barbell Model | shipped | - | IWF and IPF specification bar and plate mass assembly |
| F10 | Exercise Base Builder and Registry | shipped | #119 | Common MJCF assembly pipeline, initial pose mapping, and registry |
| F11 | Trajectory Optimization Helpers | shipped | #205 | Cost computation for balance, bar path, and joint kinematics |
| F12 | CLI and Model Export Tools | shipped | #210 | Package console scripts and module CLI for exercise XML generation |
| F13 | UpstreamDrift Model Pack Manifest | shipped | #266 | Model pack v1 manifest and discovery API for UpstreamDrift |
| F14 | Rust Core Acceleration Engine | shipped | #359 | PyO3 native Rust extension module for performance-critical routines |
| F15 | C4 Architecture Contract Map | shipped | #362 | Mermaid C4 architecture map contract and validation script |
| F16 | Sphinx API Documentation | shipped | #194 | Sphinx documentation layout and API reference build |

## Links

- Status (generated): [`STATUS.md`](STATUS.md)
- Steward playbook: Repository_Management `docs/fleet-project-steward.md`
