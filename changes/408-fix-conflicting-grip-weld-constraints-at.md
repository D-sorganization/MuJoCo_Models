---
issue: 408
summary: "Fix conflicting grip-weld constraints at the barbell exercise initial pose"
dl_state: "in_review"
next_step: "Open the PR and arm auto-merge."
title: "Fix Conflicting Grip-Weld Constraints at the Barbell Exercise Initial Pose"
owner: "claude"
branch: "claude/grip-welds-408"
paths: "`src/mujoco_models/exercises/base.py`, `src/mujoco_models/exercises/deadlift/deadlift_model.py`, `src/mujoco_models/exercises/snatch/snatch_model.py`, `src/mujoco_models/exercises/clean_and_jerk/clean_and_jerk_model.py`, `src/mujoco_models/shared/body/body_anthropometrics.py`, `tests/integration/test_grip_weld_residual.py`"
---

**Objective:** both grip welds (barbell-to-hand_l/r) must have under 5 mm
position residual at the initial pose, in real MuJoCo, for every barbell
exercise that grips the bar in both hands (#408).

**Root cause:** `grip_width` was only ever used as the weld's `relpose`
offset; nothing connected it to the body's actual pose, so the hand's real
lateral position stayed at the fixed `shoulder_half_width` (0.168 m)
regardless of the exercise's documented grip width. Fixed by adding shared
shoulder-abduction pose helpers (`ExerciseModelBuilder._grip_abduction_angle`
/ `_grip_pose_offsets` / `_achieved_grip_width` / `_grip_vertical_rise` in
`base.py`, plus `BodyModelSpec.arm_reach_length`) and wiring them into
deadlift, clean_and_jerk and snatch via `keyframe_angle_offsets` (the
mechanism bench_press already used correctly -- `set_ref_by_name_map` sets a
joint `ref`, and since MuJoCo kinematics apply `qpos - ref`, a pose set that
way is a kinematic no-op, confirmed empirically). The weld's grip width is
now computed from the pose actually achieved, so the Y residual is exact by
construction; `barbell_start_pos` is also raised by the hand's height gain
from abduction, closing a Z residual the first pass missed (5.8 mm /
77.8 mm before that correction).

**Snatch ROM clamp:** the shared `SHOULDER_ADDUCT_MIN` limit (-30 deg) caps
the achievable grip at this body's limb length to ~0.4585 m, short of the
documented ~0.60 m. The weld uses the ROM-clamped achieved width rather than
a hand-position-chasing offset, per the issue's explicit guidance to use
real shoulder abduction for wide grips, not a wider anchor. The shared ROM
constant itself was not touched (cross-repo parity standard, out of scope).

**Deferred, not fixed here:** hip_flex/knee initial-pose angles are still
set via the same `ref`-canceling `set_ref_by_name_map` path and so still do
not actually move the torso/legs at the keyframe (confirmed empirically) --
this is issue #392's territory and does not affect the grip-weld residual
(the hip/knee chain does not include the arms), so it is left untouched.

**Validation:** `pytest tests/integration/test_grip_weld_residual.py -v`
before: 3 failed (deadlift 52.0 mm, clean_and_jerk 82.0 mm, snatch 432.0 mm),
1 passed (bench_press, already 0). After: 4 passed (all residuals ~3e-7 m).
Full default `pytest` suite passes (parity/conformance included). `ruff
check`/`ruff format --check` clean on touched files. `mypy src` clean.
`bandit` shows only pre-existing low-severity findings unrelated to this
change.

**Blockers/risks:** none blocking. Snatch's effective grip width is now
narrower (~0.4585 m) than the previously-documented ~0.60 m, a real
biomechanical-fidelity change flagged for reviewer attention, not just a
bugfix. `base.py` grows to ~490 lines (already over this repo's informal
300-line module guideline before this change; not a CI gate here) -- noted
as a possible follow-up, not addressed in this PR.
