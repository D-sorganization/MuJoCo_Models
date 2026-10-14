# Development Log — MuJoCo_Models

State table for every feature in flight in this repository. Update
entries **in place**; never append dated sections. One entry per
feature, from proposal to ship. See the `development-logs` section of
`AGENTS.md` for the binding rules and
`shared_scripts/development_log.py` for the validator.

- **Portfolio:** work
- **WIP limit:** 5
- **Last audited:** 2026-08-28 by bootstrap

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked`
reachable from any live state and `abandoned` from `parked`.
`shipped` never returns to `in_progress`; open a new entry instead.

## Active

### DL-#496 · Exclude Upper-Arm Versus Bench Contact at the Bench-Press Start Pose

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #496
- **Branch:** fix/496-bench-start-contacts
- **PR:** #497
- **Paths:** see #497
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`8d8dffe8`; collated from changes/496-exclude-upper-arm-versus-bench-contact-a.md)
- **Summary:** Exclude upper-arm versus bench contact at the bench-press start pose
- **Next step:** Merge the PR.

### DL-#486 · Foot-Seating Equilibrium Solve Disables the Constraint Warm Start so MuJoCo 3.15 Cannot Terminate It at Zero Iterations

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #486
- **Branch:** fix/486-foot-seating
- **PR:** #487
- **Paths:** see #487
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`c438da20`; collated from changes/486-foot-seating-equilibrium-solve-disables.md)
- **Summary:** Foot-seating equilibrium solve disables the constraint warm start so MuJoCo 3.15 cannot terminate it at zero iterations
- **Next step:** Merge the PR; consider running requires_mujoco integration tests in CI (separate workflow PR).

### DL-#428 · Set Bench Mass to 1E-4 to Match Cross-Engine Parity Standard

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #428
- **Branch:** fix/428-bench-mass
- **PR:** #490
- **Paths:** see #490
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6d0cc53c`; collated from changes/428-set-bench-mass-to-1e-4-to-match-cross-en.md)
- **Summary:** Set bench mass to 1e-4 to match cross-engine parity standard
- **Next step:** Merge the PR.

### DL-#453 · Contact-Aware Inverse Dynamics: Inverse_Dynamics_With_Contacts Reports the Contact Generalized Force Next to the Free-Root Residual; Gait and Sit-to-Stand Feet Seated at the Contact Equilibrium Like the Lifts; Grip.Dense_Efc_Jacobian Picks the Jacobian Layout From Mj_isSparse Instead of the Unset Sparse Row Metadata (MuJoCo 3.15 Segfault and Zero Contact Load)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #453
- **Branch:** fix/453-id-contacts
- **PR:** #482
- **Paths:** see #482
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`376412d7`; collated from changes/453-contact-aware-inverse-dynamics-inverse-d.md)
- **Summary:** Contact-aware inverse dynamics: inverse_dynamics_with_contacts reports the contact generalized force next to the free-root residual; gait and sit-to-stand feet seated at the contact equilibrium like the lifts; grip.dense_efc_jacobian picks the Jacobian layout from mj_isSparse instead of the unset sparse row metadata (MuJoCo 3.15 segfault and zero contact load)
- **Next step:** Merge the PR.

### DL-#400 · Bring Src Files Under the 300-Line Budget and Enforce It in CI

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #400
- **Branch:** refactor/400-module-line-budget
- **PR:** #466
- **Paths:** see #466
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`eb788964`; collated from changes/400-bring-src-files-under-the-300-line-budge.md)
- **Summary:** Bring src files under the 300-line budget and enforce it in CI
- **Next step:** Merge the PR.

### DL-#411 · \ Import Pinocchio_Models URDF Into MuJoCo and Compare Fingerprints\

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #411
- **Branch:** feat/411-pinocchio-urdf-fingerprint
- **PR:** #475
- **Paths:** see #475
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`d4d35632`; collated from changes/411-import-pinocchio-models-urdf-into-mujoco.md)
- **Summary:** \ Import Pinocchio_Models URDF into MuJoCo and compare fingerprints\
- **Next step:** \Merge the PR.\

### DL-#427 · Exclude Barbell Self-Contacts and Bar-Lifter Contacts at the Start Pose

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #427
- **Branch:** fix/427-start-pose-contacts
- **PR:** #446
- **Paths:** see #446
- **Started:** 2026-10-11
- **Last verified:** 2026-10-11 (`c347a8ca`; collated from changes/427-exclude-barbell-self-contacts-and-bar-li.md)
- **Summary:** Exclude barbell self-contacts and bar-lifter contacts at the start pose
- **Next step:** Merge the PR.

### DL-#440 · Keep Bar Level via Weld Relpose in Tilted Hand Frame Instead of Wrist Counter-Rotation

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #440
- **Branch:** claude/grip-weld-orientation-440
- **PR:** #443
- **Paths:** see #443
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`0e080946`; collated from changes/440-keep-bar-level-via-weld-relpose-in-tilte.md)
- **Summary:** Keep bar level via weld relpose in tilted hand frame instead of wrist counter-rotation
- **Next step:** Open PR, run CI, merge

### DL-#438 · Fix Mirrored Shoulder-Adduct Joint Range for Anatomical Abduction/Adduction

- **State:** in_review
- **Owner:** claude
- **Issue:** #438
- **Branch:** claude/shoulder-rom-438
- **PR:** #441
- **Paths:** `src/mujoco_models/shared/body/segment_data.py`, `src/mujoco_models/exercises/base.py`, `src/mujoco_models/exercises/snatch/snatch_model.py`, `src/mujoco_models/shared/parity/parity_divergences.json`, `tests/unit/shared/test_segment_data.py`, `tests/integration/test_grip_weld_residual.py`, `tests/integration/test_keyframe_within_joint_ranges.py`
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`f8af32a4`; collated from changes/438-fix-mirrored-shoulder-adduct-joint-range.md)
- **Summary:** Fix mirrored shoulder-adduct joint range so the snatch reaches its documented 0.60 m grip width
- **Next step:** Open the PR and arm auto-merge.

### DL-#408 · Fix Conflicting Grip-Weld Constraints at the Barbell Exercise Initial Pose

- **State:** in_review
- **Owner:** claude
- **Issue:** #408
- **Branch:** claude/grip-welds-408
- **PR:** #437
- **Paths:** `src/mujoco_models/exercises/base.py`, `src/mujoco_models/exercises/deadlift/deadlift_model.py`, `src/mujoco_models/exercises/snatch/snatch_model.py`, `src/mujoco_models/exercises/clean_and_jerk/clean_and_jerk_model.py`, `src/mujoco_models/shared/body/body_anthropometrics.py`, `tests/integration/test_grip_weld_residual.py`
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`20a99010`; collated from changes/408-fix-conflicting-grip-weld-constraints-at.md)
- **Summary:** Fix conflicting grip-weld constraints at the barbell exercise initial pose
- **Next step:** Open the PR and arm auto-merge.

### DL-#417 · Set Sit_to_Stand Chair_to_World Weld Relpose to the Chair's Keyframe Offset

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #417
- **Branch:** agy-417
- **PR:** #435
- **Paths:** see #435
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`b6168284`; collated from changes/417-set-sit-to-stand-chair-to-world-weld-rel.md)
- **Summary:** Set sit_to_stand chair_to_world weld relpose to the chair's keyframe offset
- **Next step:** Merge the PR.

### DL-#1605 · Adopt Mermaid C4 Architecture Map Contract

- **Issue:** #1605 (https://github.com/D-sorganization/Repository_Management/issues/1605)
- **State:** in_progress
- **Owner:** local (agent session bd082424-e57d-40ba-9962-3bf4420a5b33)
- **Branch:** docs/1605-c4-architecture-map
- **PR:** not created
- **Paths:** docs/architecture/C4.md, scripts/architecture_map_contract.py, tests/scripts/test_architecture_map_contract.py, .github/workflows/architecture-map-contract.yml
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`3f81ffb`)
- **Next step:** Open PR and merge with passing architecture map contract workflow.
- **Summary:** Establish and enforce the maintainable Mermaid C4 architecture-map contract for MuJoCo_Models per Repository_Management Epic #1594.

### DL-0001 · Codex Pr280

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`d8afc4f`)
- **Summary:** Seeded from local branch `codex/pr280`, which is
  17 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

### DL-0002 · Pr 280

- **State:** parked
- **Owner:** unassigned
- **PR:** not created
- **Paths:** `.` — scope not yet narrowed; set real globs when
  this entry is reactivated.
- **Started:** 2026-08-28
- **Last verified:** 2026-08-28 (`36d0802`)
- **Summary:** Seeded from local branch `pr-280`, which is
  19 commit(s) ahead of the default branch with no
  development-log entry.
- **Parked:** 2026-08-28 — seeded during fleet rollout. Assign a
  governing issue and set `Paths` before moving this to a live
  state; a live entry without a real issue is orphaned by
  definition.

## Shipped (Last 90 Days)

### DL-#493 · Run Requires_Mujoco Tests in the CI Tests Job (Drop the Marker Exclusion)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #493
- **Branch:** merged via #494
- **PR:** #494
- **Paths:** see #494
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`a89f115c`; collated from changes/493-run-requires-mujoco-tests-in-the-ci-test.md)
- **Summary:** Run requires_mujoco tests in the CI tests job (drop the marker exclusion)
- **Next step:** Shipped in PR #494.

### DL-#404 · Add Forward Dynamics Controller and Rollout API Tracking Joint Trajectories

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #404
- **Branch:** merged via #478
- **PR:** #478
- **Paths:** see #478
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`0878bffe`; collated from changes/404-add-forward-dynamics-controller-and-roll.md)
- **Summary:** Add forward dynamics controller and rollout API tracking joint trajectories
- **Next step:** Shipped in PR #478.

### DL-#392 · Re-Sync Vendored Model-Parity Bundle to Repository_Management Main (Standard V2.1, assemble.py Phase_Names From RM#2089)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #392
- **Branch:** merged via #484
- **PR:** #484
- **Paths:** see #484
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`ce868d5d`; collated from changes/392-parity-bundle-resync.md)
- **Summary:** Re-sync vendored model-parity bundle to Repository_Management main (standard v2.1, assemble.py phase_names from RM#2089)
- **Next step:** Shipped in PR #484.

### DL-#403 · Implement Task-Target Inverse Kinematics Solver

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #403
- **Branch:** merged via #477
- **PR:** #477
- **Paths:** see #477
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`f2ea0ae0`; collated from changes/403-implement-task-target-inverse-kinematics.md)
- **Summary:** Implement task-target inverse kinematics solver
- **Next step:** Shipped in PR #477.

### DL-#406 · Promote Standing GRF to a Public API With CoP Output

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #406
- **Branch:** merged via #472
- **PR:** #472
- **Paths:** see #472
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`ad05ee40`; collated from changes/406-promote-standing-grf-to-a-public-api-wit.md)
- **Summary:** Promote standing GRF to a public API with CoP output
- **Next step:** Shipped in PR #472.

### DL-#412 · Add Offscreen MP4 Video Rendering With MUJOCO_GL=Egl

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #412
- **Branch:** merged via #473
- **PR:** #473
- **Paths:** see #473
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`b236e5b3`; collated from changes/412-add-offscreen-mp4-video-rendering-with-m.md)
- **Summary:** Add offscreen MP4 video rendering with MUJOCO_GL=egl
- **Next step:** Shipped in PR #473.

### DL-#444 · Per-Hand Grip Wrench Extraction From Weld Equality Constraints and Static Hold Analysis (LIFT-4)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #444
- **Branch:** merged via #470
- **PR:** #470
- **Paths:** see #470
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`d2730935`; collated from changes/444-per-hand-grip-wrench-extraction-from-wel.md)
- **Summary:** Per-hand grip wrench extraction from weld equality constraints and static hold analysis (LIFT-4)
- **Next step:** Shipped in PR #470.

### DL-#460 · Give-Keyframe-Servos-a-Static-Hold-so-Crouched-Starts-Do-Not-Sag

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #460
- **Branch:** merged via #468
- **PR:** #468
- **Paths:** see #468
- **Started:** 2026-10-13
- **Last verified:** 2026-10-13 (`079ba317`; collated from changes/460-give-keyframe-servos-a-static-hold-so-cr.md)
- **Summary:** Give-keyframe-servos-a-static-hold-so-crouched-starts-do-not-sag
- **Next step:** Shipped in PR #468.

### DL-#455 · Weld the Bench-Press Root at the Standard V2.0 Via-Pose

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #455
- **Branch:** merged via #464
- **PR:** #464
- **Paths:** see #464
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`1020aee6`; collated from changes/455-weld-the-bench-press-root-at-the-standar.md)
- **Summary:** Weld the bench-press root at the standard v2.0 via-pose
- **Next step:** Shipped in PR #464.

### DL-#456 · Name Gait and Sit-to-Stand Phases per Standard V2.0

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #456
- **Branch:** merged via #463
- **PR:** #463
- **Paths:** see #463
- **Started:** 2026-10-12
- **Last verified:** 2026-10-12 (`52a2e32a`; collated from changes/456-name-gait-and-sit-to-stand-phases-per-st.md)
- **Summary:** Name gait and sit-to-stand phases per standard v2.0
- **Next step:** Shipped in PR #463.

### DL-#407 · Barbell Lifts Start at the Parity Standard V2.1 Start Via-Pose (Coordinates in the Keyframe, Refs 0, Pelvis Root, Bar, Spec Load); Spec.via_Pose_Phase Replaces the Local Table (UpstreamDrift#11743)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #407
- **Branch:** merged via #461
- **PR:** #461, #481
- **Paths:** see #461
- **Started:** 2026-10-12
- **Last verified:** 2026-10-13 (`d298213c`; collated from changes/407-apply-real-exercise-initial-poses-via-in.md)
- **Summary:** Barbell lifts start at the parity standard v2.1 start via-pose (coordinates in the keyframe, refs 0, pelvis root, bar, spec load); spec.via_pose_phase replaces the local table (UpstreamDrift#11743)
- **Next step:** Shipped in PR #461.

### DL-#401 · Lift Objectives Consume the Vendored Parity Standard V2.0: Phase Names, Timing and Start/Catch/Lockout Via-Poses Come From the Spec (LIFT-2, UpstreamDrift#11742); Deadlift Now Has the 5 Standard Phases.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #401
- **Branch:** claude/lift-2-standard-v2
- **PR:** #457
- **Paths:** see #457
- **Started:** 2026-10-11
- **Last verified:** 2026-10-11 (`96ff6e19`; collated from changes/401-lift-objectives-consume-the-vendored-par.md)
- **Summary:** Lift objectives consume the vendored parity standard v2.0: phase names, timing and start/catch/lockout via-poses come from the spec (LIFT-2, UpstreamDrift#11742); deadlift now has the 5 standard phases.
- **Next step:** Shipped in PR #457.

### DL-#447 · Seat Lift-Model Foot Boxes at the Soft-Contact Equilibrium Depth (0.78-0.86 mm) so the Start Pose Does Not Settle and Contact Survives Sub-mm Root Float

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #447
- **Branch:** merged via #452
- **PR:** #452
- **Paths:** see #452
- **Started:** 2026-10-11
- **Last verified:** 2026-10-11 (`5de7f98b`; collated from changes/447-seat-lift-model-foot-boxes-at-the-soft-c.md)
- **Summary:** Seat lift-model foot boxes at the soft-contact equilibrium depth (0.78-0.86 mm) so the start pose does not settle and contact survives sub-mm root float
- **Next step:** Shipped in PR #452.

### DL-#1917 · Sync the Agent-Identity Managed Block: The Owner's Personal Token Is an Approved Agent Identity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #1917
- **Branch:** merged via #449
- **PR:** #449
- **Paths:** see #449
- **Started:** 2026-10-11
- **Last verified:** 2026-10-11 (`7c3fa0ea`; collated from changes/1917-sync-the-agent-identity-managed-block-th.md)
- **Summary:** Sync the agent-identity managed block: the owner's personal token is an approved agent identity
- **Next step:** Shipped in PR #449.

### DL-#402 · Add Public Forward_Kinematics API With Real-MuJoCo Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #402
- **Branch:** merged via #433
- **PR:** #433
- **Paths:** see #433
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`54ba188d`; collated from changes/402-add-public-forward-kinematics-api-with-r.md)
- **Summary:** Add public forward_kinematics API with real-MuJoCo test
- **Next step:** Shipped in PR #433.

### DL-#405 · Add Public Inverse_Dynamics Wrapper Around Mj_Inverse With Real-Engine Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #405
- **Branch:** merged via #431
- **PR:** #431
- **Paths:** see #431
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`854ba92b`; collated from changes/405-add-public-inverse-dynamics-wrapper-arou.md)
- **Summary:** Add public inverse_dynamics wrapper around mj_inverse with real-engine tests
- **Next step:** Shipped in PR #431.

### DL-#399 · Probe MuJoCo GL Backend in conftest.py and Fall Back to Osmesa or Unset on Failure

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #399
- **Branch:** merged via #429
- **PR:** #429
- **Paths:** see #429
- **Started:** 2026-10-09
- **Last verified:** 2026-10-09 (`2eaf83f0`; collated from changes/399-probe-mujoco-gl-backend-in-conftest-py-a.md)
- **Summary:** Probe MuJoCo GL backend in conftest.py and fall back to osmesa or unset on failure
- **Next step:** Shipped in PR #429.

### DL-#2021 · Fix(Ci): Isolate RUSTUP_HOME per Workspace for the Rust Job (RM#2021)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2021
- **Branch:** merged via #425
- **PR:** #425
- **Paths:** see #425
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`6a7c19fd`; collated from changes/2021-fix-ci-isolate-rustup-home-per-workspace.md)
- **Summary:** fix(ci): isolate RUSTUP_HOME per workspace for the Rust job (RM#2021)
- **Next step:** Shipped in PR #425.

### DL-#422 · SECURITY: Guard Fork PRs Off the Self-Hosted Fleet; Vendor Fork_Pr_Runner_Guard and Run It in CI (RM#1989)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #422
- **Branch:** merged via #423
- **PR:** #423
- **Paths:** see #423
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`8a687bd5`; collated from changes/422-security-guard-fork-prs-off-the-self-hos.md)
- **Summary:** SECURITY: guard fork PRs off the self-hosted fleet; vendor fork_pr_runner_guard and run it in CI (RM#1989)
- **Next step:** Shipped in PR #423.

### DL-#2011 · Fingerprint Reports Test-Pose Origins in the Pelvis Frame

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2011
- **Branch:** feat/issue-2011-test-pose-origins
- **PR:** #420
- **Paths:** see #420
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`8bb03799`; collated from changes/2011-test-pose-origins.md)
- **Summary:** Re-vendored parity bundle (standard 1.2.0, topology.py, Repository_Management#2011 slice 2). The MuJoCo fingerprint reports the pelvis rotation and segment origins at the standard's three test poses; conformance checks them against the reference forward kinematics with zero origin and pose divergences for every exercise.
- **Next step:** Shipped in PR #420.

### DL-#410 · Canonical Axis Convention

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #410
- **Branch:** fix/issue-410-canonical-axes
- **PR:** #418
- **Paths:** see #418
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`1861f66e`; collated from changes/410-align-the-mujoco-models-with-the-canonic.md)
- **Summary:** Align the MuJoCo models with the canonical axis convention (X forward, Y left, Z up): bilateral segments offset along Y, joint axes from the fleet standard, barbell along Y, and a real-engine axis fingerprint.
- **Next step:** Shipped in PR #418.

### DL-#2019 · Vendor RM-5 Change-Fragment Tooling and Test Suite

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2019
- **Branch:** feat/2019-vendor-rm-5-change-fragment-tooling
- **PR:** #414, #415
- **Paths:** see #414
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`98bedf64`; collated from changes/2019-wire-collate-changes-workflow-and-check.md)
- **Summary:** vendor RM-5 change-fragment tooling and test suite
- **Next step:** Shipped in PR #414.

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.
