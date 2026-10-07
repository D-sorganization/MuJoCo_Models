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
