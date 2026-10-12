# SPDX-License-Identifier: MIT
"""The lift objectives consume the vendored exercise spec (standard v2.1).

UpstreamDrift#11742 (LIFT-2), Repository_Management#2060 (Option A): phase
names, phase timing and the start, catch and lockout via-poses come from the
vendored ``biomech_parity_standard.json``, never from pack-local tables.
"""

from __future__ import annotations

import pytest

from mujoco_models.optimization.exercise_objectives import get_exercise_objective
from mujoco_models.optimization.objective_data import spec_lifts
from mujoco_models.shared.parity._canonical import conformance, spec
from mujoco_models.shared.parity.fingerprint import fingerprint

STD = conformance.load_standard()
LIFTS = ("squat", "deadlift", "bench_press", "snatch", "clean_and_jerk")
ALL_EXERCISES = (
    "squat",
    "deadlift",
    "bench_press",
    "snatch",
    "clean_and_jerk",
    "gait",
    "sit_to_stand",
)

# sha256 of Repository_Management/shared_scripts/model_parity/
# biomech_parity_standard.json at v2.1.0 (Repository_Management#2079).
RM_STANDARD_V21_SHA256 = (
    "a0eb6c2bbbfeb70011bad62cf6d21a9245ffabce757c390633b516f542eb3ef8"
)


def _full_pose(exercise: str, pose: str) -> dict[str, float]:
    """Every canonical coordinate of a via-pose; unlisted ones are 0 (pose_frame)."""
    out = dict.fromkeys(conformance.expected_coordinates(STD), 0.0)
    out.update(spec.pose_coordinates(STD, exercise, pose))
    return out


def test_vendored_standard_is_repository_management_v21() -> None:
    assert conformance.standard_sha256() == RM_STANDARD_V21_SHA256
    assert STD["version"] == "2.1.0"


@pytest.mark.parametrize("exercise", ALL_EXERCISES)
def test_phase_names_and_timing_come_from_spec(exercise: str) -> None:
    objective = get_exercise_objective(exercise)
    want = spec.exercise_spec(STD, exercise)["phases"]
    assert [p.name for p in objective.phases] == [p["name"] for p in want]
    assert [p.fraction for p in objective.phases] == [p["t_frac"] for p in want]


@pytest.mark.parametrize("exercise", LIFTS)
def test_start_and_end_poses_are_spec_via_poses(exercise: str) -> None:
    objective = get_exercise_objective(exercise)
    assert objective.start_pose == _full_pose(exercise, "start")
    assert objective.end_pose == _full_pose(exercise, "lockout")


@pytest.mark.parametrize("exercise", LIFTS)
def test_anchored_phases_hold_their_via_pose(exercise: str) -> None:
    objective = get_exercise_objective(exercise)
    poses = spec.exercise_spec(STD, exercise)["via_poses"]
    assert set(spec_lifts.via_pose_phases(exercise)) == set(poses)
    for pose in poses:
        phase = spec.via_pose_phase(STD, exercise, pose)
        assert objective.get_phase(phase).target_joints == _full_pose(exercise, pose)


@pytest.mark.parametrize("exercise", LIFTS)
def test_start_anchors_the_first_phase(exercise: str) -> None:
    first = spec.phase_names(STD, exercise)[0]
    assert spec.via_pose_phase(STD, exercise, "start") == first


def test_pack_carries_no_local_pose_phase_table() -> None:
    """RM#2077: the pose-to-phase binding lives in the standard only."""
    assert not hasattr(spec_lifts, "VIA_POSE_PHASES")


def test_unanchored_phase_is_the_blend_of_its_anchors() -> None:
    """Deadlift first_pull (t 0.10) lies between setup (0) and lockout (0.65)."""
    objective = get_exercise_objective("deadlift")
    start, lockout = _full_pose("deadlift", "start"), _full_pose("deadlift", "lockout")
    w = 0.1 / 0.65
    got = objective.get_phase("first_pull").target_joints
    for name, value in got.items():
        assert value == pytest.approx(start[name] + w * (lockout[name] - start[name]))
    # After the last anchor the target holds the last anchored pose.
    assert objective.get_phase("descent").target_joints == lockout


def test_unknown_exercise_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown exercise"):
        spec_lifts.lift_objective(
            "curl", name="curl", bar_path_constraint="none", balance_mode="supine"
        )


@pytest.mark.parametrize("exercise", ALL_EXERCISES)
def test_fingerprint_reports_spec_phase_names(exercise: str) -> None:
    fp = fingerprint(exercise)
    assert fp["phase_names"] == spec.phase_names(STD, exercise)
    assert fp["phase_count"] == STD["exercises"][exercise]["phase_count"]


def test_bench_press_root_is_weld() -> None:
    assert fingerprint("bench_press")["root_joint"] == "weld"
