"""Real-MuJoCo segment origins match the standard's reference FK (RM#2011)."""

from __future__ import annotations

import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.shared.parity._canonical import conformance, topology
from mujoco_models.shared.parity.fingerprint import fingerprint

STD = conformance.load_standard()


@pytest.mark.parametrize("exercise", sorted(EXERCISE_REGISTRY))
def test_origins_match_the_reference_at_every_test_pose(exercise: str) -> None:
    fp = fingerprint(exercise)
    assert set(fp["segment_origins_test_poses_m"]) == set(topology.standard_poses(STD))
    assert topology.check_origins(fp, STD) == []
