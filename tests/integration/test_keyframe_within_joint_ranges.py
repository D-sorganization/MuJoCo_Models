# SPDX-License-Identifier: MIT
"""Every exercise's initial keyframe qpos and joint ``ref`` lie within joint
ranges (issues #438, #409).

A keyframe value outside its own hinge's ``<joint range="...">`` would be
silently clamped by MuJoCo's compiler, desyncing the documented start pose
from what the engine actually simulates.
"""

from __future__ import annotations

import mujoco
import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.model_pack import list_exercises

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

_EXERCISES = list_exercises()

# Known, tracked exceptions: ratcheted like the parity divergence ledger --
# a listed (exercise, joint) pair that stops violating its range is stale and
# must be removed, so this allowlist cannot silently grow stale in the other
# direction either.
#
# No known violations: the barbell lifts start at the spec start via-pose
# (UpstreamDrift#11743), every hinge ``ref`` is 0, and the grip welds are
# measured from the posed keyframe, so no start angle is clamped.
_KNOWN_RANGE_VIOLATIONS: dict[str, frozenset[str]] = {}


@pytest.mark.parametrize("exercise", sorted(_EXERCISES))
def test_keyframe_qpos_within_joint_ranges(exercise: str) -> None:
    """Precondition: the exercise emits at least one limited hinge joint,
    so the test is not vacuously true."""
    xml = EXERCISE_REGISTRY[exercise]().build()
    model = mujoco.MjModel.from_xml_string(xml)
    known = _KNOWN_RANGE_VIOLATIONS.get(exercise, frozenset())
    violated: set[str] = set()
    checked = 0
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[j]:
            continue
        qadr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for value in (model.key_qpos[0][qadr], model.qpos0[qadr]):  # key, ref
            if not (lo <= value <= hi):
                violated.add(name)
        checked += 1
    assert checked > 0, f"{exercise}: no limited hinge joints found"

    unexpected = violated - known
    assert not unexpected, f"{exercise}: unexpected out-of-range joints {unexpected}"
    stale = known - violated
    assert not stale, (
        f"{exercise}: known range violation(s) {stale} no longer reproduce; "
        "remove from _KNOWN_RANGE_VIOLATIONS"
    )
