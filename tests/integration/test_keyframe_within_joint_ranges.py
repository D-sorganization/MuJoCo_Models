# SPDX-License-Identifier: MIT
"""Every barbell exercise's initial keyframe qpos lies within joint ranges
(issue #438).

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

_BARBELL_EXERCISES = [
    name for name in list_exercises() if EXERCISE_REGISTRY[name]().uses_barbell
]

# Known, tracked exceptions: ratcheted like the parity divergence ledger --
# a listed (exercise, joint) pair that stops violating its range is stale and
# must be removed, so this allowlist cannot silently grow stale in the other
# direction either.
#
# wrist_{l,r}_deviate on snatch: ``_grip_pose_offsets`` sizes the wrist
# counter-rotation to exactly cancel the shoulder abduction's tilt on the
# hand (keeps the grip weld residual near zero, #408/#437). The snatch's
# documented 0.60 m grip needs ~48 deg of shoulder abduction (#438), which
# exceeds the wrist's own +30 deg deviate limit. Clamping the counter-
# rotation to the wrist's own ROM instead reintroduces a ~188 mm weld
# residual (measured with the real engine), so it is not fixed here.
# Tracked as MuJoCo_Models#440.
_KNOWN_RANGE_VIOLATIONS: dict[str, frozenset[str]] = {
    "snatch": frozenset({"wrist_l_deviate", "wrist_r_deviate"}),
}


@pytest.mark.parametrize("exercise", sorted(_BARBELL_EXERCISES))
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
        value = model.key_qpos[0][qadr]
        lo, hi = model.jnt_range[j]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
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
