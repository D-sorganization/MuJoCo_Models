# SPDX-License-Identifier: MIT
"""Contact-aware inverse dynamics on seated standing models (issue #453).

With the soles seated at the contact equilibrium (issue #447) the foot-floor
contacts are active, so ``mj_inverse`` subtracts their generalized force from
``qfrc_inverse``.  The root (unactuated free joint) entries of
``qfrc_inverse`` are then the load the contacts do *not* explain.
:func:`inverse_dynamics_with_contacts` reports both parts explicitly, so the
contact load is never silently folded into the free joint.

Tolerance: ``_IDENTITY_ATOL_N`` is the round-off bound of an identity
evaluated in double precision.  The generalized forces are O(1e3) N; summing
O(nv) = 34 terms of that size accumulates about ``34 * 1e3 * 2.2e-16`` (about
1e-11 N).  1e-9 N keeps a 100x margin and equals the tolerance the existing
``inverse_dynamics`` == ``mj_inverse`` parity checks already use.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np
import pytest

from mujoco_models.dynamics import inverse_dynamics, inverse_dynamics_with_contacts
from mujoco_models.exercises.gait.gait_model import build_gait_model
from mujoco_models.exercises.sit_to_stand.sit_to_stand_model import (
    build_sit_to_stand_model,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

SEATED_BUILDERS: dict[str, Callable[[], str]] = {
    "gait": build_gait_model,
    "sit_to_stand": build_sit_to_stand_model,
}
_IDENTITY_ATOL_N = 1e-9


def _keyframe(exercise: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(SEATED_BUILDERS[exercise]())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("exercise", sorted(SEATED_BUILDERS))
def test_root_residual_vanishes_once_contact_wrench_is_counted(exercise: str) -> None:
    """At the forward-dynamics acceleration the contacts explain the root load.

    The keyframe's actual motion (``qacc`` from ``mj_forward``) is produced by
    gravity, servos and the foot contacts alone, so after the contact wrench
    is accounted for nothing is left for the free joint.
    """
    model, data = _keyframe(exercise)
    assert data.ncon > 0, "seated soles must be in active contact"

    report = inverse_dynamics_with_contacts(model, data.qpos, data.qvel, data.qacc)

    weight = float(model.body_subtreemass[model.body("pelvis").id]) * float(
        np.linalg.norm(model.opt.gravity)
    )
    assert report.contact_count == data.ncon
    assert report.root_contact[2] > 0.5 * weight, "contacts must carry the body"
    np.testing.assert_allclose(report.root_residual, 0.0, atol=_IDENTITY_ATOL_N)


@pytest.mark.parametrize("exercise", sorted(SEATED_BUILDERS))
def test_contact_and_residual_close_the_equations_of_motion(exercise: str) -> None:
    """``qfrc_inverse + qfrc_contact + qfrc_passive == M qacc + qfrc_bias``.

    These models have no equality constraint on the human tree, so the
    contact force is the only constraint load; the split is exact.
    """
    model, data = _keyframe(exercise)
    qacc = np.zeros(model.nv)

    report = inverse_dynamics_with_contacts(model, data.qpos, np.zeros(model.nv), qacc)

    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos
    mujoco.mj_inverse(model, probe)
    inertial = np.zeros(model.nv)
    mujoco.mj_mulM(model, probe, inertial, qacc)
    expected = inertial + probe.qfrc_bias - probe.qfrc_passive
    np.testing.assert_allclose(
        report.qfrc_inverse + report.qfrc_contact, expected, atol=_IDENTITY_ATOL_N
    )


@pytest.mark.parametrize("exercise", sorted(SEATED_BUILDERS))
def test_generalized_forces_are_unchanged_mj_inverse(exercise: str) -> None:
    """The report's ``qfrc_inverse`` is exactly what ``inverse_dynamics`` returns."""
    model, data = _keyframe(exercise)
    report = inverse_dynamics_with_contacts(model, data.qpos)
    np.testing.assert_allclose(
        report.qfrc_inverse,
        inverse_dynamics(model, data.qpos),
        atol=_IDENTITY_ATOL_N,
    )
