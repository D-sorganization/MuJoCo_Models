# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Unit tests for public dynamics API (issue #405)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.dynamics import inverse_dynamics, inverse_dynamics_with_contacts
from mujoco_models.exceptions import ValidationError
from mujoco_models.exercises.gait.gait_model import GaitModelBuilder


@pytest.fixture
def gait_model_and_q() -> tuple[mujoco.MjModel, np.ndarray]:
    builder = GaitModelBuilder()
    model = mujoco.MjModel.from_xml_string(builder.build())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    return model, data.qpos.copy()


@pytest.mark.unit
def test_export_from_top_level_package() -> None:
    import mujoco_models

    assert hasattr(mujoco_models, "inverse_dynamics")
    assert callable(mujoco_models.inverse_dynamics)


@pytest.mark.unit
def test_inverse_dynamics_with_exercise_name(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    model, q = gait_model_and_q
    qvel = np.zeros(model.nv)
    qacc = np.zeros(model.nv)

    torques = inverse_dynamics("gait", q, qvel, qacc)
    assert isinstance(torques, np.ndarray)
    assert torques.shape == (model.nv,)

    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = qvel
    data.qacc[:] = qacc
    mujoco.mj_inverse(model, data)
    np.testing.assert_allclose(torques, data.qfrc_inverse, atol=1e-9)


@pytest.mark.unit
def test_inverse_dynamics_with_builder_and_model(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    model, q = gait_model_and_q
    qvel = np.zeros(model.nv)
    qacc = np.zeros(model.nv)

    torques_builder = inverse_dynamics(GaitModelBuilder(), q, qvel, qacc)
    torques_model = inverse_dynamics(model, q, qvel, qacc)

    np.testing.assert_allclose(torques_builder, torques_model, atol=1e-9)


@pytest.mark.unit
def test_inverse_dynamics_default_qvel_and_qacc(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    model, q = gait_model_and_q
    torques = inverse_dynamics("gait", q)

    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_inverse(model, data)
    np.testing.assert_allclose(torques, data.qfrc_inverse, atol=1e-9)


@pytest.mark.unit
def test_inverse_dynamics_unknown_exercise_raises(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    _, q = gait_model_and_q
    with pytest.raises(ValidationError, match="Unknown exercise 'nonexistent'"):
        inverse_dynamics("nonexistent", q)


@pytest.mark.unit
def test_inverse_dynamics_invalid_exercise_type_raises(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    _, q = gait_model_and_q
    with pytest.raises(ValidationError, match="exercise must be an exercise name"):
        inverse_dynamics(12345, q)  # type: ignore[arg-type]


@pytest.mark.unit
def test_inverse_dynamics_wrong_q_shape_raises(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    _, q = gait_model_and_q
    with pytest.raises(ValidationError, match="q must have shape"):
        inverse_dynamics("gait", q[:5])


@pytest.mark.unit
def test_inverse_dynamics_wrong_qvel_shape_raises(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    _, q = gait_model_and_q
    with pytest.raises(ValidationError, match="qvel must have shape"):
        inverse_dynamics("gait", q, qvel=[1.0, 2.0])


@pytest.mark.unit
def test_inverse_dynamics_wrong_qacc_shape_raises(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    _, q = gait_model_and_q
    with pytest.raises(ValidationError, match="qacc must have shape"):
        inverse_dynamics("gait", q, qacc=[1.0, 2.0])


@pytest.mark.unit
def test_inverse_dynamics_non_finite_inputs_raise(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    model, q = gait_model_and_q
    q_nan = q.copy()
    q_nan[0] = float("nan")
    with pytest.raises(ValidationError, match="q contains non-finite"):
        inverse_dynamics("gait", q_nan)

    qvel_nan = np.zeros(model.nv)
    qvel_nan[0] = float("inf")
    with pytest.raises(ValidationError, match="qvel contains non-finite"):
        inverse_dynamics("gait", q, qvel=qvel_nan)

    qacc_nan = np.zeros(model.nv)
    qacc_nan[0] = float("nan")
    with pytest.raises(ValidationError, match="qacc contains non-finite"):
        inverse_dynamics("gait", q, qacc=qacc_nan)


_NO_FREE_JOINT_XML = (
    '<mujoco><worldbody><body name="arm"><joint name="hinge" type="hinge"/>'
    '<geom type="capsule" size="0.05 0.2"/></body></worldbody></mujoco>'
)


@pytest.mark.unit
def test_contact_inverse_dynamics_requires_a_free_root_joint() -> None:
    """The root residual is defined only for a floating-base model."""
    model = mujoco.MjModel.from_xml_string(_NO_FREE_JOINT_XML)
    with pytest.raises(ValidationError, match="free root joint"):
        inverse_dynamics_with_contacts(model, np.zeros(model.nq))


@pytest.mark.unit
def test_contact_inverse_dynamics_report_shapes(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    model, q = gait_model_and_q
    report = inverse_dynamics_with_contacts(model, q)
    assert report.qfrc_inverse.shape == (model.nv,)
    assert report.qfrc_contact.shape == (model.nv,)
    assert report.root_residual.shape == (6,)
    assert report.root_contact.shape == (6,)


@pytest.mark.unit
def test_contact_inverse_dynamics_validates_inputs(
    gait_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    _, q = gait_model_and_q
    with pytest.raises(ValidationError, match="qacc must have shape"):
        inverse_dynamics_with_contacts("gait", q, qacc=[1.0, 2.0])
