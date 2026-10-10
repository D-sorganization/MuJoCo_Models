# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Unit tests for the public forward kinematics API (issue #402)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.exceptions import ValidationError
from mujoco_models.exercises.squat.squat_model import SquatModelBuilder
from mujoco_models.kinematics import forward_kinematics


@pytest.fixture
def squat_model_and_q() -> tuple[mujoco.MjModel, np.ndarray]:
    model = mujoco.MjModel.from_xml_string(SquatModelBuilder().build())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    return model, data.qpos.copy()


def _perturbed(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(402)
    q2 = q + 0.05 * rng.standard_normal(model.nq)
    mujoco.mj_normalizeQuat(model, q2)
    return q2


@pytest.mark.unit
def test_export_from_top_level_package() -> None:
    import mujoco_models

    assert callable(mujoco_models.forward_kinematics)
    assert mujoco_models.SegmentPose is not None


@pytest.mark.unit
@pytest.mark.parametrize("perturb", [False, True])
def test_matches_real_mujoco_every_segment(
    squat_model_and_q: tuple[mujoco.MjModel, np.ndarray], perturb: bool
) -> None:
    model, q = squat_model_and_q
    if perturb:
        q = _perturbed(model, q)
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)

    poses = forward_kinematics("squat", q)

    assert len(poses) == model.nbody
    for i in range(model.nbody):
        pose = poses[model.body(i).name]
        np.testing.assert_allclose(pose.position, data.xpos[i], atol=1e-9)
        np.testing.assert_allclose(
            pose.orientation, data.xmat[i].reshape(3, 3), atol=1e-9
        )


@pytest.mark.unit
def test_accepts_builder_and_model(
    squat_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    model, q = squat_model_and_q
    by_name = forward_kinematics("squat", q)
    for exercise in (SquatModelBuilder(), model):
        other = forward_kinematics(exercise, q)
        assert other.keys() == by_name.keys()
        for name, pose in by_name.items():
            np.testing.assert_allclose(other[name].position, pose.position)


@pytest.mark.unit
def test_invalid_inputs_raise_value_error(
    squat_model_and_q: tuple[mujoco.MjModel, np.ndarray],
) -> None:
    _, q = squat_model_and_q
    with pytest.raises(ValueError, match="Unknown exercise"):
        forward_kinematics("nope", q)
    with pytest.raises(ValueError, match="exercise must be"):
        forward_kinematics(123, q)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        forward_kinematics("squat", np.zeros(3))
    for bad in (np.nan, np.inf):
        q_bad = q.copy()
        q_bad[0] = bad
        with pytest.raises(ValidationError):
            forward_kinematics("squat", q_bad)
