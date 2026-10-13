# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Unit tests for the dense constraint Jacobian helper (issue #453).

The oracle is MuJoCo's own layout-aware ``mj_mulJacTVec``: for any vector
``v`` over the constraint rows, ``J^T v`` must match it in both the dense and
the sparse Jacobian layouts.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_models.exercises.gait.gait_model import GaitModelBuilder
from mujoco_models.grip import dense_efc_jacobian

_LAYOUTS = {
    "dense": mujoco.mjtJacobian.mjJAC_DENSE,
    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
}


def _gait_with_contacts(layout: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(GaitModelBuilder().build())
    model.opt.jacobian = _LAYOUTS[layout]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert data.nefc > 0
    return model, data


def _jacobian_transpose_times(
    model: mujoco.MjModel, data: mujoco.MjData, vec: np.ndarray
) -> np.ndarray:
    result = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, data, result, vec)
    return result


@pytest.mark.unit
@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_dense_efc_jacobian_matches_mujoco_in_both_layouts(layout: str) -> None:
    model, data = _gait_with_contacts(layout)
    vec = np.random.default_rng(453).standard_normal(int(data.nefc))
    jacobian = dense_efc_jacobian(model, data)
    assert jacobian.shape == (data.nefc, model.nv)
    np.testing.assert_allclose(
        jacobian.T @ vec, _jacobian_transpose_times(model, data, vec), atol=1e-9
    )


@pytest.mark.unit
def test_dense_layout_ignores_stale_sparse_row_metadata() -> None:
    """A dense Jacobian is read as dense whatever ``efc_J_rownnz`` holds.

    MuJoCo leaves the sparse row metadata unset in the dense layout (it held
    arena garbage on Linux with MuJoCo 3.15, which segfaulted the helper or
    zeroed the contact load).  Small poison values keep the old code path
    from reading out of bounds while still exposing it.
    """
    model, data = _gait_with_contacts("dense")
    vec = np.random.default_rng(2026).standard_normal(int(data.nefc))
    expected = _jacobian_transpose_times(model, data, vec)
    data.efc_J_rownnz[: data.nefc] = 1
    data.efc_J_rowadr[: data.nefc] = 0
    data.efc_J_colind[: data.nefc] = 0
    np.testing.assert_allclose(
        dense_efc_jacobian(model, data).T @ vec, expected, atol=1e-9
    )
