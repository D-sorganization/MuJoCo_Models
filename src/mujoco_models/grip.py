# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Per-hand grip wrench extraction and static hold analysis (LIFT-4 / issue #444).

Sign convention (GCV-7 / ADR-0052):
    Every wrench represents the loading exerted by the hand ON THE BAR (or club),
    expressed in the world frame in SI units (N, N*m, m).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mujoco_models.exceptions import ValidationError

logger = logging.getLogger(__name__)

BAR_BODY_NAME = "barbell_shaft"
GRIP_WELD_PREFIX = "barbell_to_hand_"
GRIP_SIDES = ("l", "r")


@dataclass(frozen=True)
class Wrench:
    """World-frame wrench (force and torque) acting at a specific point.

    Attributes:
        force: (fx, fy, fz) 3-vector in Newtons, world frame.
            Sign convention: loading exerted by the hand ON THE BAR.
        torque: (tx, ty, tz) 3-vector in Newton-meters, world frame about point.
        point: (x, y, z) 3-vector in meters, world frame application point.
    """

    force: tuple[float, float, float]
    torque: tuple[float, float, float]
    point: tuple[float, float, float]


def dense_efc_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Dense ``(nefc, nv)`` constraint Jacobian for dense or sparse layouts.

    The layout is the model's (``mj_isSparse``).  The sparse row metadata
    (``efc_J_rownnz``/``rowadr``/``colind``) is not written in the dense
    layout and may hold stale memory, so it must not decide the layout.
    """
    nefc, nv = int(data.nefc), int(model.nv)
    if nefc == 0 or nv == 0:
        return np.zeros((nefc, nv), dtype=np.float64)
    if mujoco.mj_isSparse(model):
        dense = np.zeros((nefc, nv), dtype=np.float64)
        mujoco.mju_sparse2dense(
            dense,
            np.asarray(data.efc_J, dtype=np.float64),
            data.efc_J_rownnz,
            data.efc_J_rowadr,
            data.efc_J_colind,
        )
        return dense
    return np.asarray(data.efc_J[: nefc * nv], dtype=np.float64).reshape(nefc, nv)


def _compute_weld_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eq_id: int,
    bar_body_id: int,
) -> Wrench | None:
    """Extract world-frame wrench on the bar for one weld constraint."""
    if eq_id < 0 or int(data.eq_active[eq_id]) == 0:
        return None

    rows = np.flatnonzero(
        (data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        & (data.efc_id == eq_id)
    )
    if rows.size == 0:
        return None

    body1 = int(model.eq_obj1id[eq_id])
    body2 = int(model.eq_obj2id[eq_id])

    j_dense = dense_efc_jacobian(model, data)
    qfrc = j_dense[rows].T @ data.efc_force[rows]

    point = np.array(data.xpos[bar_body_id], dtype=np.float64)
    j1 = np.zeros((6, model.nv), dtype=np.float64)
    j2 = np.zeros((6, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, j1[:3], j1[3:], point, body1)
    mujoco.mj_jac(model, data, j2[:3], j2[3:], point, body2)
    j_diff = j2 - j1

    if np.linalg.matrix_rank(j_diff) < 6:
        raise ValidationError(f"weld equality {eq_id} Jacobian is rank deficient")

    wrench_on_obj2, *_ = np.linalg.lstsq(j_diff.T, qfrc, rcond=None)
    wrench = wrench_on_obj2 if bar_body_id == body2 else -wrench_on_obj2

    return Wrench(
        force=(float(wrench[0]), float(wrench[1]), float(wrench[2])),
        torque=(float(wrench[3]), float(wrench[4]), float(wrench[5])),
        point=(float(point[0]), float(point[1]), float(point[2])),
    )


def grip_wrenches(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, Wrench | None]:
    """Return per-hand world-frame wrench exerted by each hand on the barbell.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled MuJoCo model.
    data : mujoco.MjData
        Current simulation data (must have executed ``mj_forward`` or ``mj_step``).

    Returns
    -------
    dict[str, Wrench | None]
        Mapping ``{"l": Wrench | None, "r": Wrench | None}``. A hand with no
        active weld returns ``None`` (unavailable is never zero).

    Raises
    -------
    ValidationError
        If the model has no grip weld equality constraints.
    """
    bar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BAR_BODY_NAME)
    eq_ids = {
        side: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{GRIP_WELD_PREFIX}{side}"
        )
        for side in GRIP_SIDES
    }

    if all(eid < 0 for eid in eq_ids.values()):
        raise ValidationError("model has no grip welds")

    result: dict[str, Wrench | None] = {}
    for side, eq_id in eq_ids.items():
        if eq_id < 0:
            result[side] = None
        else:
            result[side] = _compute_weld_wrench(model, data, eq_id, bar_id)

    return result


def export_grip_wrench_series(
    model_or_xml: str | mujoco.MjModel,
    steps: int = 20,
    output_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Simulate initial hold and export per-hand wrench time series per ADR-0052.

    Parameters
    ----------
    model_or_xml : str or mujoco.MjModel
        MJCF XML string or compiled MjModel.
    steps : int, default 20
        Number of forward simulation steps to record.
    output_path : str, Path or None, default None
        Optional file path to write (.csv or .npz).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of 1D arrays for each time and force/torque/point column.
    """
    if isinstance(model_or_xml, str):
        model = mujoco.MjModel.from_xml_string(model_or_xml)
    else:
        model = model_or_xml

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    columns: dict[str, list[float]] = {
        "time_s": [],
        "left_force_x_n": [],
        "left_force_y_n": [],
        "left_force_z_n": [],
        "right_force_x_n": [],
        "right_force_y_n": [],
        "right_force_z_n": [],
        "left_torque_x_nm": [],
        "left_torque_y_nm": [],
        "left_torque_z_nm": [],
        "right_torque_x_nm": [],
        "right_torque_y_nm": [],
        "right_torque_z_nm": [],
        "left_point_x_m": [],
        "left_point_y_m": [],
        "left_point_z_m": [],
        "right_point_x_m": [],
        "right_point_y_m": [],
        "right_point_z_m": [],
    }

    nan3 = (np.nan, np.nan, np.nan)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        columns["time_s"].append(float(data.time))
        w = grip_wrenches(model, data)
        wl, wr = w.get("l"), w.get("r")

        fl = wl.force if wl is not None else nan3
        tl = wl.torque if wl is not None else nan3
        pl = wl.point if wl is not None else nan3

        fr = wr.force if wr is not None else nan3
        tr = wr.torque if wr is not None else nan3
        pr = wr.point if wr is not None else nan3

        columns["left_force_x_n"].append(fl[0])
        columns["left_force_y_n"].append(fl[1])
        columns["left_force_z_n"].append(fl[2])
        columns["right_force_x_n"].append(fr[0])
        columns["right_force_y_n"].append(fr[1])
        columns["right_force_z_n"].append(fr[2])

        columns["left_torque_x_nm"].append(tl[0])
        columns["left_torque_y_nm"].append(tl[1])
        columns["left_torque_z_nm"].append(tl[2])
        columns["right_torque_x_nm"].append(tr[0])
        columns["right_torque_y_nm"].append(tr[1])
        columns["right_torque_z_nm"].append(tr[2])

        columns["left_point_x_m"].append(pl[0])
        columns["left_point_y_m"].append(pl[1])
        columns["left_point_z_m"].append(pl[2])
        columns["right_point_x_m"].append(pr[0])
        columns["right_point_y_m"].append(pr[1])
        columns["right_point_z_m"].append(pr[2])

    arrays = {k: np.array(v, dtype=np.float64) for k, v in columns.items()}

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".csv":
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = list(columns.keys())
                writer.writerow(header)
                for i in range(steps):
                    writer.writerow([columns[k][i] for k in header])
        elif p.suffix == ".npz":
            np.savez(p, **arrays)  # type: ignore[arg-type]
        else:
            raise ValidationError(
                f"unsupported format {p.suffix!r}, expected .csv or .npz"
            )

    return arrays
