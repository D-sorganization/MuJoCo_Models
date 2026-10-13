# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Tests for per-hand bar wrench extraction and static hold (LIFT-4 / issue #444)."""

from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_models.exceptions import ValidationError
from mujoco_models.exercises.bench_press.bench_press_model import (
    build_bench_press_model,
)
from mujoco_models.exercises.clean_and_jerk.clean_and_jerk_model import (
    build_clean_and_jerk_model,
)
from mujoco_models.exercises.deadlift.deadlift_model import build_deadlift_model
from mujoco_models.exercises.gait.gait_model import build_gait_model
from mujoco_models.exercises.snatch.snatch_model import build_snatch_model
from mujoco_models.grip import (
    Wrench,
    export_grip_wrench_series,
    grip_wrenches,
)
from mujoco_models.shared.body.foot_seating import hold_hinges_rigid

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

GRIP_BUILDERS = {
    "deadlift": build_deadlift_model,
    "snatch": build_snatch_model,
    "clean_and_jerk": build_clean_and_jerk_model,
    "bench_press": build_bench_press_model,
}
_GRAVITY = 9.80665
_SETTLE_STEPS = 500


def _barbell_weight(model: mujoco.MjModel) -> float:
    """Weight (N) of the barbell shaft and sleeves."""
    bar_mass = 0.0
    for b in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if bname and "barbell" in bname:
            bar_mass += float(model.body_mass[b])
    return bar_mass * _GRAVITY


@pytest.mark.parametrize("exercise", sorted(GRIP_BUILDERS))
def test_static_hold_carries_bar_and_plate_weight(exercise: str) -> None:
    """Vertical sum of hand-on-bar forces equals bar-plus-plate weight within 1%."""
    model = mujoco.MjModel.from_xml_string(GRIP_BUILDERS[exercise]())
    hold_hinges_rigid(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(_SETTLE_STEPS):
        mujoco.mj_step(model, data)

    wrenches = grip_wrenches(model, data)
    assert wrenches["l"] is not None
    assert wrenches["r"] is not None

    total_vertical_force = wrenches["l"].force[2] + wrenches["r"].force[2]
    expected_weight = _barbell_weight(model)
    assert total_vertical_force == pytest.approx(expected_weight, rel=0.01)


@pytest.mark.parametrize("exercise", sorted(GRIP_BUILDERS))
def test_symmetric_grip_splits_force_evenly(exercise: str) -> None:
    """With a symmetric grip, left and right vertical forces agree closely."""
    model = mujoco.MjModel.from_xml_string(GRIP_BUILDERS[exercise]())
    hold_hinges_rigid(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(_SETTLE_STEPS):
        mujoco.mj_step(model, data)

    wrenches = grip_wrenches(model, data)
    assert wrenches["l"] is not None
    assert wrenches["r"] is not None

    diff = abs(wrenches["l"].force[2] - wrenches["r"].force[2])
    # Measured difference is < 1e-4 N on settled symmetric models
    assert diff < 1.0, f"{exercise}: L-R difference was {diff:.6f} N"


def test_unavailable_when_no_weld_active() -> None:
    """When a weld equality is disabled, its wrench returns None (never fabricated zero)."""
    model = mujoco.MjModel.from_xml_string(build_deadlift_model())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Disable left hand weld equality
    eq_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "barbell_to_hand_l")
    data.eq_active[eq_l] = 0
    mujoco.mj_forward(model, data)

    wrenches = grip_wrenches(model, data)
    assert wrenches["l"] is None, "inactive weld must return None"
    assert isinstance(wrenches["r"], Wrench)


def test_model_without_grip_welds_raises_validation_error() -> None:
    """Models with no grip welds (e.g. gait) raise ValidationError."""
    model = mujoco.MjModel.from_xml_string(build_gait_model())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(ValidationError, match="no grip welds"):
        grip_wrenches(model, data)


@pytest.mark.parametrize("exercise", sorted(GRIP_BUILDERS))
def test_export_grip_wrench_series_csv_and_npz(exercise: str, tmp_path: Path) -> None:
    """Exporting series produces valid CSV and NPZ with ADR-0052 column headers."""
    csv_file = tmp_path / f"{exercise}_grip.csv"
    npz_file = tmp_path / f"{exercise}_grip.npz"

    series_csv = export_grip_wrench_series(
        GRIP_BUILDERS[exercise](), steps=10, output_path=csv_file
    )
    assert csv_file.exists()
    assert "time_s" in series_csv
    assert "left_force_z_n" in series_csv
    assert "right_force_z_n" in series_csv

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "time_s" in header
        assert "left_force_z_n" in header
        assert "right_force_z_n" in header
        rows = list(reader)
        assert len(rows) == 10

    series_npz = export_grip_wrench_series(
        GRIP_BUILDERS[exercise](), steps=10, output_path=npz_file
    )
    assert npz_file.exists()
    assert "time_s" in series_npz
    loaded = np.load(npz_file)
    assert "time_s" in loaded
    assert "left_force_z_n" in loaded
    assert len(loaded["time_s"]) == 10
