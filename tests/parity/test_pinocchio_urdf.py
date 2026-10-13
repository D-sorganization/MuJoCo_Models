# SPDX-License-Identifier: MIT
"""Cross-engine conformance: Pinocchio_Models URDF in MuJoCo (issue #411).

Loads Pinocchio_Models URDFs into real MuJoCo via MjModel and verifies that
the resulting model fingerprint agrees with native MuJoCo models within
documented parity tolerances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mujoco_models.model_pack import list_exercises
from mujoco_models.shared.parity._canonical import conformance
from mujoco_models.shared.parity.fingerprint import fingerprint as native_fp
from mujoco_models.shared.parity.urdf_import import (
    EXPECTED_SEGMENTS,
    load_pinocchio_urdf,
    pinocchio_urdf_fingerprint,
    resolve_pinocchio_urdf,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = REPO_ROOT / "src" / "mujoco_models" / "shared" / "parity"
URDF_LEDGER = PARITY_DIR / "urdf_import_divergences.json"
STD = conformance.load_standard()
EXERCISES = list_exercises()


def _get_urdf_or_skip(exercise: str) -> str:
    """Return Pinocchio URDF string or skip the test cleanly."""
    urdf = resolve_pinocchio_urdf(exercise, repo_root=REPO_ROOT)
    if urdf is None:
        pytest.skip(f"Pinocchio_Models URDF for {exercise!r} is unavailable")
    return urdf


def _load_urdf_ledger() -> dict[str, Any]:
    """Load documented cross-engine URDF divergences."""
    if not URDF_LEDGER.is_file():
        return {}
    return json.loads(URDF_LEDGER.read_text(encoding="utf-8")).get("divergences", {})


@pytest.mark.parametrize("exercise", EXERCISES)
def test_pinocchio_urdf_loads_in_real_mujoco(exercise: str) -> None:
    """Pinocchio URDF must load cleanly in real MuJoCo without error."""
    urdf = _get_urdf_or_skip(exercise)
    model = load_pinocchio_urdf(urdf)
    assert model.nbody >= 1 + 15
    assert model.njnt >= 28


@pytest.mark.parametrize("exercise", EXERCISES)
def test_pinocchio_urdf_joint_count(exercise: str) -> None:
    """Pinocchio URDF model in MuJoCo must have exactly 28 human hinge joints."""
    urdf = _get_urdf_or_skip(exercise)
    model = load_pinocchio_urdf(urdf)
    fp = pinocchio_urdf_fingerprint(model)
    assert fp["joint_count"] == 28


def test_pinocchio_urdf_total_mass_matches_native_gait() -> None:
    """Total body mass in unencumbered gait matches native MuJoCo."""
    urdf = _get_urdf_or_skip("gait")
    model = load_pinocchio_urdf(urdf)
    fp = pinocchio_urdf_fingerprint(model)
    nat = native_fp("gait")
    native_total = float(nat["body_mass_kg"])
    tol = float(STD["tolerances"]["mass_rel"])
    assert fp["total_mass_kg"] == pytest.approx(native_total, rel=tol)


@pytest.mark.parametrize("exercise", EXERCISES)
def test_pinocchio_urdf_human_segment_masses(exercise: str) -> None:
    """Canonical human segment masses in Pinocchio URDF match native MuJoCo."""
    urdf = _get_urdf_or_skip(exercise)
    model = load_pinocchio_urdf(urdf)
    fp = pinocchio_urdf_fingerprint(model)
    nat = native_fp(exercise)
    ledger = _load_urdf_ledger()
    tol = float(STD["tolerances"]["mass_rel"])

    for seg in EXPECTED_SEGMENTS:
        nat_mass = float(nat["segments"][seg]["mass_kg"])
        urdf_mass = fp["segment_masses_kg"].get(seg, 0.0)

        divergence_key = f"segment_mass.{exercise}.{seg}"
        if divergence_key in ledger:
            continue

        assert urdf_mass == pytest.approx(nat_mass, rel=tol), (
            f"Segment {seg} mass mismatch in {exercise}: "
            f"native={nat_mass} urdf={urdf_mass}"
        )


@pytest.mark.parametrize("exercise", EXERCISES)
def test_pinocchio_urdf_segment_origins_rel_pelvis(exercise: str) -> None:
    """Neutral-pose segment origins relative to pelvis match native MuJoCo."""
    urdf = _get_urdf_or_skip(exercise)
    model = load_pinocchio_urdf(urdf)
    fp = pinocchio_urdf_fingerprint(model)
    nat = native_fp(exercise)
    ledger = _load_urdf_ledger()

    nat_neutral = nat["segment_origins_neutral_m"]
    pelvis_origin = nat_neutral["pelvis"]
    pos_tol = float(STD["tolerances"]["cross_engine_position_abs_m"])

    for seg in EXPECTED_SEGMENTS:
        if seg not in fp["segment_origins_rel_pelvis_m"]:
            continue
        divergence_key = f"segment_origin.{exercise}.{seg}"
        if divergence_key in ledger:
            continue

        urdf_rel = fp["segment_origins_rel_pelvis_m"][seg]
        nat_rel = [nat_neutral[seg][i] - pelvis_origin[i] for i in range(3)]
        for axis_i, (u_val, n_val) in enumerate(zip(urdf_rel, nat_rel, strict=True)):
            assert abs(u_val - n_val) <= pos_tol, (
                f"Origin mismatch on {seg} axis {axis_i} in {exercise}: "
                f"native_rel={n_val:.4f} urdf_rel={u_val:.4f}"
            )


def test_pinocchio_urdf_ledger_no_stale_entries() -> None:
    """Every divergence in the URDF ledger corresponds to an actual difference."""
    ledger = _load_urdf_ledger()
    if not ledger:
        return

    stale: list[str] = []
    for key in ledger:
        parts = key.split(".")
        if len(parts) != 3:
            continue
        kind, exercise, segment = parts
        urdf = resolve_pinocchio_urdf(exercise, repo_root=REPO_ROOT)
        if urdf is None:
            continue
        model = load_pinocchio_urdf(urdf)
        fp = pinocchio_urdf_fingerprint(model)
        nat = native_fp(exercise)

        if kind == "segment_mass":
            nat_mass = float(nat["segments"][segment]["mass_kg"])
            urdf_mass = fp["segment_masses_kg"].get(segment, 0.0)
            if (
                abs(urdf_mass - nat_mass)
                <= float(STD["tolerances"]["mass_rel"]) * nat_mass
            ):
                stale.append(key)
        elif kind == "segment_origin":
            nat_neutral = nat["segment_origins_neutral_m"]
            pelvis_origin = nat_neutral["pelvis"]
            pos_tol = float(STD["tolerances"]["cross_engine_position_abs_m"])
            urdf_rel = fp["segment_origins_rel_pelvis_m"].get(segment, [0.0, 0.0, 0.0])
            nat_rel = [nat_neutral[segment][i] - pelvis_origin[i] for i in range(3)]
            if all(
                abs(u - n) <= pos_tol for u, n in zip(urdf_rel, nat_rel, strict=True)
            ):
                stale.append(key)

    assert not stale, f"Stale ledger entries found: {stale}"


def test_unavailable_urdf_skips_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a URDF is not found, resolution returns None and callers skip."""
    res = resolve_pinocchio_urdf("nonexistent_exercise_xyz")
    assert res is None
    monkeypatch.setattr(
        "tests.parity.test_pinocchio_urdf.resolve_pinocchio_urdf",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(pytest.skip.Exception):
        _get_urdf_or_skip("squat")
