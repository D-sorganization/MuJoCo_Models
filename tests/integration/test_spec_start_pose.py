# SPDX-License-Identifier: MIT
"""Every barbell lift starts at the standard's ``start`` via-pose (LIFT-3).

UpstreamDrift#11743, MuJoCo_Models#407/#409: the generated model's keyframe
must hold the spec start pose -- joint coordinates, pelvis root and bar
position -- with the hands closed on the bar, the feet flat on the floor, the
spec masses and the combined centre of mass over mid-foot.  Every check is
measured in the real MuJoCo engine.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.shared.parity._canonical import conformance, spec, topology
from mujoco_models.shared.parity.standard import STANDARD

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

LIFTS = ("squat", "deadlift", "bench_press", "snatch", "clean_and_jerk")
STANDING = ("squat", "deadlift", "snatch", "clean_and_jerk")
PULLS = ("deadlift", "snatch", "clean_and_jerk")
_TOL = STANDARD["tolerances"]
_ANGLE_TOL = _TOL["limit_abs_rad"]
_POSITION_TOL = 1e-4  # m; the pose is placed exactly, so only rounding remains
_PALM_TOL = 0.005  # m; hand-bar closure (UpstreamDrift#11743)
_SOLE_TOL = 0.002  # m; sole corner above/below the floor (soft-contact seating)
_COM_TOL = _TOL["cross_engine_position_abs_m"]
_ANTHRO = STANDARD["anthropometrics"]


def _start(exercise: str) -> dict:
    return spec.exercise_spec(STANDARD, exercise)["via_poses"]["start"]


def _keyframe(exercise: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY[exercise]().build())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture(scope="module", params=LIFTS)
def lift(request: pytest.FixtureRequest) -> tuple[str, mujoco.MjModel, mujoco.MjData]:
    model, data = _keyframe(request.param)
    return request.param, model, data


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return np.array(data.xpos[model.body(name).id])


def test_start_coordinates_equal_the_spec(lift) -> None:
    """Keyframe angle (qpos - ref) of every canonical coordinate is the spec's."""
    exercise, model, _data = lift
    want = dict.fromkeys(conformance.expected_coordinates(STANDARD), 0.0)
    want.update(spec.pose_coordinates(STANDARD, exercise, "start"))
    for name, angle in want.items():
        adr = model.jnt_qposadr[model.joint(name).id]
        got = model.key_qpos[0][adr] - model.qpos0[adr]
        assert got == pytest.approx(angle, abs=_ANGLE_TOL), f"{exercise}: {name}"


def test_every_joint_ref_is_zero(lift) -> None:
    """``ref`` never poses a segment (#407); a lift keeps every hinge ref at 0
    so the start angle lives in the keyframe, inside the joint range (#409)."""
    exercise, model, _data = lift
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            assert model.qpos0[model.jnt_qposadr[j]] == 0.0, (
                f"{exercise}: {model.joint(j).name} ref is not 0"
            )


def test_pelvis_root_equals_the_spec(lift) -> None:
    exercise, model, data = lift
    root = _start(exercise)["root"]
    pelvis = model.body("pelvis").id
    np.testing.assert_allclose(
        data.xpos[pelvis], root["position_m"], atol=_POSITION_TOL, err_msg=exercise
    )
    want = np.array(topology.rpy_matrix(root["orientation_rpy_rad"]))
    np.testing.assert_allclose(
        data.xmat[pelvis].reshape(3, 3), want, atol=_ANGLE_TOL, err_msg=exercise
    )


def test_bar_starts_at_the_spec_bar_position(lift) -> None:
    exercise, model, data = lift
    np.testing.assert_allclose(
        _body_pos(model, data, "barbell_shaft"),
        _start(exercise)["bar_position_m"],
        atol=_POSITION_TOL,
        err_msg=exercise,
    )


@pytest.mark.parametrize("exercise", PULLS)
def test_pull_bar_starts_on_the_floor_at_plate_radius(exercise: str) -> None:
    model, data = _keyframe(exercise)
    radius = spec.exercise_spec(STANDARD, exercise)["load"]["plate_diameter_m"] / 2
    assert _body_pos(model, data, "barbell_shaft")[2] == pytest.approx(
        radius, abs=_POSITION_TOL
    )


def _sole_corners(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> list:
    gid = model.geom(f"foot_{side}_contact").id
    rot = data.geom_xmat[gid].reshape(3, 3)
    hx, hy, hz = model.geom_size[gid]
    return [
        data.geom_xpos[gid] + rot @ np.array([sx * hx, sy * hy, -hz])
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]


@pytest.mark.parametrize("side", ("l", "r"))
def test_feet_flat_on_the_floor(lift, side: str) -> None:
    exercise, model, data = lift
    gid = model.geom(f"foot_{side}_contact").id
    sole_normal = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    tilt = math.acos(min(1.0, float(sole_normal[2])))
    assert tilt <= _TOL["axis_angle_abs_rad"], f"{exercise}: sole tilt {tilt:.4f}"
    for corner in _sole_corners(model, data, side):
        assert abs(corner[2]) <= _SOLE_TOL, f"{exercise}: sole corner z {corner[2]}"


@pytest.mark.parametrize("side", ("l", "r"))
def test_palm_centre_closes_on_the_bar(lift, side: str) -> None:
    """The palm centre (grip_point) lies on the bar axis at half the grip width."""
    exercise, model, data = lift
    hand = model.body(f"hand_{side}").id
    hand_length = _ANTHRO["segments"]["hand"]["length_frac"] * _ANTHRO["height_m"]
    palm = data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ np.array(
        [0.0, 0.0, -0.5 * hand_length]
    )
    shaft = model.body("barbell_shaft").id
    axis = data.xmat[shaft].reshape(3, 3)[:, 1]
    offset = palm - data.xpos[shaft]
    along = float(offset @ axis)
    off_axis = float(np.linalg.norm(offset - along * axis))
    half_width = spec.exercise_spec(STANDARD, exercise)["grip"]["width_m"] / 2
    assert off_axis <= _PALM_TOL, f"{exercise}: palm {off_axis * 1e3:.1f} mm off bar"
    assert abs(along) == pytest.approx(half_width, abs=_PALM_TOL), exercise


def _human_bodies(model: mujoco.MjModel) -> list[int]:
    segments = conformance.expected_segments(STANDARD)
    return [b for b in range(model.nbody) if model.body(b).name in segments]


def test_human_and_segment_masses_equal_the_spec(lift) -> None:
    exercise, model, _data = lift
    body_mass = _ANTHRO["body_mass_kg"]
    rel = _TOL["mass_rel"]
    total = sum(model.body_mass[b] for b in _human_bodies(model))
    assert total == pytest.approx(body_mass, rel=rel), exercise
    for b in _human_bodies(model):
        name = model.body(b).name
        seg = name.rsplit("_", 1)[0] if name.endswith(("_l", "_r")) else name
        frac = _ANTHRO["segments"][seg]["mass_frac"]
        assert model.body_mass[b] == pytest.approx(frac * body_mass, rel=rel), name


def test_barbell_mass_is_the_spec_load(lift) -> None:
    exercise, model, _data = lift
    load = spec.exercise_spec(STANDARD, exercise)["load"]
    want = STANDARD["barbell"]["mens"]["bar_mass_kg"] + 2 * load["plates_per_side_kg"]
    got = sum(
        model.body_mass[b]
        for b in range(model.nbody)
        if model.body(b).name.startswith("barbell")
    )
    assert got == pytest.approx(want, rel=_TOL["mass_rel"]), exercise


@pytest.mark.parametrize("exercise", STANDING)
def test_combined_com_is_over_mid_foot(exercise: str) -> None:
    """Body + bar centre of mass sits horizontally over mid-foot (x = y = 0)."""
    model, data = _keyframe(exercise)
    com = data.subtree_com[0]  # world subtree: lifter + barbell
    assert abs(com[0]) <= _COM_TOL, f"{exercise}: CoM x {com[0] * 1e3:.1f} mm"
    assert abs(com[1]) <= _COM_TOL, f"{exercise}: CoM y {com[1] * 1e3:.1f} mm"
