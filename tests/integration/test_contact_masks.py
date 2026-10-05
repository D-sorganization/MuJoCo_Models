# SPDX-License-Identifier: MIT
"""Human/equipment contact masks in the real MuJoCo engine (issue #394)."""

from __future__ import annotations

import itertools

import mujoco
import pytest

from mujoco_models.exercises.bench_press.bench_press_model import (
    build_bench_press_model,
)
from mujoco_models.exercises.sit_to_stand.sit_to_stand_model import (
    _CHAIR_CENTER_Y,
    build_sit_to_stand_model,
)
from mujoco_models.exercises.squat.squat_model import build_squat_model

pytestmark = [pytest.mark.integration, pytest.mark.requires_mujoco]

BUILDERS = {
    "bench_press": (build_bench_press_model, "bench_contact"),
    "sit_to_stand": (build_sit_to_stand_model, "chair_seat"),
    "squat": (build_squat_model, "barbell_shaft_geom"),
}


def _collides(model: mujoco.MjModel, a: int, b: int) -> bool:
    """MuJoCo's documented pair rule for two geoms."""
    return bool(
        (model.geom_contype[a] & model.geom_conaffinity[b])
        or (model.geom_contype[b] & model.geom_conaffinity[a])
    )


def _classify(model: mujoco.MjModel) -> tuple[list[int], list[int], list[int]]:
    """Return (human, equipment, foot-contact) geom ids; floor id separate."""
    human: list[int] = []
    equipment: list[int] = []
    feet: list[int] = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]
        )
        if name == "ground":
            continue
        if name.endswith("_contact") and name.startswith("foot_"):
            feet.append(gid)
        elif body and body.startswith(("barbell", "bench", "chair")):
            equipment.append(gid)
        else:
            human.append(gid)
    return human, equipment, feet


@pytest.mark.parametrize("exercise", sorted(BUILDERS))
def test_human_collides_with_equipment_not_itself(exercise: str) -> None:
    builder, equip_geom = BUILDERS[exercise]
    model = mujoco.MjModel.from_xml_string(builder())
    human, equipment, feet = _classify(model)
    assert human and equipment and feet
    equip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, equip_geom)
    assert equip_id in equipment
    # (a) every human geom pairs with the equipment geom.
    assert all(_collides(model, h, equip_id) for h in human)
    # (b) no two human geoms collide.
    assert not any(_collides(model, a, b) for a, b in itertools.combinations(human, 2))
    # (c) feet collide with the floor; other human geoms do not.
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert all(_collides(model, f, ground) for f in feet)
    assert not any(_collides(model, h, ground) for h in human)
    # Equipment rests on the floor.
    assert all(_collides(model, e, ground) for e in equipment)


def test_human_leg_in_chair_generates_contact() -> None:
    """Moving the lifter into the chair yields real human-vs-chair contacts."""
    model = mujoco.MjModel.from_xml_string(build_sit_to_stand_model())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    chair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chair")
    human, _, _ = _classify(model)

    def human_chair_contacts() -> int:
        return sum(
            1
            for c in data.contact[: data.ncon]
            if chair in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
            and (c.geom1 in human or c.geom2 in human)
        )

    # Standing clear of the chair: no contact.
    assert human_chair_contacts() == 0
    # Slide the root (first freejoint, y is qpos[1]) back into the seat.
    data.qpos[1] += _CHAIR_CENTER_Y
    mujoco.mj_forward(model, data)
    assert human_chair_contacts() > 0
