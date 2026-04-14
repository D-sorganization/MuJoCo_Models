"""Unit tests for the chair-building helpers on SitToStandModelBuilder.

Covers the helpers extracted during the A-N Refresh 2026-04-14 decomposition
of ``_post_worldbody_hook`` (issue #129).  TDD: these assertions pin down
the expected XML shape of each piece individually, so future refactors
can edit a single helper with confidence.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_models.exercises.sit_to_stand.sit_to_stand_model import (
    SitToStandModelBuilder,
)


def test_create_chair_body_attaches_named_body() -> None:
    """``_create_chair_body`` returns a body named ``chair`` parented to worldbody."""
    worldbody = ET.Element("worldbody")
    chair = SitToStandModelBuilder._create_chair_body(worldbody)
    assert chair.tag == "body"
    assert chair.get("name") == "chair"
    assert chair in list(worldbody)


def test_add_chair_seat_geom_is_a_box_geom_with_contact_flags() -> None:
    """Seat geom is a contact-enabled box named ``chair_seat``."""
    chair = ET.Element("body")
    SitToStandModelBuilder._add_chair_seat_geom(chair)
    geoms = chair.findall("geom")
    assert len(geoms) == 1
    seat = geoms[0]
    assert seat.get("name") == "chair_seat"
    assert seat.get("type") == "box"
    assert seat.get("contype") == "1"
    assert seat.get("conaffinity") == "1"


def test_add_chair_back_geom_positions_behind_seat() -> None:
    """Chair back has negative Y position (behind the pelvis origin)."""
    chair = ET.Element("body")
    SitToStandModelBuilder._add_chair_back_geom(chair)
    back = chair.find("geom")
    assert back is not None
    assert back.get("name") == "chair_back"
    pos = back.get("pos", "")
    # pos is "0 <negY> <posZ>" -- second token should start with minus sign
    y_token = pos.split()[1]
    assert y_token.startswith("-"), f"expected negative Y in back pos, got {pos!r}"


def test_weld_chair_to_world_creates_fixed_weld() -> None:
    """``_weld_chair_to_world`` emits an identity-weld for the chair body."""
    equality = ET.Element("equality")
    SitToStandModelBuilder._weld_chair_to_world(equality)
    welds = equality.findall("weld")
    assert len(welds) == 1
    weld = welds[0]
    assert weld.get("name") == "chair_to_world"
    assert weld.get("body1") == "chair"
    assert weld.get("relpose") == "0 0 0 1 0 0 0"


def test_post_worldbody_hook_assembles_full_chair() -> None:
    """The public hook still produces: chair body + 2 geoms + 1 weld."""
    builder = SitToStandModelBuilder()
    worldbody = ET.Element("worldbody")
    equality = ET.Element("equality")
    builder._post_worldbody_hook(worldbody, equality)

    chairs = worldbody.findall("body[@name='chair']")
    assert len(chairs) == 1
    geoms = chairs[0].findall("geom")
    assert {g.get("name") for g in geoms} == {"chair_seat", "chair_back"}
    assert len(equality.findall("weld[@name='chair_to_world']")) == 1
