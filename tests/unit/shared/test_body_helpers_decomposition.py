"""Unit tests for the foot-contact helpers in body_helpers.py.

Covers the decomposition of ``add_foot_contact_geoms`` (issue #129,
A-N Refresh 2026-04-14) into:

* ``_demote_visual_geom_group`` -- moves any existing geom into group "0".
* ``_add_single_foot_contact_geom`` -- appends the contact box for one side.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_models.shared.body import body_helpers
from mujoco_models.shared.body.body_helpers import (
    _add_single_foot_contact_geom,
    _demote_visual_geom_group,
    add_foot_contact_geoms,
)


def test_demote_visual_geom_group_sets_group_zero() -> None:
    """An existing geom should be marked as group 0 (visual)."""
    foot = ET.Element("body", name="foot_l")
    existing = ET.SubElement(foot, "geom", type="capsule")
    _demote_visual_geom_group(foot)
    assert existing.get("group") == "0"


def test_demote_visual_geom_group_is_noop_without_geom() -> None:
    """Missing visual geom is not an error."""
    foot = ET.Element("body", name="foot_r")
    _demote_visual_geom_group(foot)  # must not raise


def test_add_single_foot_contact_geom_writes_all_attributes() -> None:
    """Contact geom carries contact flags, friction, and the group=1 marker."""
    foot = ET.Element("body", name="foot_l")
    _add_single_foot_contact_geom(foot, "l")
    geoms = foot.findall("geom")
    assert len(geoms) == 1
    g = geoms[0]
    assert g.get("name") == "foot_l_contact"
    assert g.get("type") == "box"
    assert g.get("size") == body_helpers._FOOT_CONTACT_SIZE
    assert g.get("pos") == body_helpers._FOOT_CONTACT_POS
    assert g.get("friction") == body_helpers._FOOT_CONTACT_FRICTION
    assert g.get("rgba") == body_helpers._FOOT_CONTACT_RGBA
    assert g.get("contype") == "1"
    assert g.get("conaffinity") == "1"
    assert g.get("condim") == "3"
    assert g.get("group") == "1"


def test_add_foot_contact_geoms_handles_both_feet() -> None:
    """Both ``foot_l`` and ``foot_r`` gain a contact box and visual demotion."""
    foot_l = ET.Element("body", name="foot_l")
    ET.SubElement(foot_l, "geom", type="capsule", name="foot_l_visual")
    foot_r = ET.Element("body", name="foot_r")
    ET.SubElement(foot_r, "geom", type="capsule", name="foot_r_visual")

    add_foot_contact_geoms({"foot_l": foot_l, "foot_r": foot_r})

    for side, foot in (("l", foot_l), ("r", foot_r)):
        names = {g.get("name") for g in foot.findall("geom")}
        assert f"foot_{side}_contact" in names
        assert f"foot_{side}_visual" in names
        visual = foot.find("geom[@name='foot_" + side + "_visual']")
        assert visual is not None
        assert visual.get("group") == "0"


def test_add_foot_contact_geoms_skips_missing_sides() -> None:
    """A dict without the expected keys is tolerated silently (no raise)."""
    add_foot_contact_geoms({})  # must not raise
