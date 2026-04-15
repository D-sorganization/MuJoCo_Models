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
    _add_limb_joints,
    _add_single_foot_contact_geom,
    _build_limb_body,
    _create_limb_side_body,
    _demote_visual_geom_group,
    _LimbSideSpec,
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


def test_build_limb_body_writes_capsule_body() -> None:
    """The capsule-body helper should set the body shell and inertial data."""
    parent = ET.Element("body", name="parent")
    child = _build_limb_body(
        parent,
        body_name="forearm_l",
        pos=(0.1, 0.2, 0.3),
        mass=3.5,
        inertia=(1.0, 2.0, 3.0),
        radius=0.04,
        length=0.5,
    )

    assert child.get("name") == "forearm_l"
    assert child.get("pos") == "0.100000 0.200000 0.300000"
    inertial = child.find("inertial")
    assert inertial is not None
    assert inertial.get("mass") == "3.500000"
    assert inertial.get("diaginertia") == "1.000000 2.000000 3.000000"
    geom = child.find("geom")
    assert geom is not None
    assert geom.get("type") == "capsule"
    assert geom.get("size") == "0.040000 0.250000"


def test_add_limb_joints_writes_primary_and_extra_joints() -> None:
    """The joint helper should attach the flexion joint and side-specific extras."""
    body = ET.Element("body", name="hand_r")

    _add_limb_joints(
        body,
        coord_prefix="wrist",
        side="r",
        range_min=-1.0,
        range_max=1.0,
        extra_joints=[
            ("deviate", (0, 0, 1), -0.5, 0.5),
        ],
    )

    joints = body.findall("joint")
    assert [joint.get("name") for joint in joints] == [
        "wrist_r_flex",
        "wrist_r_deviate",
    ]
    assert joints[0].get("type") == "hinge"
    assert joints[0].get("axis") == "1.000000 0.000000 0.000000"
    assert joints[0].get("range") == "-1.0000 1.0000"
    assert joints[1].get("axis") == "0.000000 0.000000 1.000000"
    assert joints[1].get("range") == "-0.5000 0.5000"


def test_create_limb_side_body_composes_body_and_joints() -> None:
    """The orchestration helper should keep the body and joint wiring together."""
    parent = ET.Element("body", name="upper_arm_l")
    spec = _LimbSideSpec(
        parent_el=parent,
        body_name="forearm_l",
        pos=(0.0, 0.0, -0.3),
        mass=2.0,
        inertia=(0.1, 0.2, 0.3),
        radius=0.03,
        length=0.4,
        coord_prefix="elbow",
        side="l",
        range_min=0.0,
        range_max=2.618,
        extra_joints=None,
    )

    child = _create_limb_side_body(spec)

    assert child.get("name") == "forearm_l"
    assert child.find("geom") is not None
    assert [joint.get("name") for joint in child.findall("joint")] == ["elbow_l_flex"]
