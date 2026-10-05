# SPDX-License-Identifier: MIT
"""Pure-Python checks of the collision mask classes (issue #394)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_models.shared import contact_masks as cm


def _hits(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return bool((a[0] & b[1]) or (b[0] & a[1]))


def test_mask_rule_matrix() -> None:
    human = (cm.HUMAN_CONTYPE, cm.HUMAN_CONAFFINITY)
    equip = (cm.EQUIPMENT_CONTYPE, cm.EQUIPMENT_CONAFFINITY)
    floor = (cm.FLOOR_CONTYPE, cm.FLOOR_CONAFFINITY)
    foot = (cm.FOOT_CONTACT_CONTYPE, cm.FOOT_CONTACT_CONAFFINITY)
    assert not _hits(human, human)
    assert _hits(human, equip)
    assert _hits(equip, floor)
    assert not _hits(human, floor)
    assert _hits(foot, floor)
    assert not _hits(foot, human)


def test_apply_masks_sets_attributes_and_returns_element() -> None:
    geom = ET.Element("geom")
    assert cm.apply_masks(geom, cm.EQUIPMENT_MASKS) is geom
    assert geom.get("contype") == "1"
    assert geom.get("conaffinity") == "3"
