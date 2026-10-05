# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Collision bitmask classes shared by every MJCF builder (issue #394).

MuJoCo collides two geoms when
``(g1.contype & g2.conaffinity) or (g2.contype & g1.conaffinity)``.
The classes below give:

* human vs human: no contact (``2 & 0 == 0``)
* human vs equipment (bench, chair, barbell): contact (``2 & 3``)
* equipment vs floor / equipment: contact (``1 & 1``, ``1 & 3``)
* human body segments vs floor: no contact (``2 & 1 == 0`` and ``1 & 0 == 0``)
* foot contact boxes vs floor: contact (``1 & 1``); vs human geoms: none
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping

# Human body geoms: belong to class 2, accept nothing.
HUMAN_CONTYPE: int = 2
HUMAN_CONAFFINITY: int = 0

# Equipment (bench, chair, barbell): class 1, accepts classes 1 and 2.
EQUIPMENT_CONTYPE: int = 1
EQUIPMENT_CONAFFINITY: int = 3

# Floor and foot contact boxes: class 1, accept class 1 only.
FLOOR_CONTYPE: int = 1
FLOOR_CONAFFINITY: int = 1
FOOT_CONTACT_CONTYPE: int = FLOOR_CONTYPE
FOOT_CONTACT_CONAFFINITY: int = FLOOR_CONAFFINITY

HUMAN_MASKS: dict[str, str] = {
    "contype": str(HUMAN_CONTYPE),
    "conaffinity": str(HUMAN_CONAFFINITY),
}
EQUIPMENT_MASKS: dict[str, str] = {
    "contype": str(EQUIPMENT_CONTYPE),
    "conaffinity": str(EQUIPMENT_CONAFFINITY),
}
FLOOR_MASKS: dict[str, str] = {
    "contype": str(FLOOR_CONTYPE),
    "conaffinity": str(FLOOR_CONAFFINITY),
}
FOOT_CONTACT_MASKS: dict[str, str] = {
    "contype": str(FOOT_CONTACT_CONTYPE),
    "conaffinity": str(FOOT_CONTACT_CONAFFINITY),
}


def apply_masks(geom: ET.Element, masks: Mapping[str, str]) -> ET.Element:
    """Set the ``contype``/``conaffinity`` attributes of ``geom`` and return it."""
    for key, value in masks.items():
        geom.set(key, value)
    return geom
