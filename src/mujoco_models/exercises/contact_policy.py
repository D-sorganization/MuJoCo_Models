# SPDX-License-Identifier: MIT
"""Contact and servo policy shared by every exercise model (issue #390)."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_models.shared.parity.standard import GROUND_FRICTION

# Self-collision policy (issues #390, #394): collision bitmasks live in
# ``mujoco_models.shared.contact_masks``.  Human body geoms (contype=2,
# conaffinity=0) never collide with each other, yet collide with equipment
# (bench, chair, barbell: contype=1, conaffinity=3).  The floor and the
# dedicated foot contact boxes use contype=conaffinity=1, so only the feet
# touch the floor.  This removes every non-physical self contact
# (torso-thigh, thigh-thigh, hand-pelvis, ...) that previously produced forces
# up to 64 kN.  The explicit <exclude> pairs below stay as
# defence in depth (e.g. if a downstream caller re-enables body collisions).
#
# Servo gains: position actuators hold the keyframe pose (ctrl = keyframe
# qpos).  kp is finite and every actuator is force limited.
# Ground sliding friction = bundle static coefficient (single-coefficient engine).
_GROUND_FRICTION: str = f"{GROUND_FRICTION['static']:g} 0.005 0.0001"
_SERVO_KP: float = 1000.0
_SERVO_KV: float = 100.0
_SERVO_FORCERANGE: float = 500.0

# Adjacent body segment pairs to exclude from self-collision.
# Central pairs (no side suffix needed):
_CENTRAL_EXCLUSION_PAIRS: list[tuple[str, str]] = [
    ("pelvis", "torso"),
    ("torso", "head"),
]

# Bilateral pairs (will be expanded with _l and _r suffixes):
_BILATERAL_EXCLUSION_PAIRS: list[tuple[str, str]] = [
    ("pelvis", "thigh"),
    ("torso", "upper_arm"),
    ("upper_arm", "forearm"),
    ("forearm", "hand"),
    ("thigh", "shank"),
    ("shank", "foot"),
]

# Whole-body pairs whose geoms must never collide even though they are not
# adjacent in the kinematic tree.
_SIDE_EXCLUSION_PAIRS: list[tuple[str, str]] = [("foot_l", "foot_r")]


def _add_contact_exclusions(contact: ET.Element) -> None:  # noqa: C901
    """Add <exclude> elements to prevent self-collision between adjacent segments.

    Central body pairs (pelvis-torso, torso-head) are excluded once.
    Bilateral pairs are excluded for both left and right sides.
    """
    for body1, body2 in _CENTRAL_EXCLUSION_PAIRS:
        ET.SubElement(
            contact,
            "exclude",
            name=f"exclude_{body1}_{body2}",
            body1=body1,
            body2=body2,
        )

    for body1, body2 in _BILATERAL_EXCLUSION_PAIRS:
        for side in ("l", "r"):
            b1 = f"{body1}_{side}" if body1 not in ("pelvis", "torso") else body1
            b2 = f"{body2}_{side}"
            ET.SubElement(
                contact,
                "exclude",
                name=f"exclude_{b1}_{b2}",
                body1=b1,
                body2=b2,
            )

    for b1, b2 in _SIDE_EXCLUSION_PAIRS:
        ET.SubElement(contact, "exclude", name=f"exclude_{b1}_{b2}", body1=b1, body2=b2)


_BARBELL_BODY_PREFIX = "barbell"


def _add_barbell_exclusions(contact: ET.Element, worldbody: ET.Element) -> None:
    """Exclude every barbell body from the lifter and from the other bar parts.

    The shaft is held by the grip weld and the sleeves by the sleeve welds, so
    bar-versus-lifter and shaft-versus-sleeve contacts are non-physical
    constraint fights (hundreds of kN at the start pose, issue #427).  Bar
    contact with the floor and with the bench or chair stays enabled.

    Postcondition: one ``<exclude>`` per intra-bar pair and per bar/lifter-body
    pair; none when the model has no barbell.
    """
    bar_names = [
        name
        for body in worldbody.findall("body")
        if (name := body.get("name", "")).startswith(_BARBELL_BODY_PREFIX)
    ]
    if not bar_names:
        return
    lifter_names = [
        name
        for top in worldbody.findall("body")
        if top.find("freejoint") is not None
        and not (top.get("name", "")).startswith(_BARBELL_BODY_PREFIX)
        for body in top.iter("body")
        if (name := body.get("name", ""))
    ]
    pairs = [(a, b) for i, a in enumerate(bar_names) for b in bar_names[i + 1 :]] + [
        (bar, body) for bar in bar_names for body in lifter_names
    ]
    for body1, body2 in pairs:
        ET.SubElement(
            contact,
            "exclude",
            name=f"exclude_{body1}_{body2}",
            body1=body1,
            body2=body2,
        )
