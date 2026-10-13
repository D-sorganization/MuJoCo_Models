# SPDX-License-Identifier: MIT
"""MJCF root, worldbody, contacts, and state section construction scaffolding."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable

from mujoco_models.exercises.contact_policy import (
    _GROUND_FRICTION,
    _SERVO_FORCERANGE,
    _SERVO_KP,
    _SERVO_KV,
    _add_barbell_exclusions,
    _add_contact_exclusions,
)
from mujoco_models.shared.contact_masks import (
    FLOOR_MASKS,
    HUMAN_MASKS,
    apply_masks,
)


def set_ref_by_name_map(
    worldbody: ET.Element,
    refs_by_name_fragment: dict[str, float],
) -> None:
    """Set joint ``ref`` attributes by matching name fragments.

    Iterates all hinge joints in *worldbody*. For each joint whose
    name contains a key from *refs_by_name_fragment*, sets ``ref`` to
    the corresponding value. The first matching fragment wins.

    Args:
        worldbody: The ``<worldbody>`` element of the MJCF model.
        refs_by_name_fragment: Mapping from name substring to ref angle
            (radians). E.g. ``{"hip_flex": 1.396, "knee": -1.047}``.
    """
    for joint in worldbody.iter("joint"):
        if joint.get("type", "hinge") != "hinge":
            continue
        name = joint.get("name", "")
        for fragment, ref in refs_by_name_fragment.items():
            if fragment in name:
                joint.set("ref", str(ref))
                break


def create_root_element(
    exercise_name: str, gravity: tuple[float, float, float]
) -> ET.Element:
    """Create the ``<mujoco>`` root with option, compiler, and defaults."""
    root = ET.Element("mujoco", model=exercise_name)

    ET.SubElement(
        root,
        "option",
        gravity=f"{gravity[0]:.6f} {gravity[1]:.6f} {gravity[2]:.6f}",
        timestep="0.001",
        integrator="implicit",
        cone="elliptic",
        solver="Newton",
        tolerance="1e-8",
    )

    ET.SubElement(root, "compiler", angle="radian")

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="5.0", armature="0.1")
    # Human geoms never collide with each other; equipment opts in via masks.
    apply_masks(ET.SubElement(default, "geom", condim="3"), HUMAN_MASKS)

    return root


def create_worldbody(root: ET.Element) -> ET.Element:
    """Create worldbody with ground plane."""
    worldbody = ET.SubElement(root, "worldbody")
    ground = ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        size="5 5 0.1",
        rgba="0.9 0.9 0.9 1",
        condim="3",
        friction=_GROUND_FRICTION,
    )
    apply_masks(ground, FLOOR_MASKS)
    return worldbody


def add_contact_section(root: ET.Element) -> ET.Element:
    """Create the contact section and populate adjacent-segment exclusions."""
    contact = ET.SubElement(root, "contact")
    _add_contact_exclusions(contact)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        _add_barbell_exclusions(contact, worldbody)
    return contact


def add_state_sections(
    root: ET.Element,
    worldbody: ET.Element,
    exercise_name: str,
    keyframe_offsets: dict[str, float],
    set_initial_pose_fn: Callable[[ET.Element], None] | None = None,
) -> None:
    """Apply initial pose metadata, actuators, sensors, and keyframe data."""
    if set_initial_pose_fn is not None:
        set_initial_pose_fn(worldbody)

    actuator = ET.SubElement(root, "actuator")
    sensor = ET.SubElement(root, "sensor")

    qpos_values: list[str] = []
    ctrl_values: list[str] = []

    # Single document-order (depth-first) traversal: MuJoCo orders qpos by
    # the kinematic tree, so every freejoint (pelvis AND each barbell body)
    # contributes its 7 values exactly where its body appears.
    for el in worldbody.iter():
        tag = el.tag
        if tag == "joint":
            name = el.get("name", "")
            ref = el.get("ref", "0")
            if name in keyframe_offsets:
                ref = repr(float(ref) + keyframe_offsets[name])
            if name:
                ET.SubElement(
                    actuator,
                    "position",
                    name=f"act_{name}",
                    joint=name,
                    kp=f"{_SERVO_KP:g}",
                    kv=f"{_SERVO_KV:g}",
                    forcerange=f"{-_SERVO_FORCERANGE:g} {_SERVO_FORCERANGE:g}",
                )
                ET.SubElement(sensor, "jointpos", name=f"pos_{name}", joint=name)
                ctrl_values.append(ref)
            qpos_values.append(ref)
        elif tag == "body" and el.find("freejoint") is not None:
            qpos_values.extend(
                el.get("pos", "0 0 0").split() + el.get("quat", "1 0 0 0").split()
            )

    if qpos_values:
        keyframe = ET.SubElement(root, "keyframe")
        # ctrl = pose so the position servos hold the keyframe instead of
        # springing every joint to 0.
        ET.SubElement(
            keyframe,
            "key",
            name=f"{exercise_name}_start",
            qpos=" ".join(qpos_values),
            ctrl=" ".join(ctrl_values),
        )
