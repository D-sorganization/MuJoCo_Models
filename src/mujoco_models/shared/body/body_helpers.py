"""Helper functions for body model assembly.

Extracted from body_model.py to keep modules under the 300-line budget.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from mujoco_models.shared.utils.geometry import capsule_inertia
from mujoco_models.shared.utils.mjcf_helpers import add_body, add_hinge_joint

logger = logging.getLogger(__name__)

# Type alias for extra hinge-joint specs: (suffix, axis, range_min, range_max)
_ExtraJoints = list[tuple[str, tuple[float, float, float], float, float]]


# Foot contact-geometry constants.  Centralizing here removes magic numbers
# from the sub-element calls and keeps the geometry documented in one place.
# Half-sizes correspond to a 0.26 m x 0.10 m x 0.02 m contact box.
_FOOT_CONTACT_SIZE = "0.13 0.05 0.01"
_FOOT_CONTACT_POS = "0.04 0 -0.02"  # slightly forward and at bottom of foot
_FOOT_CONTACT_FRICTION = "1.0 0.005 0.0001"  # tangent, torsion, rolling
_FOOT_CONTACT_RGBA = "0.8 0.6 0.4 0.3"  # semi-transparent for visualization


@dataclass(frozen=True)
class _LimbSideSpec:
    """Inputs needed to build one side of a bilateral limb."""

    parent_el: ET.Element
    body_name: str
    pos: tuple[float, float, float]
    mass: float
    inertia: tuple[float, float, float]
    radius: float
    length: float
    coord_prefix: str
    side: str
    range_min: float
    range_max: float
    extra_joints: _ExtraJoints | None = None


def _demote_visual_geom_group(foot_body: ET.Element) -> None:
    """Move the existing (visual) foot geom into group 0 if present.

    Separating visual and contact geoms by ``group`` lets MuJoCo render
    without the contact box drawn on top of the visual capsule.
    """
    visual_geom = foot_body.find("geom")
    if visual_geom is not None:
        visual_geom.set("group", "0")


def _add_single_foot_contact_geom(foot_body: ET.Element, side: str) -> None:
    """Attach the standard sole-contact box geom to a single foot body.

    Args:
        foot_body: The ``<body name="foot_{side}">`` element.
        side: One of ``"l"`` or ``"r"``.
    """
    contact_geom = ET.SubElement(foot_body, "geom")
    contact_geom.set("name", f"foot_{side}_contact")
    contact_geom.set("type", "box")
    contact_geom.set("size", _FOOT_CONTACT_SIZE)
    contact_geom.set("pos", _FOOT_CONTACT_POS)
    contact_geom.set("contype", "1")
    contact_geom.set("conaffinity", "1")
    contact_geom.set("condim", "3")
    contact_geom.set("friction", _FOOT_CONTACT_FRICTION)
    contact_geom.set("group", "1")
    contact_geom.set("rgba", _FOOT_CONTACT_RGBA)


def add_foot_contact_geoms(bodies: dict[str, ET.Element]) -> None:
    """Add box collision geometry to foot segments for ground contact.

    Each foot gets a box geom representing the sole contact area:
    ~0.26 m long x 0.10 m wide x 0.02 m thick, positioned at the
    bottom of the foot segment.

    Contact properties: contype=1, conaffinity=1, condim=3,
    friction="1.0 0.005 0.0001" (tangent, torsion, rolling).
    Group=1 separates contact geoms from visual geoms (group=0).

    Missing sides (e.g. unilateral rigs) are silently skipped; callers
    that require a specific side should check ``bodies`` themselves.
    """
    for side in ("l", "r"):
        foot_body = bodies.get(f"foot_{side}")
        if foot_body is None:
            continue
        _demote_visual_geom_group(foot_body)
        _add_single_foot_contact_geom(foot_body, side)

    logger.debug("Added contact sole geometry to foot segments")


def _create_limb_side_body(spec: _LimbSideSpec) -> ET.Element:
    """Create body and hinge joints for one side of a bilateral limb."""
    child_body = _build_limb_body(
        spec.parent_el,
        body_name=spec.body_name,
        pos=spec.pos,
        mass=spec.mass,
        inertia=spec.inertia,
        radius=spec.radius,
        length=spec.length,
    )
    _add_limb_joints(
        child_body,
        coord_prefix=spec.coord_prefix,
        side=spec.side,
        range_min=spec.range_min,
        range_max=spec.range_max,
        extra_joints=spec.extra_joints,
    )
    return child_body


def _build_limb_body(
    parent_el: ET.Element,
    *,
    body_name: str,
    pos: tuple[float, float, float],
    mass: float,
    inertia: tuple[float, float, float],
    radius: float,
    length: float,
) -> ET.Element:
    """Create the capsule body for one side of a bilateral limb."""
    return add_body(
        parent_el,
        name=body_name,
        pos=pos,
        mass=mass,
        inertia_diag=inertia,
        geom_type="capsule",
        geom_size=(radius, length / 2.0),
        geom_rgba="0.8 0.6 0.4 1",
    )


def _add_limb_joints(
    child_body: ET.Element,
    *,
    coord_prefix: str,
    side: str,
    range_min: float,
    range_max: float,
    extra_joints: _ExtraJoints | None = None,
) -> None:
    """Attach the flexion joint and any optional side-specific joints."""
    add_hinge_joint(
        child_body,
        name=f"{coord_prefix}_{side}_flex",
        axis=(1, 0, 0),
        range_min=range_min,
        range_max=range_max,
    )
    if extra_joints:
        for suffix, axis, ex_min, ex_max in extra_joints:
            add_hinge_joint(
                child_body,
                name=f"{coord_prefix}_{side}_{suffix}",
                axis=axis,
                range_min=ex_min,
                range_max=ex_max,
            )


def add_bilateral_limb(
    parent_bodies: dict[str, ET.Element],
    mass: float,
    length: float,
    radius: float,
    *,
    seg_name: str,
    parent_name: str,
    parent_offset_z: float,
    parent_lateral_x: float,
    coord_prefix: str,
    range_min: float,
    range_max: float,
    extra_joints: _ExtraJoints | None = None,
) -> dict[str, ET.Element]:
    """Add left and right limb segments with hinge joints.

    MuJoCo convention: Z-up, so vertical offsets use Z coordinate.
    Hinge joints rotate about the X-axis (medio-lateral) by default
    for sagittal-plane flexion/extension.

    Parameters
    ----------
    extra_joints : list of (suffix, axis, range_min, range_max) or None
        Additional hinge joints to add after the primary flexion joint.
        Each entry creates ``{coord_prefix}_{side}_{suffix}`` on the
        given axis with the given range limits.
    """
    inertia = capsule_inertia(mass, radius, length)
    parent_is_bilateral = f"{parent_name}_l" in parent_bodies
    created: dict[str, ET.Element] = {}
    for side, sign in [("l", -1.0), ("r", 1.0)]:
        key = f"{parent_name}_{side}" if parent_is_bilateral else parent_name
        side_spec = _LimbSideSpec(
            parent_el=parent_bodies[key],
            body_name=f"{seg_name}_{side}",
            pos=(sign * parent_lateral_x, 0, parent_offset_z),
            mass=mass,
            inertia=inertia,
            radius=radius,
            length=length,
            coord_prefix=coord_prefix,
            side=side,
            range_min=range_min,
            range_max=range_max,
            extra_joints=extra_joints,
        )
        created[side_spec.body_name] = _create_limb_side_body(side_spec)
    return created
