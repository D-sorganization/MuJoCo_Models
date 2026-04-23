"""MJCF XML generation helpers for MuJoCo models.

DRY: All bodies, joints, and geometry share these formatting functions
so that MJCF structure is defined in exactly one place.

MuJoCo MJCF reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def vec3_str(x: float, y: float, z: float) -> str:
    """Format three floats as a space-separated string for MJCF XML."""
    return f"{x:.6f} {y:.6f} {z:.6f}"


def diag_inertia_str(ixx: float, iyy: float, izz: float) -> str:
    """Format diagonal inertia as a space-separated string for MJCF."""
    return f"{ixx:.6f} {iyy:.6f} {izz:.6f}"


def _build_geom_attrs(
    name: str,
    geom_type: str,
    geom_rgba: str,
    geom_size: tuple[float, ...] | None,
    geom_euler: tuple[float, float, float] | None,
) -> dict[str, str]:
    """Build the attribute dict for an MJCF ``<geom>`` element."""
    attrs: dict[str, str] = {
        "name": f"{name}_geom",
        "type": geom_type,
        "rgba": geom_rgba,
    }

    # ⚡ Bolt Optimization:
    # Hardcode string formatting for common tuple lengths (1, 2, 3)
    # instead of using generator expressions and string joins.
    # This avoids generator allocation and reduces execution time.
    if geom_size is not None:
        l_size = len(geom_size)
        if l_size == 2:
            attrs["size"] = f"{geom_size[0]:.6f} {geom_size[1]:.6f}"
        elif l_size == 1:
            attrs["size"] = f"{geom_size[0]:.6f}"
        elif l_size == 3:
            attrs["size"] = f"{geom_size[0]:.6f} {geom_size[1]:.6f} {geom_size[2]:.6f}"
        else:
            attrs["size"] = " ".join(f"{s:.6f}" for s in geom_size)

    if geom_euler is not None:
        if len(geom_euler) == 3:
            attrs["euler"] = f"{geom_euler[0]:.6f} {geom_euler[1]:.6f} {geom_euler[2]:.6f}"
        else:
            attrs["euler"] = " ".join(f"{s:.6f}" for s in geom_euler)

    return attrs


def add_body(
    parent: ET.Element,
    *,
    name: str,
    pos: tuple[float, float, float],
    mass: float,
    inertia_diag: tuple[float, float, float],
    geom_type: str = "cylinder",
    geom_size: tuple[float, ...] | None = None,
    geom_rgba: str = "0.8 0.6 0.4 1",
    geom_euler: tuple[float, float, float] | None = None,
) -> ET.Element:
    """Create an MJCF ``<body>`` with ``<inertial>`` and ``<geom>`` children.

    Appends the body to *parent* (typically ``<worldbody>`` or another body).
    *pos* is the (x, y, z) offset from the parent origin.
    *inertia_diag* is (Ixx, Iyy, Izz) in kg·m².
    *geom_type* is a MuJoCo type string (cylinder, box, capsule, sphere).
    *geom_size* and *geom_euler* are forwarded verbatim to the ``<geom>``.
    Returns the created ``<body>`` element.
    """
    body = ET.SubElement(parent, "body", name=name, pos=vec3_str(*pos))
    ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass=f"{mass:.6f}",
        diaginertia=diag_inertia_str(*inertia_diag),
    )
    ET.SubElement(
        body,
        "geom",
        attrib=_build_geom_attrs(name, geom_type, geom_rgba, geom_size, geom_euler),
    )
    return body


def add_hinge_joint(
    body: ET.Element,
    *,
    name: str,
    axis: tuple[float, float, float] = (1, 0, 0),
    range_min: float = -1.5708,
    range_max: float = 1.5708,
) -> ET.Element:
    """Add a ``<joint type="hinge">`` to an MJCF body.

    Parameters
    ----------
    body : ET.Element
        The body element to add the joint to.
    name : str
        Joint name.
    axis : tuple
        Rotation axis (unit vector).
    range_min, range_max : float
        Joint limits in radians.

    Returns
    -------
    ET.Element
        The created ``<joint>`` element.
    """
    joint = ET.SubElement(
        body,
        "joint",
        name=name,
        type="hinge",
        axis=vec3_str(*axis),
    )
    joint.set("range", f"{range_min:.4f} {range_max:.4f}")
    return joint


def add_free_joint(body: ET.Element, *, name: str) -> ET.Element:
    """Add a ``<freejoint>`` to an MJCF body (6-DOF).

    Parameters
    ----------
    body : ET.Element
        The body element to add the joint to.
    name : str
        Joint name.

    Returns
    -------
    ET.Element
        The created ``<freejoint>`` element.
    """
    return ET.SubElement(body, "freejoint", name=name)


def add_weld_constraint(
    equality: ET.Element,
    *,
    name: str,
    body1: str,
    body2: str,
    relpose: tuple[float, ...] | None = None,
) -> ET.Element:
    """Add a ``<weld>`` constraint to the ``<equality>`` section.

    Parameters
    ----------
    equality : ET.Element
        The ``<equality>`` element.
    name : str
        Constraint name.
    body1, body2 : str
        Names of the two bodies to weld.
    relpose : tuple or None
        Optional 7-element relative pose (x y z qw qx qy qz).

    Returns
    -------
    ET.Element
        The created ``<weld>`` element.
    """
    attrs: dict[str, str] = {
        "name": name,
        "body1": body1,
        "body2": body2,
    }
    if relpose is not None:
        attrs["relpose"] = " ".join(f"{v:.6f}" for v in relpose)

    return ET.SubElement(equality, "weld", attrib=attrs)


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add whitespace indentation to an ElementTree in-place.

    Delegates to :func:`xml.etree.ElementTree.indent` (stdlib, Python 3.9+).
    The *level* parameter is accepted for backward compatibility but ignored;
    the stdlib implementation always starts from the root indentation level.
    """
    ET.indent(elem, space="  ", level=level)
    if level == 0:
        elem.tail = "\n"


def serialize_model(root: ET.Element) -> str:
    """Serialize a MuJoCo MJCF ElementTree to a formatted XML string."""
    logger.debug("Serializing MJCF model with root tag=%s", root.tag)
    indent_xml(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
