"""Helper functions for body model assembly.

Extracted from body_model.py to keep modules under the 300-line budget.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.shared.utils.geometry import capsule_inertia
from mujoco_models.shared.utils.mjcf_helpers import add_body, add_hinge_joint

logger = logging.getLogger(__name__)


def add_foot_contact_geoms(bodies: dict[str, ET.Element]) -> None:
    """Add box collision geometry to foot segments for ground contact.

    Each foot gets a box geom representing the sole contact area:
    ~0.26 m long x 0.10 m wide x 0.02 m thick, positioned at the
    bottom of the foot segment.

    Contact properties: contype=1, conaffinity=1, condim=3,
    friction="1.0 0.005 0.0001" (tangent, torsion, rolling).
    Group=1 separates contact geoms from visual geoms (group=0).
    """
    for side in ("l", "r"):
        foot_body = bodies.get(f"foot_{side}")
        if foot_body is None:
            continue

        # Get the visual geom to determine the foot's vertical extent
        visual_geom = foot_body.find("geom")
        if visual_geom is not None:
            visual_geom.set("group", "0")

        contact_geom = ET.SubElement(foot_body, "geom")
        contact_geom.set("name", f"foot_{side}_contact")
        contact_geom.set("type", "box")
        contact_geom.set(
            "size", "0.13 0.05 0.01"
        )  # half-sizes: 0.26/2 x 0.10/2 x 0.02/2
        contact_geom.set(
            "pos", "0.04 0 -0.02"
        )  # slightly forward and at bottom of foot
        contact_geom.set("contype", "1")
        contact_geom.set("conaffinity", "1")
        contact_geom.set("condim", "3")
        contact_geom.set("friction", "1.0 0.005 0.0001")
        contact_geom.set("group", "1")
        contact_geom.set(
            "rgba", "0.8 0.6 0.4 0.3"
        )  # semi-transparent for visualization

    logger.debug("Added contact sole geometry to foot segments")


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
    extra_joints: list[tuple[str, tuple[float, float, float], float, float]]
    | None = None,
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

    created: dict[str, ET.Element] = {}

    # Determine if parent is bilateral (has _l/_r variants) or central
    parent_is_bilateral = f"{parent_name}_l" in parent_bodies

    for side, sign in [("l", -1.0), ("r", 1.0)]:
        body_name = f"{seg_name}_{side}"
        resolved_parent = (
            f"{parent_name}_{side}" if parent_is_bilateral else parent_name
        )
        parent_el = parent_bodies[resolved_parent]

        child_body = add_body(
            parent_el,
            name=body_name,
            pos=(sign * parent_lateral_x, 0, parent_offset_z),
            mass=mass,
            inertia_diag=inertia,
            geom_type="capsule",
            geom_size=(radius, length / 2.0),
            geom_rgba="0.8 0.6 0.4 1",
        )

        add_hinge_joint(
            child_body,
            name=f"{coord_prefix}_{side}_flex",
            axis=(1, 0, 0),
            range_min=range_min,
            range_max=range_max,
        )

        # Add extra DOFs (stacked hinge joints on the same body)
        if extra_joints:
            for suffix, axis, ex_min, ex_max in extra_joints:
                add_hinge_joint(
                    child_body,
                    name=f"{coord_prefix}_{side}_{suffix}",
                    axis=axis,
                    range_min=ex_min,
                    range_max=ex_max,
                )

        created[body_name] = child_body

    return created
