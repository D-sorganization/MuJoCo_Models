"""MJCF body assembly and orchestration for the full-body musculoskeletal model.

Segments (bilateral where noted):
  pelvis, torso, head,
  upper_arm_{l,r}, forearm_{l,r}, hand_{l,r},
  thigh_{l,r}, shank_{l,r}, foot_{l,r}

Joints (multi-DOF where physiologically appropriate):
  ground_pelvis (freejoint -- 6 DOF),
  lumbar (3-DOF: flex, lateral, rotate),
  neck (hinge),
  shoulder_{l,r} (3-DOF: flex, adduct, rotate),
  elbow_{l,r} (hinge),
  wrist_{l,r} (2-DOF: flex, deviate),
  hip_{l,r} (3-DOF: flex, adduct, rotate),
  knee_{l,r} (hinge),
  ankle_{l,r} (2-DOF: flex, invert)

Anthropometric defaults are for a 50th-percentile male (height=1.75 m,
mass=80 kg) following Winter (2009) segment proportions.

MuJoCo convention: Z-up. Vertical axis is Z, not Y.

Law of Demeter: exercise modules call create_full_body() and receive
body elements -- they never manipulate segment internals.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from mujoco_models.shared.body.body_anthropometrics import (
    _LEG_HEIGHT_FRACTION,
    BodyModelSpec,
    _seg,
)
from mujoco_models.shared.body.segment_data import (
    ANKLE_FLEX_MAX,
    ANKLE_FLEX_MIN,
    ANKLE_INVERT_MAX,
    ANKLE_INVERT_MIN,
    HIP_ADDUCT_MAX,
    HIP_ADDUCT_MIN,
    HIP_FLEX_MAX,
    HIP_FLEX_MIN,
    HIP_ROTATE_MAX,
    HIP_ROTATE_MIN,
    LUMBAR_FLEX_MAX,
    LUMBAR_FLEX_MIN,
    LUMBAR_LATERAL_MAX,
    LUMBAR_LATERAL_MIN,
    LUMBAR_ROTATE_MAX,
    LUMBAR_ROTATE_MIN,
    SHOULDER_ADDUCT_MAX,
    SHOULDER_ADDUCT_MIN,
    SHOULDER_FLEX_MAX,
    SHOULDER_FLEX_MIN,
    SHOULDER_ROTATE_MAX,
    SHOULDER_ROTATE_MIN,
    WRIST_DEVIATE_MAX,
    WRIST_DEVIATE_MIN,
    WRIST_FLEX_MAX,
    WRIST_FLEX_MIN,
)
from mujoco_models.shared.utils.geometry import (
    capsule_inertia,
    rectangular_prism_inertia,
    sphere_inertia,
)
from mujoco_models.shared.utils.mjcf_helpers import (
    add_body,
    add_free_joint,
    add_hinge_joint,
)

logger = logging.getLogger(__name__)

# Type alias for extra hinge-joint specs: (suffix, axis, range_min, range_max)
_ExtraJoints = list[tuple[str, tuple[float, float, float], float, float]]

# Re-exported so existing ``from mujoco_models.shared.body.body_model import
# _LEG_HEIGHT_FRACTION`` imports (including tests) continue to work.
__all__ = [
    "BodyModelSpec",
    "_LEG_HEIGHT_FRACTION",
    "create_full_body",
]


def _add_limb_side(
    parent_el: ET.Element,
    *,
    body_name: str,
    pos: tuple[float, float, float],
    mass: float,
    inertia: tuple[float, float, float],
    radius: float,
    length: float,
    coord_prefix: str,
    side: str,
    range_min: float,
    range_max: float,
    extra_joints: _ExtraJoints | None = None,
) -> ET.Element:
    """Create one side of a bilateral limb segment with its hinge joints."""
    child_body = add_body(
        parent_el,
        name=body_name,
        pos=pos,
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
    if extra_joints:
        for suffix, axis, ex_min, ex_max in extra_joints:
            add_hinge_joint(
                child_body,
                name=f"{coord_prefix}_{side}_{suffix}",
                axis=axis,
                range_min=ex_min,
                range_max=ex_max,
            )
    return child_body


def _add_bilateral_limb(
    parent_bodies: dict[str, ET.Element],
    spec: BodyModelSpec,
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
    mass, length, radius = _seg(spec, seg_name)
    inertia = capsule_inertia(mass, radius, length)
    parent_is_bilateral = f"{parent_name}_l" in parent_bodies
    created: dict[str, ET.Element] = {}
    for side, sign in [("l", -1.0), ("r", 1.0)]:
        body_name = f"{seg_name}_{side}"
        key = f"{parent_name}_{side}" if parent_is_bilateral else parent_name
        created[body_name] = _add_limb_side(
            parent_bodies[key],
            body_name=body_name,
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
    return created


def _build_pelvis(
    worldbody: ET.Element,
    spec: BodyModelSpec,
) -> tuple[ET.Element, float, float]:
    """Add the pelvis body (with freejoint) to worldbody.

    Returns (pelvis_body, p_len, p_rad) so callers can derive child offsets.
    """
    p_mass, p_len, p_rad = _seg(spec, "pelvis")
    p_inertia = rectangular_prism_inertia(p_mass, p_rad * 2, p_len, p_rad * 2)
    pelvis_body = add_body(
        worldbody,
        name="pelvis",
        pos=(0, 0, spec.pelvis_height),
        mass=p_mass,
        inertia_diag=p_inertia,
        geom_type="box",
        geom_size=(p_rad, p_rad, p_len / 2.0),
        geom_rgba="0.8 0.6 0.4 1",
    )
    add_free_joint(pelvis_body, name="ground_pelvis")
    return pelvis_body, p_len, p_rad


def _build_torso(
    pelvis_body: ET.Element,
    spec: BodyModelSpec,
    p_len: float,
) -> tuple[ET.Element, float, float]:
    """Add torso (with lumbar joints) as child of pelvis.

    Returns (torso_body, t_len, t_rad) so callers can derive child offsets.
    """
    t_mass, t_len, t_rad = _seg(spec, "torso")
    t_inertia = rectangular_prism_inertia(t_mass, t_rad * 2, t_len, t_rad * 2)
    torso_body = add_body(
        pelvis_body,
        name="torso",
        pos=(0, 0, p_len / 2.0),
        mass=t_mass,
        inertia_diag=t_inertia,
        geom_type="box",
        geom_size=(t_rad, t_rad * 0.8, t_len / 2.0),
        geom_rgba="0.8 0.6 0.4 1",
    )
    add_hinge_joint(
        torso_body,
        name="lumbar_flex",
        axis=(1, 0, 0),
        range_min=LUMBAR_FLEX_MIN,
        range_max=LUMBAR_FLEX_MAX,
    )
    add_hinge_joint(
        torso_body,
        name="lumbar_lateral",
        axis=(0, 0, 1),
        range_min=LUMBAR_LATERAL_MIN,
        range_max=LUMBAR_LATERAL_MAX,
    )
    add_hinge_joint(
        torso_body,
        name="lumbar_rotate",
        axis=(0, 1, 0),
        range_min=LUMBAR_ROTATE_MIN,
        range_max=LUMBAR_ROTATE_MAX,
    )
    return torso_body, t_len, t_rad


def _build_head(
    torso_body: ET.Element,
    spec: BodyModelSpec,
    t_len: float,
) -> ET.Element:
    """Add head (with neck hinge) as child of torso."""
    h_mass, _h_len, h_rad = _seg(spec, "head")
    h_inertia = sphere_inertia(h_mass, h_rad)
    head_body = add_body(
        torso_body,
        name="head",
        pos=(0, 0, t_len),
        mass=h_mass,
        inertia_diag=h_inertia,
        geom_type="sphere",
        geom_size=(h_rad,),
        geom_rgba="0.9 0.75 0.6 1",
    )
    add_hinge_joint(
        head_body, name="neck_flex", axis=(1, 0, 0), range_min=-0.5236, range_max=0.5236
    )
    return head_body


def _build_axial_skeleton(
    worldbody: ET.Element,
    spec: BodyModelSpec,
) -> dict[str, ET.Element]:
    """Build pelvis, torso, and head -- the axial skeleton.

    Returns a dict of body-name to ET.Element for the three central
    segments.  The pelvis height is derived from anthropometric data
    via ``spec.pelvis_height``.
    """
    pelvis_body, p_len, _p_rad = _build_pelvis(worldbody, spec)
    torso_body, t_len, _t_rad = _build_torso(pelvis_body, spec, p_len)
    head_body = _build_head(torso_body, spec, t_len)
    return {"pelvis": pelvis_body, "torso": torso_body, "head": head_body}


def _attach_upper_arms(bodies: dict[str, ET.Element], spec: BodyModelSpec) -> None:
    """Attach bilateral upper-arm segments (shoulder joints) to torso."""
    _t_mass, t_len, t_rad = _seg(spec, "torso")
    bodies.update(
        _add_bilateral_limb(
            bodies,
            spec,
            seg_name="upper_arm",
            parent_name="torso",
            parent_offset_z=t_len * 0.95,
            parent_lateral_x=t_rad * 1.2,
            coord_prefix="shoulder",
            range_min=SHOULDER_FLEX_MIN,
            range_max=SHOULDER_FLEX_MAX,
            extra_joints=[
                ("adduct", (0, 0, 1), SHOULDER_ADDUCT_MIN, SHOULDER_ADDUCT_MAX),
                ("rotate", (0, 1, 0), SHOULDER_ROTATE_MIN, SHOULDER_ROTATE_MAX),
            ],
        )
    )


def _attach_forearms(bodies: dict[str, ET.Element], spec: BodyModelSpec) -> None:
    """Attach bilateral forearm segments (elbow hinges) to upper arms."""
    _ua_mass, ua_len, _ua_rad = _seg(spec, "upper_arm")
    bodies.update(
        _add_bilateral_limb(
            bodies,
            spec,
            seg_name="forearm",
            parent_name="upper_arm",
            parent_offset_z=-ua_len,
            parent_lateral_x=0,
            coord_prefix="elbow",
            range_min=0,
            range_max=2.618,
        )
    )


def _attach_hands(bodies: dict[str, ET.Element], spec: BodyModelSpec) -> None:
    """Attach bilateral hand segments (wrist joints) to forearms."""
    _fa_mass, fa_len, _fa_rad = _seg(spec, "forearm")
    bodies.update(
        _add_bilateral_limb(
            bodies,
            spec,
            seg_name="hand",
            parent_name="forearm",
            parent_offset_z=-fa_len,
            parent_lateral_x=0,
            coord_prefix="wrist",
            range_min=WRIST_FLEX_MIN,
            range_max=WRIST_FLEX_MAX,
            extra_joints=[("deviate", (0, 0, 1), WRIST_DEVIATE_MIN, WRIST_DEVIATE_MAX)],
        )
    )


def _build_upper_limbs(
    bodies: dict[str, ET.Element],
    spec: BodyModelSpec,
) -> None:
    """Attach bilateral upper-limb chains (arms + hands) to the torso."""
    _attach_upper_arms(bodies, spec)
    _attach_forearms(bodies, spec)
    _attach_hands(bodies, spec)


def _attach_thighs(bodies: dict[str, ET.Element], spec: BodyModelSpec) -> None:
    """Attach bilateral thigh segments (hip joints) to pelvis."""
    _p_mass, p_len, p_rad = _seg(spec, "pelvis")
    bodies.update(
        _add_bilateral_limb(
            bodies,
            spec,
            seg_name="thigh",
            parent_name="pelvis",
            parent_offset_z=-p_len / 2.0,
            parent_lateral_x=p_rad * 0.6,
            coord_prefix="hip",
            range_min=HIP_FLEX_MIN,
            range_max=HIP_FLEX_MAX,
            extra_joints=[
                ("adduct", (0, 0, 1), HIP_ADDUCT_MIN, HIP_ADDUCT_MAX),
                ("rotate", (0, 1, 0), HIP_ROTATE_MIN, HIP_ROTATE_MAX),
            ],
        )
    )


def _attach_shanks(bodies: dict[str, ET.Element], spec: BodyModelSpec) -> None:
    """Attach bilateral shank segments (knee hinges) to thighs."""
    _th_mass, th_len, _th_rad = _seg(spec, "thigh")
    bodies.update(
        _add_bilateral_limb(
            bodies,
            spec,
            seg_name="shank",
            parent_name="thigh",
            parent_offset_z=-th_len,
            parent_lateral_x=0,
            coord_prefix="knee",
            range_min=-2.618,
            range_max=0,
        )
    )


def _attach_feet(bodies: dict[str, ET.Element], spec: BodyModelSpec) -> None:
    """Attach bilateral foot segments (ankle joints) to shanks."""
    _sh_mass, sh_len, _sh_rad = _seg(spec, "shank")
    bodies.update(
        _add_bilateral_limb(
            bodies,
            spec,
            seg_name="foot",
            parent_name="shank",
            parent_offset_z=-sh_len,
            parent_lateral_x=0,
            coord_prefix="ankle",
            range_min=ANKLE_FLEX_MIN,
            range_max=ANKLE_FLEX_MAX,
            extra_joints=[("invert", (0, 0, 1), ANKLE_INVERT_MIN, ANKLE_INVERT_MAX)],
        )
    )


def _build_lower_limbs(
    bodies: dict[str, ET.Element],
    spec: BodyModelSpec,
) -> None:
    """Attach bilateral lower-limb chains (legs + feet) to the pelvis."""
    _attach_thighs(bodies, spec)
    _attach_shanks(bodies, spec)
    _attach_feet(bodies, spec)


def create_full_body(
    worldbody: ET.Element,
    spec: BodyModelSpec | None = None,
) -> dict[str, ET.Element]:
    """Build the full-body model and append bodies to worldbody.

    Returns dict of body name -> ET.Element for all created bodies.

    The pelvis height is derived from Winter (2009) anthropometric
    proportions: ``height * 0.530 + pelvis_length / 2``.  For a
    1.75 m person this evaluates to approximately 1.015 m.
    """
    if spec is None:
        spec = BodyModelSpec()

    logger.info(
        "Building full-body model: mass=%.1f kg, height=%.2f m",
        spec.total_mass,
        spec.height,
    )

    # Stage 1: axial skeleton (pelvis, torso, head)
    bodies = _build_axial_skeleton(worldbody, spec)

    # Stage 2: upper limbs (shoulders -> hands)
    _build_upper_limbs(bodies, spec)

    # Stage 3: lower limbs (hips -> feet)
    _build_lower_limbs(bodies, spec)

    # Stage 4: foot contact geometry
    _add_foot_contact_geoms(bodies)

    logger.debug("Created %d body segments", len(bodies))
    return bodies


def _add_foot_contact_geoms(bodies: dict[str, ET.Element]) -> None:
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
