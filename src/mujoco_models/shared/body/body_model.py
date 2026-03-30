"""Simplified full-body musculoskeletal model for MuJoCo MJCF.

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
from dataclasses import dataclass

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
    segment_properties,
)
from mujoco_models.shared.contracts.preconditions import (
    require_positive,
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

# Winter (2009): thigh(0.245) + shank(0.246) + foot(0.039) ≈ 0.530 of height.
# Used to derive standing pelvis height from anthropometric data rather than
# a hardcoded constant.  For a 1.75 m person this yields approximately 1.015 m.
_LEG_HEIGHT_FRACTION: float = 0.530


@dataclass(frozen=True)
class BodyModelSpec:
    """Anthropometric specification for the full-body model.

    All lengths in meters, mass in kg.
    """

    total_mass: float = 80.0
    height: float = 1.75

    def __post_init__(self) -> None:
        """Validate that total_mass and height are strictly positive."""
        require_positive(self.total_mass, "total_mass")
        require_positive(self.height, "height")

    @property
    def pelvis_height(self) -> float:
        """Derive standing pelvis (hip-joint) height from anthropometric data.

        Uses Winter (2009) leg-height fraction (thigh + shank + foot = 0.530
        of total height) plus half the pelvis segment length so the pelvis
        center sits just above the hip joints.

        For a 1.75 m person this yields approximately 1.015 m, derived
        from *height* rather than being hard-coded.
        """
        _mass, p_len, _radius = segment_properties(
            self.total_mass, self.height, "pelvis"
        )
        return self.height * _LEG_HEIGHT_FRACTION + p_len / 2.0


def _seg(spec: BodyModelSpec, name: str) -> tuple[float, float, float]:
    """Return (mass, length, radius) for a named segment."""
    return segment_properties(spec.total_mass, spec.height, name)


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
    mass, length, radius = _seg(spec, seg_name)
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


def _build_axial_skeleton(
    worldbody: ET.Element,
    spec: BodyModelSpec,
) -> dict[str, ET.Element]:
    """Build pelvis, torso, and head -- the axial skeleton.

    Returns a dict of body-name to ET.Element for the three central
    segments.  The pelvis height is derived from anthropometric data
    via ``spec.pelvis_height``.
    """
    bodies: dict[str, ET.Element] = {}

    # --- Pelvis (connected to ground via freejoint) ---
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
    bodies["pelvis"] = pelvis_body

    # --- Torso (child of pelvis) ---
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
    bodies["torso"] = torso_body

    # --- Head (child of torso) ---
    h_mass, h_len, h_rad = _seg(spec, "head")
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
        head_body,
        name="neck_flex",
        axis=(1, 0, 0),
        range_min=-0.5236,
        range_max=0.5236,
    )
    bodies["head"] = head_body

    return bodies


def _build_upper_limbs(
    bodies: dict[str, ET.Element],
    spec: BodyModelSpec,
) -> None:
    """Attach bilateral upper-limb chains (arms + hands) to the torso."""
    _t_mass, t_len, t_rad = _seg(spec, "torso")
    shoulder_z = t_len * 0.95
    shoulder_x = t_rad * 1.2

    arm_bodies = _add_bilateral_limb(
        bodies,
        spec,
        seg_name="upper_arm",
        parent_name="torso",
        parent_offset_z=shoulder_z,
        parent_lateral_x=shoulder_x,
        coord_prefix="shoulder",
        range_min=SHOULDER_FLEX_MIN,
        range_max=SHOULDER_FLEX_MAX,
        extra_joints=[
            ("adduct", (0, 0, 1), SHOULDER_ADDUCT_MIN, SHOULDER_ADDUCT_MAX),
            ("rotate", (0, 1, 0), SHOULDER_ROTATE_MIN, SHOULDER_ROTATE_MAX),
        ],
    )
    bodies.update(arm_bodies)

    _ua_mass, ua_len, _ua_rad = _seg(spec, "upper_arm")
    forearm_bodies = _add_bilateral_limb(
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
    bodies.update(forearm_bodies)

    _fa_mass, fa_len, _fa_rad = _seg(spec, "forearm")
    hand_bodies = _add_bilateral_limb(
        bodies,
        spec,
        seg_name="hand",
        parent_name="forearm",
        parent_offset_z=-fa_len,
        parent_lateral_x=0,
        coord_prefix="wrist",
        range_min=WRIST_FLEX_MIN,
        range_max=WRIST_FLEX_MAX,
        extra_joints=[
            ("deviate", (0, 0, 1), WRIST_DEVIATE_MIN, WRIST_DEVIATE_MAX),
        ],
    )
    bodies.update(hand_bodies)


def _build_lower_limbs(
    bodies: dict[str, ET.Element],
    spec: BodyModelSpec,
) -> None:
    """Attach bilateral lower-limb chains (legs + feet) to the pelvis."""
    _p_mass, p_len, p_rad = _seg(spec, "pelvis")
    hip_x = p_rad * 0.6

    thigh_bodies = _add_bilateral_limb(
        bodies,
        spec,
        seg_name="thigh",
        parent_name="pelvis",
        parent_offset_z=-p_len / 2.0,
        parent_lateral_x=hip_x,
        coord_prefix="hip",
        range_min=HIP_FLEX_MIN,
        range_max=HIP_FLEX_MAX,
        extra_joints=[
            ("adduct", (0, 0, 1), HIP_ADDUCT_MIN, HIP_ADDUCT_MAX),
            ("rotate", (0, 1, 0), HIP_ROTATE_MIN, HIP_ROTATE_MAX),
        ],
    )
    bodies.update(thigh_bodies)

    _th_mass, th_len, _th_rad = _seg(spec, "thigh")
    shank_bodies = _add_bilateral_limb(
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
    bodies.update(shank_bodies)

    _sh_mass, sh_len, _sh_rad = _seg(spec, "shank")
    foot_bodies = _add_bilateral_limb(
        bodies,
        spec,
        seg_name="foot",
        parent_name="shank",
        parent_offset_z=-sh_len,
        parent_lateral_x=0,
        coord_prefix="ankle",
        range_min=ANKLE_FLEX_MIN,
        range_max=ANKLE_FLEX_MAX,
        extra_joints=[
            ("invert", (0, 0, 1), ANKLE_INVERT_MIN, ANKLE_INVERT_MAX),
        ],
    )
    bodies.update(foot_bodies)


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
