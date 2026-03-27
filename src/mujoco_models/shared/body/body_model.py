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

# Winter (2009): thigh(0.245) + shank(0.246) + foot(0.039) ≈ 0.530 of height
_LEG_HEIGHT_FRACTION = 0.530


@dataclass(frozen=True)
class BodyModelSpec:
    """Anthropometric specification for the full-body model.

    All lengths in meters, mass in kg.
    """

    total_mass: float = 80.0
    height: float = 1.75

    def __post_init__(self) -> None:
        require_positive(self.total_mass, "total_mass")
        require_positive(self.height, "height")


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


def create_full_body(
    worldbody: ET.Element,
    spec: BodyModelSpec | None = None,
) -> dict[str, ET.Element]:
    """Build the full-body model and append bodies to worldbody.

    Returns dict of body name -> ET.Element for all created bodies.

    MuJoCo convention: Z-up. The pelvis starts at approximately
    standing hip height (0.93 m for a 1.75 m person).
    """
    if spec is None:
        spec = BodyModelSpec()

    logger.info(
        "Building full-body model: mass=%.1f kg, height=%.2f m",
        spec.total_mass,
        spec.height,
    )

    bodies: dict[str, ET.Element] = {}

    # --- Pelvis (connected to ground via freejoint) ---
    p_mass, p_len, p_rad = _seg(spec, "pelvis")
    p_inertia = rectangular_prism_inertia(p_mass, p_rad * 2, p_len, p_rad * 2)
    pelvis_z = spec.height * _LEG_HEIGHT_FRACTION + p_len / 2.0
    pelvis_body = add_body(
        worldbody,
        name="pelvis",
        pos=(0, 0, pelvis_z),
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

    # --- Arms (children of torso) ---
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

    # --- Legs (children of pelvis) ---
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

    logger.debug("Created %d body segments", len(bodies))
    return bodies
