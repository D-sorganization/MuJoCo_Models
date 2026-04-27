# SPDX-License-Identifier: MIT
"""Lower-limb assembly: hips, knees, and ankles.

Attaches bilateral thigh, shank, and foot segments to the pelvis,
forming the complete lower-limb kinematic chains.  Also adds foot
contact geometry for ground interaction.

Extracted from body_model.py to keep modules under the 300-line guideline.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_models.shared.body.body_anthropometrics import (
    BodyModelSpec,
    _seg,
)
from mujoco_models.shared.body.body_helpers import (
    add_bilateral_limb,
    add_foot_contact_geoms,
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
)


def _attach_thighs(bodies: dict[str, ET.Element], spec: BodyModelSpec) -> None:
    """Attach bilateral thigh segments (hip joints) to pelvis."""
    _p_mass, p_len, p_rad = _seg(spec, "pelvis")
    th_mass, th_len, th_rad = _seg(spec, "thigh")
    bodies.update(
        add_bilateral_limb(
            bodies,
            th_mass,
            th_len,
            th_rad,
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
    sh_mass, sh_len, sh_rad = _seg(spec, "shank")
    bodies.update(
        add_bilateral_limb(
            bodies,
            sh_mass,
            sh_len,
            sh_rad,
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
    ft_mass, ft_len, ft_rad = _seg(spec, "foot")
    bodies.update(
        add_bilateral_limb(
            bodies,
            ft_mass,
            ft_len,
            ft_rad,
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


def build_lower_limbs(
    bodies: dict[str, ET.Element],
    spec: BodyModelSpec,
) -> None:
    """Attach bilateral lower-limb chains (legs + feet) to the pelvis.

    Also adds foot contact geometry for ground interaction.
    """
    _attach_thighs(bodies, spec)
    _attach_shanks(bodies, spec)
    _attach_feet(bodies, spec)
    add_foot_contact_geoms(bodies)
