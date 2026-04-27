# SPDX-License-Identifier: MIT
"""Axial skeleton assembly: pelvis, torso, and head.

Builds the central body segments (pelvis with freejoint, torso with
lumbar joints, and head with neck hinge) that form the axial skeleton
of the musculoskeletal model.

Extracted from body_model.py to keep modules under the 300-line guideline.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_models.shared.body.body_anthropometrics import (
    BodyModelSpec,
    _seg,
)
from mujoco_models.shared.body.segment_data import (
    LUMBAR_FLEX_MAX,
    LUMBAR_FLEX_MIN,
    LUMBAR_LATERAL_MAX,
    LUMBAR_LATERAL_MIN,
    LUMBAR_ROTATE_MAX,
    LUMBAR_ROTATE_MIN,
)
from mujoco_models.shared.utils.geometry import (
    rectangular_prism_inertia,
    sphere_inertia,
)
from mujoco_models.shared.utils.mjcf_helpers import (
    add_body,
    add_free_joint,
    add_hinge_joint,
)


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


def build_axial_skeleton(
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
