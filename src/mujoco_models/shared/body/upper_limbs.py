"""Upper-limb assembly: shoulders, elbows, and wrists.

Attaches bilateral upper-arm, forearm, and hand segments to the torso,
forming the complete upper-limb kinematic chains.

Extracted from body_model.py to keep modules under the 300-line guideline.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mujoco_models.shared.body.body_anthropometrics import (
    BodyModelSpec,
    _seg,
)
from mujoco_models.shared.body.body_helpers import add_bilateral_limb
from mujoco_models.shared.body.segment_data import (
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


def _attach_upper_arms(bodies: dict[str, ET.Element], spec: BodyModelSpec) -> None:
    """Attach bilateral upper-arm segments (shoulder joints) to torso."""
    _t_mass, t_len, t_rad = _seg(spec, "torso")
    ua_mass, ua_len, ua_rad = _seg(spec, "upper_arm")
    bodies.update(
        add_bilateral_limb(
            bodies,
            ua_mass,
            ua_len,
            ua_rad,
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
    fa_mass, fa_len, fa_rad = _seg(spec, "forearm")
    bodies.update(
        add_bilateral_limb(
            bodies,
            fa_mass,
            fa_len,
            fa_rad,
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
    h_mass, h_len, h_rad = _seg(spec, "hand")
    bodies.update(
        add_bilateral_limb(
            bodies,
            h_mass,
            h_len,
            h_rad,
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


def build_upper_limbs(
    bodies: dict[str, ET.Element],
    spec: BodyModelSpec,
) -> None:
    """Attach bilateral upper-limb chains (arms + hands) to the torso."""
    _attach_upper_arms(bodies, spec)
    _attach_forearms(bodies, spec)
    _attach_hands(bodies, spec)
