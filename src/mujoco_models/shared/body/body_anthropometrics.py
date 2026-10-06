# SPDX-License-Identifier: MIT
"""Anthropometric data and lookup helpers for the full-body musculoskeletal model.

Provides the ``BodyModelSpec`` dataclass (total mass and standing height) and
the ``_seg`` helper that maps a spec + segment name to (mass, length, radius)
using Winter (2009) proportions from ``segment_data``.

The leg-height fraction constant is defined here so that both the spec's
``pelvis_height`` property and the MJCF assembly layer share one source of truth.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from __future__ import annotations

from dataclasses import dataclass

from mujoco_models.shared.body.segment_data import segment_properties
from mujoco_models.shared.contracts.preconditions import require_positive

# Winter (2009): thigh(0.245) + shank(0.246) + foot(0.039) ≈ 0.530 of height.
# Used to derive standing pelvis height from anthropometric data rather than
# a hardcoded constant.  For a 1.75 m person this yields approximately 1.015 m.
_LEG_HEIGHT_FRACTION: float = 0.530

# Shoulder joints sit this multiple of the torso radius left/right of the midline.
_SHOULDER_OFFSET_OVER_TORSO_RADIUS: float = 1.2


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

    @property
    def ankle_height(self) -> float:
        """Height of the foot-body origin above the ground at the neutral pose.

        Pelvis centre minus half the pelvis, the thigh and the shank; the foot
        contact box is placed so its sole sits exactly on the ground here.
        """
        _m, p_len, _r = segment_properties(self.total_mass, self.height, "pelvis")
        _m, th_len, _r = segment_properties(self.total_mass, self.height, "thigh")
        _m, sh_len, _r = segment_properties(self.total_mass, self.height, "shank")
        return self.pelvis_height - p_len / 2.0 - th_len - sh_len

    @property
    def hand_height(self) -> float:
        """Height of the hand-body origin above the ground at the neutral pose."""
        _m, p_len, _r = segment_properties(self.total_mass, self.height, "pelvis")
        _m, t_len, _r = segment_properties(self.total_mass, self.height, "torso")
        _m, ua_len, _r = segment_properties(self.total_mass, self.height, "upper_arm")
        _m, fa_len, _r = segment_properties(self.total_mass, self.height, "forearm")
        return self.pelvis_height + p_len / 2.0 + 0.95 * t_len - ua_len - fa_len

    @property
    def shoulder_half_width(self) -> float:
        """Lateral (Y) distance from the midline to each shoulder joint."""
        _m, _l, t_rad = segment_properties(self.total_mass, self.height, "torso")
        return _SHOULDER_OFFSET_OVER_TORSO_RADIUS * t_rad

    @property
    def shoulder_height(self) -> float:
        """Height of the shoulder (upper-arm origin) above the ground."""
        _m, p_len, _r = segment_properties(self.total_mass, self.height, "pelvis")
        _m, t_len, _r = segment_properties(self.total_mass, self.height, "torso")
        return self.pelvis_height + p_len / 2.0 + 0.95 * t_len


def _seg(spec: BodyModelSpec, name: str) -> tuple[float, float, float]:
    """Return (mass, length, radius) for a named segment."""
    return segment_properties(spec.total_mass, spec.height, name)
