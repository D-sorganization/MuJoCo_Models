"""Olympic barbell model for MuJoCo MJCF.

Standard Olympic barbell dimensions (IWF / IPF regulations):
- Men's bar: 2.20 m total length, 1.31 m between collars, 28 mm shaft
  diameter, 50 mm sleeve diameter, 20 kg unloaded.
- Women's bar: 2.01 m total length, 1.31 m between collars, 25 mm shaft
  diameter, 50 mm sleeve diameter, 15 kg unloaded.

The barbell is modelled as three rigid bodies (left_sleeve, shaft, right_sleeve)
connected by weld constraints. Plates are added as additional mass on the sleeves.

MuJoCo convention: Z-up, barbell shaft extends along the X-axis.

Law of Demeter: callers interact only with BarbellSpec and create_barbell_bodies;
internal geometry details remain encapsulated.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization


from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from mujoco_models.shared.contracts.preconditions import (
    require_non_negative,
    require_positive,
)
from mujoco_models.shared.utils.geometry import (
    cylinder_inertia,
    hollow_cylinder_inertia,
)
from mujoco_models.shared.utils.mjcf_helpers import (
    add_body,
    add_weld_constraint,
)


@dataclass(frozen=True)
class BarbellSpec:
    """Immutable specification for a barbell.

    All lengths in meters, masses in kg, diameters in meters.
    """

    total_length: float = 2.20
    shaft_length: float = 1.31
    shaft_diameter: float = 0.028
    sleeve_diameter: float = 0.050
    bar_mass: float = 20.0
    plate_mass_per_side: float = 0.0

    def __post_init__(self) -> None:
        """Validate barbell specification dimensions and masses."""
        require_positive(self.total_length, "total_length")
        require_positive(self.shaft_length, "shaft_length")
        require_positive(self.shaft_diameter, "shaft_diameter")
        require_positive(self.sleeve_diameter, "sleeve_diameter")
        require_positive(self.bar_mass, "bar_mass")
        require_non_negative(self.plate_mass_per_side, "plate_mass_per_side")
        if self.shaft_length >= self.total_length:
            raise ValueError(
                f"shaft_length ({self.shaft_length}) must be < "
                f"total_length ({self.total_length})"
            )

    @property
    def sleeve_length(self) -> float:
        """Length of one sleeve (half of non-shaft portion)."""
        return (self.total_length - self.shaft_length) / 2.0

    @property
    def shaft_radius(self) -> float:
        """Shaft radius in meters (half the shaft diameter)."""
        return self.shaft_diameter / 2.0

    @property
    def sleeve_radius(self) -> float:
        """Sleeve radius in meters (half the sleeve diameter)."""
        return self.sleeve_diameter / 2.0

    @property
    def shaft_mass(self) -> float:
        """Mass attributed to the shaft (proportional to length)."""
        shaft_fraction = self.shaft_length / self.total_length
        return self.bar_mass * shaft_fraction

    @property
    def sleeve_mass(self) -> float:
        """Mass of one bare sleeve (no plates)."""
        sleeve_fraction = self.sleeve_length / self.total_length
        return self.bar_mass * sleeve_fraction

    @property
    def total_mass(self) -> float:
        """Total barbell mass including plates on both sides."""
        return self.bar_mass + 2.0 * self.plate_mass_per_side

    @classmethod
    def mens_olympic(cls, plate_mass_per_side: float = 0.0) -> BarbellSpec:
        """Standard men's Olympic barbell (20 kg, 2.20 m)."""
        return cls(plate_mass_per_side=plate_mass_per_side)

    @classmethod
    def womens_olympic(cls, plate_mass_per_side: float = 0.0) -> BarbellSpec:
        """Standard women's Olympic barbell (15 kg, 2.01 m)."""
        return cls(
            total_length=2.01,
            shaft_length=1.31,
            shaft_diameter=0.025,
            sleeve_diameter=0.050,
            bar_mass=15.0,
            plate_mass_per_side=plate_mass_per_side,
        )


def _compute_sleeve_inertia(
    spec: BarbellSpec,
) -> tuple[float, float, float]:
    """Compute combined sleeve+plate inertia for one sleeve.

    Starts with bare sleeve inertia and adds plate contribution when
    plates are loaded.  Plate outer radius follows IWF standard (0.225 m).
    """
    sleeve_inertia = cylinder_inertia(
        spec.sleeve_mass, spec.sleeve_radius, spec.sleeve_length
    )
    if spec.plate_mass_per_side <= 0:
        return sleeve_inertia

    plate_thickness = max(0.01, spec.plate_mass_per_side * 0.002)
    plate_inertia = hollow_cylinder_inertia(
        spec.plate_mass_per_side,
        inner_radius=spec.sleeve_radius,
        outer_radius=0.225,
        length=plate_thickness,
    )
    return (
        sleeve_inertia[0] + plate_inertia[0],
        sleeve_inertia[1] + plate_inertia[1],
        sleeve_inertia[2] + plate_inertia[2],
    )


def _add_barbell_shaft(
    worldbody: ET.Element,
    spec: BarbellSpec,
    shaft_name: str,
    shaft_inertia: tuple[float, float, float],
) -> ET.Element:
    """Add the central shaft body to worldbody at the origin."""
    return add_body(
        worldbody,
        name=shaft_name,
        pos=(0, 0, 0),
        mass=spec.shaft_mass,
        inertia_diag=shaft_inertia,
        geom_type="cylinder",
        geom_size=(spec.shaft_radius, spec.shaft_length / 2.0),
        geom_rgba="0.7 0.7 0.7 1",
        geom_euler=(0, 90, 0),
    )


def _add_barbell_sleeve(
    worldbody: ET.Element,
    spec: BarbellSpec,
    sleeve_name: str,
    sleeve_total_mass: float,
    sleeve_inertia: tuple[float, float, float],
    x_offset: float,
) -> ET.Element:
    """Add one sleeve body at the given X offset from the shaft centre."""
    return add_body(
        worldbody,
        name=sleeve_name,
        pos=(x_offset, 0, 0),
        mass=sleeve_total_mass,
        inertia_diag=sleeve_inertia,
        geom_type="cylinder",
        geom_size=(spec.sleeve_radius, spec.sleeve_length / 2.0),
        geom_rgba="0.5 0.5 0.5 1",
        geom_euler=(0, 90, 0),
    )


def _weld_sleeves_to_shaft(
    equality: ET.Element, shaft_name: str, left_name: str, right_name: str, prefix: str
) -> None:
    """Add weld constraints connecting both sleeves to the shaft."""
    add_weld_constraint(
        equality, name=f"{prefix}_left_weld", body1=shaft_name, body2=left_name
    )
    add_weld_constraint(
        equality, name=f"{prefix}_right_weld", body1=shaft_name, body2=right_name
    )


def create_barbell_bodies(
    worldbody: ET.Element,
    equality: ET.Element,
    spec: BarbellSpec,
    *,
    prefix: str = "barbell",
) -> dict[str, ET.Element]:
    """Add barbell bodies and weld constraints to an MJCF model.

    Returns dict of created body elements keyed by name.

    The barbell shaft center is at the local origin. Sleeves extend
    symmetrically along the X-axis (left = -X, right = +X).
    """
    shaft_inertia = cylinder_inertia(
        spec.shaft_mass, spec.shaft_radius, spec.shaft_length
    )
    sleeve_inertia = _compute_sleeve_inertia(spec)
    sleeve_total_mass = spec.sleeve_mass + spec.plate_mass_per_side
    sleeve_offset = spec.shaft_length / 2.0 + spec.sleeve_length / 2.0

    shaft_name = f"{prefix}_shaft"
    left_name = f"{prefix}_left_sleeve"
    right_name = f"{prefix}_right_sleeve"

    shaft_body = _add_barbell_shaft(worldbody, spec, shaft_name, shaft_inertia)
    left_body = _add_barbell_sleeve(
        worldbody, spec, left_name, sleeve_total_mass, sleeve_inertia, -sleeve_offset
    )
    right_body = _add_barbell_sleeve(
        worldbody, spec, right_name, sleeve_total_mass, sleeve_inertia, sleeve_offset
    )
    _weld_sleeves_to_shaft(equality, shaft_name, left_name, right_name, prefix)

    return {shaft_name: shaft_body, left_name: left_body, right_name: right_body}
