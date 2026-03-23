"""Base exercise model builder for MuJoCo MJCF.

DRY: All five exercises share the same skeleton for creating a MuJoCo
MJCF model XML -- differing only in barbell attachment strategy, initial
pose, and joint coordinate defaults. This base class encapsulates the
shared workflow; subclasses override hooks to customize.

Law of Demeter: Exercise builders interact with BarbellSpec and BodyModelSpec
through their public APIs, never reaching into internal segment tables.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mujoco_models.shared.barbell import BarbellSpec, create_barbell_bodies
from mujoco_models.shared.body import BodyModelSpec, create_full_body
from mujoco_models.shared.contracts.postconditions import ensure_mjcf_root
from mujoco_models.shared.utils.mjcf_helpers import serialize_model


@dataclass
class ExerciseConfig:
    """Configuration common to all exercise models."""

    body_spec: BodyModelSpec = field(default_factory=BodyModelSpec)
    barbell_spec: BarbellSpec = field(default_factory=BarbellSpec.mens_olympic)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.80665)


class ExerciseModelBuilder(ABC):
    """Abstract builder for exercise-specific MuJoCo MJCF models.

    Subclasses must implement:
      - exercise_name: str property
      - attach_barbell(): how the barbell connects to the body
      - set_initial_pose(): default coordinate values for the start position
    """

    def __init__(self, config: ExerciseConfig | None = None) -> None:
        self.config = config or ExerciseConfig()

    @property
    @abstractmethod
    def exercise_name(self) -> str:
        """Human-readable exercise name used in the model XML."""

    @abstractmethod
    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        """Add constraints connecting barbell to body (exercise-specific)."""

    @abstractmethod
    def set_initial_pose(self, worldbody: ET.Element) -> None:
        """Set default coordinate values for the starting position."""

    def build(self) -> str:
        """Build the complete MuJoCo MJCF model XML and return as string.

        Postcondition: returned string is well-formed MJCF XML with
        ``<mujoco>`` as the root element.
        """
        root = ET.Element("mujoco", model=self.exercise_name)

        # Option: gravity and timestep
        g = self.config.gravity
        ET.SubElement(
            root,
            "option",
            gravity=f"{g[0]:.6f} {g[1]:.6f} {g[2]:.6f}",
            timestep="0.001",
        )

        # Compiler settings (Z-up)
        ET.SubElement(root, "compiler", angle="radian", coordinate="local")

        # Default joint damping
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "joint", damping="5.0", armature="0.1")
        ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")

        # Worldbody
        worldbody = ET.SubElement(root, "worldbody")

        # Ground plane
        ET.SubElement(
            worldbody,
            "geom",
            name="ground",
            type="plane",
            size="5 5 0.1",
            rgba="0.9 0.9 0.9 1",
        )

        # Equality constraints section
        equality = ET.SubElement(root, "equality")

        # Build body
        body_bodies = create_full_body(worldbody, self.config.body_spec)

        # Build barbell
        barbell_bodies = create_barbell_bodies(
            worldbody, equality, self.config.barbell_spec
        )

        # Exercise-specific attachment
        self.attach_barbell(equality, body_bodies, barbell_bodies)

        # Exercise-specific initial pose
        self.set_initial_pose(worldbody)

        xml_str = serialize_model(root)

        # Postcondition: well-formed MJCF XML
        ensure_mjcf_root(xml_str)

        return xml_str
