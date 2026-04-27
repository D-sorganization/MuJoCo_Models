"""Base exercise model builder for MuJoCo MJCF.

DRY: All five exercises share the same skeleton for creating a MuJoCo
MJCF model XML -- differing only in barbell attachment strategy, initial
pose, and joint coordinate defaults. This base class encapsulates the
shared workflow; subclasses override hooks to customize.

Law of Demeter: Exercise builders interact with BarbellSpec and BodyModelSpec
through their public APIs, never reaching into internal segment tables.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization


from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mujoco_models.shared.barbell import BarbellSpec, create_barbell_bodies
from mujoco_models.shared.body import BodyModelSpec, create_full_body
from mujoco_models.shared.contracts.postconditions import ensure_mjcf_root
from mujoco_models.shared.utils.mjcf_helpers import (
    add_weld_constraint,
    serialize_model,
)

logger = logging.getLogger(__name__)

# Shared initial angles for floor-pull exercises (deadlift, snatch, clean-and-jerk).
# Defined here so subclasses can import from a single authoritative location.
FLOOR_PULL_HIP_FLEX: float = 1.3963  # ~80° hip flexion (radians)
FLOOR_PULL_KNEE_FLEX: float = -1.0472  # ~60° knee flexion (radians)

# Adjacent body segment pairs to exclude from self-collision.
# Central pairs (no side suffix needed):
_CENTRAL_EXCLUSION_PAIRS: list[tuple[str, str]] = [
    ("pelvis", "torso"),
    ("torso", "head"),
]

# Bilateral pairs (will be expanded with _l and _r suffixes):
_BILATERAL_EXCLUSION_PAIRS: list[tuple[str, str]] = [
    ("pelvis", "thigh"),
    ("torso", "upper_arm"),
    ("upper_arm", "forearm"),
    ("forearm", "hand"),
    ("thigh", "shank"),
    ("shank", "foot"),
]


def _add_contact_exclusions(contact: ET.Element) -> None:
    """Add <exclude> elements to prevent self-collision between adjacent segments.

    Central body pairs (pelvis-torso, torso-head) are excluded once.
    Bilateral pairs are excluded for both left and right sides.
    """
    for body1, body2 in _CENTRAL_EXCLUSION_PAIRS:
        ET.SubElement(
            contact,
            "exclude",
            name=f"exclude_{body1}_{body2}",
            body1=body1,
            body2=body2,
        )

    for body1, body2 in _BILATERAL_EXCLUSION_PAIRS:
        for side in ("l", "r"):
            b1 = f"{body1}_{side}" if body1 not in ("pelvis", "torso") else body1
            b2 = f"{body2}_{side}"
            ET.SubElement(
                contact,
                "exclude",
                name=f"exclude_{b1}_{b2}",
                body1=b1,
                body2=b2,
            )


@dataclass(frozen=True)
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
        """Initialize with an optional exercise configuration.

        Args:
            config: Exercise configuration. Uses default ``ExerciseConfig``
                (50th-percentile male anthropometrics, men's Olympic barbell)
                if ``None``.
        """
        self.config = config or ExerciseConfig()

    @property
    def body_spec(self) -> BodyModelSpec:
        """Forward to the configured body specification."""
        return self.config.body_spec

    @property
    def barbell_spec(self) -> BarbellSpec:
        """Forward to the configured barbell specification."""
        return self.config.barbell_spec

    @property
    def gravity(self) -> tuple[float, float, float]:
        """Forward to the configured gravity vector."""
        return self.config.gravity

    @property
    def grip_offset(self) -> tuple[float, ...] | None:
        """Optional grip offset for subclasses that need a custom weld pose."""
        return None

    @property
    @abstractmethod
    def exercise_name(self) -> str:
        """Human-readable exercise name used in the model XML."""

    @property
    def uses_barbell(self) -> bool:
        """Return True if this exercise builds and attaches a barbell.

        Override in subclasses (e.g. gait, sit-to-stand) that do not
        use a barbell so that barbell bodies, welds, and attachment
        logic are skipped entirely rather than created with zero mass.
        """
        return True

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

    def _post_worldbody_hook(self, worldbody: ET.Element, equality: ET.Element) -> None:
        """No-op hook called after worldbody is built, before actuator generation.

        Subclasses may override to inject additional bodies or constraints
        (e.g. BenchPressModelBuilder adds the bench body here).
        This is an intentional empty method -- not abstract -- so the base class
        remains instantiable and subclasses are not required to override it.

        B027 is suppressed at the ruff config level for this pattern (intentional
        empty method in an ABC that is not itself abstract).
        """

    @staticmethod
    def _barbell_relpose_for_hand(
        side: str,
        grip_width: float | None = None,
        grip_offset: tuple[float, ...] | None = None,
    ) -> tuple[float, ...] | None:
        """Return the weld relpose for one hand.

        ``grip_offset`` wins when provided. Otherwise, ``grip_width`` is
        converted into a left/right X offset anchored at the hand.
        """
        if grip_offset is not None:
            return grip_offset
        if grip_width is None:
            return None

        sign = -1.0 if side == "l" else 1.0
        return (-sign * grip_width, 0, 0, 1, 0, 0, 0)

    @staticmethod
    def _attach_barbell_to_hand(
        equality: ET.Element, *, side: str, relpose: tuple[float, ...] | None
    ) -> None:
        """Write one hand weld to the shared equality section."""
        add_weld_constraint(
            equality,
            name=f"barbell_to_hand_{side}",
            body1=f"hand_{side}",
            body2="barbell_shaft",
            relpose=relpose,
        )

    def _attach_barbell_to_hands(
        self,
        equality: ET.Element,
        *,
        grip_width: float | None = None,
        grip_offset: tuple[float, ...] | None = None,
    ) -> None:
        """Weld barbell shaft to both hands (DRY helper for subclasses).

        Parameters
        ----------
        equality : ET.Element
            The ``<equality>`` section of the MJCF model.
        grip_width : float or None
            Distance from the barbell shaft center to each hand along the X-axis.
            If None, the initial pose determines the grip width.
        grip_offset : tuple or None
            Optional 7-element relative pose (x y z qw qx qy qz) for the grip
            offset from the hand to the barbell shaft.
        """
        for side in ("l", "r"):
            relpose = self._barbell_relpose_for_hand(
                side, grip_width=grip_width, grip_offset=grip_offset
            )
            self._attach_barbell_to_hand(equality, side=side, relpose=relpose)

    # ------------------------------------------------------------------
    # Shared pose helper (issue #116)
    # ------------------------------------------------------------------

    @staticmethod
    def set_ref_by_name_map(
        worldbody: ET.Element,
        refs_by_name_fragment: dict[str, float],
    ) -> None:
        """Set joint ``ref`` attributes by matching name fragments.

        Iterates all hinge joints in *worldbody*.  For each joint whose
        name contains a key from *refs_by_name_fragment*, sets ``ref`` to
        the corresponding value.  The first matching fragment wins.

        Args:
            worldbody: The ``<worldbody>`` element of the MJCF model.
            refs_by_name_fragment: Mapping from name substring to ref angle
                (radians).  E.g. ``{"hip_flex": 1.396, "knee": -1.047}``.
        """
        for joint in worldbody.iter("joint"):
            if joint.get("type", "hinge") != "hinge":
                continue
            name = joint.get("name", "")
            for fragment, ref in refs_by_name_fragment.items():
                if fragment in name:
                    joint.set("ref", str(ref))
                    break

    # ------------------------------------------------------------------
    # Build pipeline -- decomposed from the original monolithic build()
    # ------------------------------------------------------------------

    def _create_root_element(self) -> ET.Element:
        """Create the ``<mujoco>`` root with option, compiler, and defaults."""
        root = ET.Element("mujoco", model=self.exercise_name)

        g = self.gravity
        ET.SubElement(
            root,
            "option",
            gravity=f"{g[0]:.6f} {g[1]:.6f} {g[2]:.6f}",
            timestep="0.001",
            integrator="implicit",
            cone="elliptic",
            solver="Newton",
            tolerance="1e-8",
        )

        ET.SubElement(root, "compiler", angle="radian")

        default = ET.SubElement(root, "default")
        ET.SubElement(default, "joint", damping="5.0", armature="0.1")
        ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")

        return root

    def _create_worldbody(self, root: ET.Element) -> ET.Element:
        """Create worldbody with ground plane."""
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(
            worldbody,
            "geom",
            name="ground",
            type="plane",
            size="5 5 0.1",
            rgba="0.9 0.9 0.9 1",
            contype="1",
            conaffinity="1",
            condim="3",
            friction="1.0 0.005 0.0001",
        )
        return worldbody

    def _add_actuators_and_sensors(
        self, root: ET.Element, worldbody: ET.Element
    ) -> None:
        """Add position actuators and joint-position sensors for all hinge joints."""
        actuator = ET.SubElement(root, "actuator")
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            if name:
                act = ET.SubElement(actuator, "position")
                act.set("name", f"act_{name}")
                act.set("joint", name)
                act.set("kp", "100")

        sensor = ET.SubElement(root, "sensor")
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            if name:
                s = ET.SubElement(sensor, "jointpos")
                s.set("name", f"pos_{name}")
                s.set("joint", name)

    def _build_keyframe(self, root: ET.Element, worldbody: ET.Element) -> None:
        """Build a named keyframe from joint ref values set by set_initial_pose()."""
        qpos_values: list[str] = []
        for joint_el in worldbody.iter("joint"):
            ref_val = joint_el.get("ref", "0")
            qpos_values.append(ref_val)

        for fj in worldbody.iter("freejoint"):
            for body_el in worldbody.iter("body"):
                fj_in_body = body_el.find("freejoint")
                if fj_in_body is not None and fj_in_body is fj:
                    pos_str = body_el.get("pos", "0 0 0")
                    pos_parts = pos_str.split()
                    fj_qpos = pos_parts + ["1", "0", "0", "0"]
                    qpos_values = fj_qpos + qpos_values
                    break

        if qpos_values:
            keyframe = ET.SubElement(root, "keyframe")
            key = ET.SubElement(keyframe, "key")
            key.set("name", f"{self.exercise_name}_start")
            key.set("qpos", " ".join(qpos_values))

    def _build_bodies_and_barbell(
        self, worldbody: ET.Element, equality: ET.Element
    ) -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
        """Build body segments and (optionally) barbell inside the worldbody.

        Returns ``(body_bodies, barbell_bodies)`` where ``barbell_bodies`` is
        empty when :attr:`uses_barbell` is ``False``.
        """
        body_bodies = create_full_body(worldbody, self.body_spec)
        barbell_bodies: dict[str, ET.Element] = {}
        if self.uses_barbell:
            barbell_bodies = create_barbell_bodies(
                worldbody, equality, self.barbell_spec
            )
            self.attach_barbell(equality, body_bodies, barbell_bodies)
        return body_bodies, barbell_bodies

    def _create_equality(self, root: ET.Element) -> ET.Element:
        """Create the equality section that stores exercise constraints."""
        return ET.SubElement(root, "equality")

    def _add_contact_section(self, root: ET.Element) -> ET.Element:
        """Create the contact section and populate adjacent-segment exclusions."""
        contact = ET.SubElement(root, "contact")
        _add_contact_exclusions(contact)
        return contact

    def _add_state_sections(self, root: ET.Element, worldbody: ET.Element) -> None:
        """Apply initial pose metadata, actuators, sensors, and keyframe data."""
        self.set_initial_pose(worldbody)
        self._add_actuators_and_sensors(root, worldbody)
        self._build_keyframe(root, worldbody)

    def _finalize_model(self, root: ET.Element) -> str:
        """Serialize *root* and verify MJCF postconditions.

        Postcondition: returned string is well-formed MJCF XML whose root
        element is ``<mujoco>``.
        """
        ensure_mjcf_root(root)
        xml_str = serialize_model(root)
        return xml_str

    def build(self) -> str:
        """Build the complete MuJoCo MJCF model XML and return as string.

        Postcondition: returned string is well-formed MJCF XML with
        ``<mujoco>`` as the root element.
        """
        logger.info("Building %s model", self.exercise_name)

        root = self._create_root_element()
        worldbody = self._create_worldbody(root)
        equality = self._create_equality(root)

        self._build_bodies_and_barbell(worldbody, equality)
        self._post_worldbody_hook(worldbody, equality)
        self._add_contact_section(root)
        self._add_state_sections(root, worldbody)

        xml_str = self._finalize_model(root)
        logger.debug("Successfully built %s model", self.exercise_name)
        return xml_str
