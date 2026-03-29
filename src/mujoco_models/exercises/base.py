"""Base exercise model builder for MuJoCo MJCF.

DRY: All five exercises share the same skeleton for creating a MuJoCo
MJCF model XML -- differing only in barbell attachment strategy, initial
pose, and joint coordinate defaults. This base class encapsulates the
shared workflow; subclasses override hooks to customize.

Law of Demeter: Exercise builders interact with BarbellSpec and BodyModelSpec
through their public APIs, never reaching into internal segment tables.
"""

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

    def _post_worldbody_hook(  # noqa: B027
        self, worldbody: ET.Element, equality: ET.Element
    ) -> None:
        """No-op hook called after worldbody is built, before actuator generation.

        Subclasses may override to inject additional bodies or constraints
        (e.g. BenchPressModelBuilder adds the bench body here).
        This is an intentional empty method — not abstract — so the base class
        remains instantiable and subclasses are not required to override it.
        """

    def _attach_barbell_to_hands(
        self,
        equality: ET.Element,
        *,
        grip_width: float | None = None,
    ) -> None:
        """Weld barbell shaft to both hands (DRY helper for subclasses).

        Parameters
        ----------
        equality : ET.Element
            The ``<equality>`` section of the MJCF model.
        grip_width : float or None
            Distance from the barbell shaft center to each hand along the X-axis.
            If None, the initial pose determines the grip width.
        """
        for side, sign in [("l", -1.0), ("r", 1.0)]:
            relpose: tuple[float, ...] | None = None
            if grip_width is not None:
                # Left hand (side="l", sign=-1) is at -X relative to barbell center.
                # relpose defines barbell center relative to hand, so it's at +X.
                relpose = (-sign * grip_width, 0, 0, 1, 0, 0, 0)

            add_weld_constraint(
                equality,
                name=f"barbell_to_hand_{side}",
                body1=f"hand_{side}",
                body2="barbell_shaft",
                relpose=relpose,
            )

    def build(self) -> str:
        """Build the complete MuJoCo MJCF model XML and return as string.

        Postcondition: returned string is well-formed MJCF XML with
        ``<mujoco>`` as the root element.
        """
        logger.info("Building %s model", self.exercise_name)

        root = ET.Element("mujoco", model=self.exercise_name)

        # Option: gravity, timestep, and contact solver settings
        g = self.config.gravity
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

        # Compiler settings (Z-up). coordinate='local' was removed in MuJoCo 2.1.4.
        ET.SubElement(root, "compiler", angle="radian")

        # Default joint damping
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "joint", damping="5.0", armature="0.1")
        ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3")

        # Worldbody
        worldbody = ET.SubElement(root, "worldbody")

        # Ground plane with contact properties
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

        # Subclass hook: inject extra bodies/constraints before actuator generation
        self._post_worldbody_hook(worldbody, equality)

        # Add contact exclusion pairs to prevent unrealistic self-collision
        # between adjacent body segments.
        contact = ET.SubElement(root, "contact")
        _add_contact_exclusions(contact)

        # Exercise-specific initial pose
        self.set_initial_pose(worldbody)

        # Add position actuators for all hinge joints.
        # Note: worldbody.iter("joint") only finds <joint> elements, never
        # <freejoint> elements, so no freejoint guard is needed here.
        actuator = ET.SubElement(root, "actuator")
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            if name:
                act = ET.SubElement(actuator, "position")
                act.set("name", f"act_{name}")
                act.set("joint", name)
                act.set("kp", "100")

        # Add joint position sensors.
        # Note: worldbody.iter("joint") only finds <joint> elements, never
        # <freejoint> elements, so no freejoint guard is needed here.
        sensor = ET.SubElement(root, "sensor")
        for joint in worldbody.iter("joint"):
            name = joint.get("name", "")
            if name:
                s = ET.SubElement(sensor, "jointpos")
                s.set("name", f"pos_{name}")
                s.set("joint", name)

        # Build keyframe from joint ref values set by set_initial_pose().
        # MuJoCo keyframes allow resetting the model to a named configuration
        # (e.g. via mj_resetDataKeyframe). The qpos order follows: freejoint
        # (7 DOF: xyz + quaternion), then hinge joints in tree order.
        qpos_values: list[str] = []
        for joint_el in worldbody.iter("joint"):
            ref_val = joint_el.get("ref", "0")
            qpos_values.append(ref_val)
        # Prepend freejoint DOFs (7 values: x y z qw qx qy qz)
        for fj in worldbody.iter("freejoint"):
            # Find parent body to get initial position
            for body_el in worldbody.iter("body"):
                fj_in_body = body_el.find("freejoint")
                if fj_in_body is not None and fj_in_body is fj:
                    pos_str = body_el.get("pos", "0 0 0")
                    pos_parts = pos_str.split()
                    # freejoint qpos: x y z qw qx qy qz
                    fj_qpos = pos_parts + ["1", "0", "0", "0"]
                    qpos_values = fj_qpos + qpos_values
                    break

        if qpos_values:
            keyframe = ET.SubElement(root, "keyframe")
            key = ET.SubElement(keyframe, "key")
            key.set("name", f"{self.exercise_name}_start")
            key.set("qpos", " ".join(qpos_values))

        xml_str = serialize_model(root)

        # Postcondition: well-formed MJCF XML
        ensure_mjcf_root(xml_str)

        logger.debug("Successfully built %s model", self.exercise_name)
        return xml_str
