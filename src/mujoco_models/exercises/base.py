# SPDX-License-Identifier: MIT
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
from collections.abc import Mapping
from dataclasses import dataclass, field

from mujoco_models.exercises import spec_start
from mujoco_models.exercises.contact_policy import (
    _GROUND_FRICTION,
    _SERVO_FORCERANGE,
    _SERVO_KP,
    _SERVO_KV,
    _add_barbell_exclusions,
    _add_contact_exclusions,
)
from mujoco_models.shared.barbell import BarbellSpec, create_barbell_bodies
from mujoco_models.shared.body import BodyModelSpec, create_full_body
from mujoco_models.shared.body.axes import SIDE_SIGN
from mujoco_models.shared.body.foot_seating import (
    foot_seating_offset,
    lower_foot_contact_geoms,
)
from mujoco_models.shared.contact_masks import (
    FLOOR_MASKS,
    HUMAN_MASKS,
    apply_masks,
)
from mujoco_models.shared.contracts.postconditions import ensure_mjcf_root
from mujoco_models.shared.utils.mjcf_helpers import (
    add_weld_constraint,
    serialize_model,
)

logger = logging.getLogger(__name__)

HandQuat = tuple[float, float, float, float]

_IDENTITY_QUAT: HandQuat = (1.0, 0.0, 0.0, 0.0)


def _quat_rotate_vector(
    quat: HandQuat, vec: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate *vec* by *quat* (``w, x, y, z``): the quaternion sandwich product.

    Uses the standard expansion ``v' = v + 2w(q_xyz x v) + 2 q_xyz x (q_xyz x v)``
    so callers never need a separate rotation-matrix path for a quaternion
    they already hold (MuJoCo_Models#440).
    """
    w, qx, qy, qz = quat
    vx, vy, vz = vec
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w * tx + (qy * tz - qz * ty),
        vy + w * ty + (qz * tx - qx * tz),
        vz + w * tz + (qx * ty - qy * tx),
    )


def _hand_quat_for_side(
    hand_quat: HandQuat | Mapping[str, HandQuat], side: str
) -> HandQuat:
    """Resolve a hand-orientation argument that may be per-side.

    ``hand_quat`` is either a single quaternion applied to both hands
    (bench press: both hands share the same tilt) or a mapping with one
    quaternion per side (grip exercises, where shoulder abduction tilts
    the two hands oppositely, MuJoCo_Models#440).
    """
    if isinstance(hand_quat, Mapping):
        return hand_quat[side]
    return hand_quat


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
      - spec_exercise (spec lifts) or set_initial_pose()/keyframe_angle_offsets()
        for the start position
    """

    def __init__(self, config: ExerciseConfig | None = None) -> None:
        """Initialize with an optional exercise configuration.

        Args:
            config: Exercise configuration. Uses default ``ExerciseConfig``
                (50th-percentile male anthropometrics, men's Olympic barbell)
                if ``None``.
        """
        self.config = config or self.default_config()

    #: Key of the standard exercise whose ``start`` via-pose this model starts
    #: at (UpstreamDrift#11743); ``None`` for movements the spec does not pose.
    spec_exercise: str | None = None

    @classmethod
    def default_config(cls) -> ExerciseConfig:
        """Config used when none is given: the spec load for a spec lift."""
        if cls.spec_exercise is None:
            return ExerciseConfig()
        plates = spec_start.plates_per_side_kg(cls.spec_exercise)
        return ExerciseConfig(barbell_spec=BarbellSpec.mens_olympic(plates))

    @property
    def start_pose(self) -> spec_start.StartPose:
        """The spec start via-pose, placed for this body's height.

        Raises:
            ValueError: the exercise has no ``spec_exercise``.
        """
        if self.spec_exercise is None:
            raise ValueError(f"{self.exercise_name} has no spec start via-pose")
        return spec_start.start_pose(self.spec_exercise, self.body_spec.height)

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
    def barbell_start_pos(self) -> tuple[float, float, float]:
        """World position of the barbell shaft centre at the keyframe.

        A spec lift starts the bar at its via-pose ``bar_position_m``.
        Otherwise: at hand height on the body midline (X = Y = 0).
        """
        if self.spec_exercise is not None:
            return self.start_pose.bar_position
        return (0.0, 0.0, self.body_spec.hand_height)

    @property
    def grip_offset(self) -> tuple[float, ...] | None:
        """Optional grip offset for subclasses that need a custom weld pose."""
        return None

    @property
    @abstractmethod
    def exercise_name(self) -> str:
        """Human-readable exercise name used in the model XML."""

    #: Seat the foot boxes at the soft-contact equilibrium depth (issue #447).
    #: True for models that stand on the floor and start in static balance;
    #: see :mod:`mujoco_models.shared.body.foot_seating`.
    seat_feet_at_equilibrium: bool = False

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

    def set_initial_pose(self, worldbody: ET.Element) -> None:  # noqa: B027
        """Set joint ``ref`` values for the starting position.

        Default: none.  ``ref`` only relabels a joint's zero angle and never
        moves a segment (#407), so a spec lift keeps every ref at 0 and holds
        its start pose in the keyframe (:meth:`keyframe_angle_offsets`).
        """

    def keyframe_angle_offsets(self) -> dict[str, float]:
        """Joint name -> radians added to the joint's ``ref`` in the keyframe.

        A joint's ``ref`` is the value of its zero angle: the XML geometry is the
        body pose at ``qpos = ref``, so ``ref`` alone never moves a segment.  An
        exercise whose start pose really is posed (bench press) lists the pose
        angle here; the keyframe ``qpos`` and ``ctrl`` hold ``ref + offset``.
        A spec lift's offsets are its start via-pose coordinates.
        """
        if self.spec_exercise is not None:
            return dict(self.start_pose.coordinates)
        return {}

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
        hand_quat: HandQuat = _IDENTITY_QUAT,
    ) -> tuple[float, ...] | None:
        """Return the weld relpose (bar relative to the hand frame) for one hand.

        ``grip_offset`` wins when provided.  Otherwise ``grip_width`` sizes a
        lateral (Y) offset in the WORLD frame: the left hand is at +Y, so the
        bar centre sits at ``-grip_width`` in the world Y direction from it,
        and at ``+grip_width`` from the right hand's.  That world offset is
        then rotated into the hand's own frame by ``hand_quat`` -- the bar's
        orientation relative to the hand -- so a tilted hand (shoulder
        abduction without a counter-rotating wrist, MuJoCo_Models#440) still
        gets a relpose that keeps the bar at its intended world position.
        ``hand_quat`` defaults to identity for a world-aligned hand.
        """
        if grip_offset is not None:
            return grip_offset
        if grip_width is None:
            return None

        world_offset = (0.0, -SIDE_SIGN[side] * grip_width, 0.0)
        pos = _quat_rotate_vector(hand_quat, world_offset)
        return (*pos, *hand_quat)

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
        hand_quat: HandQuat | Mapping[str, HandQuat] = _IDENTITY_QUAT,
    ) -> None:
        """Weld barbell shaft to both hands (DRY helper for subclasses).

        Parameters
        ----------
        equality : ET.Element
            The ``<equality>`` section of the MJCF model.
        grip_width : float or None
            Distance from the barbell shaft center to each hand along the Y-axis
            (the shaft's own axis).  If None, the initial pose determines it.
        grip_offset : tuple or None
            Optional 7-element relative pose (x y z qw qx qy qz) for the grip
            offset from the hand to the barbell shaft.
        hand_quat : tuple or mapping
            Bar orientation in the hand frame, for exercises whose hands are
            not world-aligned at the keyframe (see ``_barbell_relpose_for_hand``).
            Either one quaternion shared by both hands (bench press), or a
            ``{"l": ..., "r": ...}`` mapping for hands tilted oppositely
            (MuJoCo_Models#440).  With neither ``grip_width`` nor
            ``grip_offset`` a spec lift's welds get their ``relpose`` from the
            posed keyframe (see ``spec_start.weld_bar_at_keyframe``).
        """
        for side in ("l", "r"):
            relpose = self._barbell_relpose_for_hand(
                side,
                grip_width=grip_width,
                grip_offset=grip_offset,
                hand_quat=_hand_quat_for_side(hand_quat, side),
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
        # Human geoms never collide with each other; equipment opts in via masks.
        apply_masks(ET.SubElement(default, "geom", condim="3"), HUMAN_MASKS)

        return root

    def _create_worldbody(self, root: ET.Element) -> ET.Element:
        """Create worldbody with ground plane."""
        worldbody = ET.SubElement(root, "worldbody")
        ground = ET.SubElement(
            worldbody,
            "geom",
            name="ground",
            type="plane",
            size="5 5 0.1",
            rgba="0.9 0.9 0.9 1",
            condim="3",
            friction=_GROUND_FRICTION,
        )
        apply_masks(ground, FLOOR_MASKS)
        return worldbody

    def _build_bodies_and_barbell(
        self, worldbody: ET.Element, equality: ET.Element
    ) -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
        """Build body segments and (optionally) barbell inside the worldbody.

        Returns ``(body_bodies, barbell_bodies)`` where ``barbell_bodies`` is
        empty when :attr:`uses_barbell` is ``False``.
        """
        body_bodies = create_full_body(worldbody, self.body_spec)
        if self.spec_exercise is not None:
            spec_start.place_start_pose(worldbody, self.start_pose)
        barbell_bodies: dict[str, ET.Element] = {}
        if self.uses_barbell:
            barbell_bodies = create_barbell_bodies(
                worldbody, equality, self.barbell_spec, pos=self.barbell_start_pos
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
        worldbody = root.find("worldbody")
        if worldbody is not None:
            _add_barbell_exclusions(contact, worldbody)
        return contact

    def _add_state_sections(self, root: ET.Element, worldbody: ET.Element) -> None:
        """Apply initial pose metadata, actuators, sensors, and keyframe data."""
        self.set_initial_pose(worldbody)

        actuator = ET.SubElement(root, "actuator")
        sensor = ET.SubElement(root, "sensor")

        qpos_values: list[str] = []
        ctrl_values: list[str] = []
        offsets = self.keyframe_angle_offsets()

        # Single document-order (depth-first) traversal: MuJoCo orders qpos by
        # the kinematic tree, so every freejoint (pelvis AND each barbell body)
        # contributes its 7 values exactly where its body appears.
        for el in worldbody.iter():
            tag = el.tag
            if tag == "joint":
                name = el.get("name", "")
                ref = el.get("ref", "0")
                if name in offsets:
                    ref = repr(float(ref) + offsets[name])
                if name:
                    ET.SubElement(
                        actuator,
                        "position",
                        name=f"act_{name}",
                        joint=name,
                        kp=f"{_SERVO_KP:g}",
                        kv=f"{_SERVO_KV:g}",
                        forcerange=f"{-_SERVO_FORCERANGE:g} {_SERVO_FORCERANGE:g}",
                    )
                    ET.SubElement(sensor, "jointpos", name=f"pos_{name}", joint=name)
                    ctrl_values.append(ref)
                qpos_values.append(ref)
            elif tag == "body" and el.find("freejoint") is not None:
                qpos_values.extend(
                    el.get("pos", "0 0 0").split() + el.get("quat", "1 0 0 0").split()
                )

        if qpos_values:
            keyframe = ET.SubElement(root, "keyframe")
            # ctrl = pose so the position servos hold the keyframe instead of
            # springing every joint to 0.
            ET.SubElement(
                keyframe,
                "key",
                name=f"{self.exercise_name}_start",
                qpos=" ".join(qpos_values),
                ctrl=" ".join(ctrl_values),
            )

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
        if self.spec_exercise is not None and self.uses_barbell:
            spec_start.weld_bar_at_keyframe(root, xml_str)
            xml_str = self._finalize_model(root)
        if self.seat_feet_at_equilibrium:
            lower_foot_contact_geoms(root, foot_seating_offset(xml_str))
            xml_str = self._finalize_model(root)
        logger.debug("Successfully built %s model", self.exercise_name)
        return xml_str
