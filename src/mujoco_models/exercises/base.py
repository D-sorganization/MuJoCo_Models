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
import math
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mujoco_models.exercises.contact_policy import (
    _GROUND_FRICTION,
    _SERVO_FORCERANGE,
    _SERVO_KP,
    _SERVO_KV,
    _add_contact_exclusions,
)
from mujoco_models.shared.barbell import BarbellSpec, create_barbell_bodies
from mujoco_models.shared.body import BodyModelSpec, create_full_body
from mujoco_models.shared.body.axes import SIDE_SIGN
from mujoco_models.shared.body.segment_data import SHOULDER_ADDUCT_MIN
from mujoco_models.shared.contact_masks import (
    FLOOR_MASKS,
    HUMAN_MASKS,
    apply_masks,
)
from mujoco_models.shared.contracts.postconditions import ensure_mjcf_root
from mujoco_models.shared.contracts.preconditions import require_non_negative
from mujoco_models.shared.utils.mjcf_helpers import (
    add_weld_constraint,
    serialize_model,
)

logger = logging.getLogger(__name__)

# Shared initial angles for floor-pull exercises (deadlift, snatch, clean-and-jerk).
# Defined here so subclasses can import from a single authoritative location.
FLOOR_PULL_HIP_FLEX: float = 1.3963  # ~80° hip flexion (radians)
FLOOR_PULL_KNEE_FLEX: float = -1.0472  # ~60° knee flexion (radians)

_IDENTITY_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


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
    def barbell_start_pos(self) -> tuple[float, float, float]:
        """World position of the barbell shaft centre at the keyframe.

        Default: at hand height on the body midline (X = Y = 0), so hand welds
        are consistent at the start pose.  Subclasses (squat, bench) override.
        """
        return (0.0, 0.0, self.body_spec.hand_height)

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

    def keyframe_angle_offsets(self) -> dict[str, float]:
        """Joint name -> radians added to the joint's ``ref`` in the keyframe.

        A joint's ``ref`` is the value of its zero angle: the XML geometry is the
        body pose at ``qpos = ref``, so ``ref`` alone never moves a segment.  An
        exercise whose start pose really is posed (bench press) lists the pose
        angle here; the keyframe ``qpos`` and ``ctrl`` hold ``ref + offset``.
        """
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
        hand_quat: tuple[float, float, float, float] = _IDENTITY_QUAT,
    ) -> tuple[float, ...] | None:
        """Return the weld relpose (bar relative to the hand frame) for one hand.

        ``grip_offset`` wins when provided.  Otherwise ``grip_width`` becomes a
        lateral (Y) offset anchored at the hand: the left hand is at +Y, so the
        bar centre sits at ``-grip_width`` in its frame and at ``+grip_width``
        in the right hand's.  ``hand_quat`` is the bar orientation in the hand
        frame (identity when the hand frame is world-aligned); it must be a
        rotation about Y so the lateral offset is unchanged.
        """
        if grip_offset is not None:
            return grip_offset
        if grip_width is None:
            return None

        return (0, -SIDE_SIGN[side] * grip_width, 0, *hand_quat)

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
        hand_quat: tuple[float, float, float, float] = _IDENTITY_QUAT,
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
        hand_quat : tuple
            Bar orientation in the hand frame, for exercises whose hands are
            not world-aligned at the keyframe (see ``_barbell_relpose_for_hand``).
        """
        for side in ("l", "r"):
            relpose = self._barbell_relpose_for_hand(
                side,
                grip_width=grip_width,
                grip_offset=grip_offset,
                hand_quat=hand_quat,
            )
            self._attach_barbell_to_hand(equality, side=side, relpose=relpose)

    # ------------------------------------------------------------------
    # Grip-width pose helpers (issue #408)
    #
    # A hand's lateral (Y) position at the initial pose is fixed at the
    # body's shoulder_half_width unless the shoulder is actually abducted.
    # A grip weld's relpose assumes the hand already sits at the exercise's
    # documented grip width, so a grip wider than shoulder_half_width needs
    # real shoulder abduction in the start pose -- not a wider weld offset --
    # or the two grip welds start violated by the gap between the two.
    # ------------------------------------------------------------------

    def _grip_abduction_angle(self, grip_width: float) -> float:
        """Shoulder-abduction magnitude (radians) that spreads both hands to
        ``grip_width``, clamped to the shoulder's own range of motion.

        Precondition: ``grip_width`` is non-negative.
        Postcondition: the returned angle lies in ``[0, abs(SHOULDER_ADDUCT_MIN)]``.
        A grip no wider than the body's natural shoulder width needs no
        abduction and returns 0.
        """
        require_non_negative(grip_width, "grip_width")
        needed = grip_width - self.body_spec.shoulder_half_width
        if needed <= 0.0:
            return 0.0
        max_angle = abs(SHOULDER_ADDUCT_MIN)
        return min(
            math.asin(min(needed / self.body_spec.arm_reach_length, 1.0)), max_angle
        )

    def _achieved_grip_width(self, grip_width: float) -> float:
        """Lateral hand distance the pose in :meth:`_grip_pose_offsets` reaches.

        Mirrors that method's range-of-motion clamp, so a grip weld built
        from this value matches the hand's true kinematic position exactly
        instead of assuming the exercise's nominal ``grip_width`` was reached.
        """
        angle = self._grip_abduction_angle(grip_width)
        if angle == 0.0:
            return self.body_spec.shoulder_half_width
        return (
            self.body_spec.shoulder_half_width
            + self.body_spec.arm_reach_length * math.sin(angle)
        )

    def _grip_pose_offsets(self, grip_width: float) -> dict[str, float]:
        """Keyframe angle offsets that abduct both shoulders toward ``grip_width``.

        Wrist deviation cancels the shoulder abduction's tilt (mirrored axis,
        equal magnitude) so the hands -- and the bar welded to them -- stay
        level. Returns ``{}`` when ``grip_width`` needs no abduction.
        """
        angle = self._grip_abduction_angle(grip_width)
        if angle == 0.0:
            return {}
        return {
            "shoulder_l_adduct": -angle,
            "shoulder_r_adduct": -angle,
            "wrist_l_deviate": angle,
            "wrist_r_deviate": angle,
        }

    def _grip_vertical_rise(self, grip_width: float) -> float:
        """Height the hand gains when the shoulder abducts toward ``grip_width``.

        An abducted arm no longer hangs straight down, so the hand sits this
        much above the straight-arm ``hand_height``; a caller welding the bar
        to hands abducted this way must raise ``barbell_start_pos`` by the
        same amount, or the grip weld starts with a vertical residual
        (MuJoCo_Models#408).
        """
        angle = self._grip_abduction_angle(grip_width)
        return self.body_spec.arm_reach_length * (1.0 - math.cos(angle))

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
        logger.debug("Successfully built %s model", self.exercise_name)
        return xml_str
