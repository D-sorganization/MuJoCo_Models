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

from mujoco_models.exercises import spec_start
from mujoco_models.exercises.config import ExerciseConfig
from mujoco_models.exercises.grip_helpers import (
    _IDENTITY_QUAT,
    HandQuat,
    _hand_quat_for_side,
    _quat_rotate_vector,
    attach_barbell_to_hand,
    attach_barbell_to_hands,
    barbell_relpose_for_hand,
)
from mujoco_models.exercises.model_scaffold import (
    add_contact_section,
    add_state_sections,
    create_root_element,
    create_worldbody,
    set_ref_by_name_map,
)
from mujoco_models.shared.barbell import BarbellSpec, create_barbell_bodies
from mujoco_models.shared.body import BodyModelSpec, create_full_body
from mujoco_models.shared.body.foot_seating import (
    foot_seating_offset,
    lower_foot_contact_geoms,
)
from mujoco_models.shared.contracts.postconditions import ensure_mjcf_root
from mujoco_models.shared.utils.mjcf_helpers import serialize_model

logger = logging.getLogger(__name__)


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

        Default: none. ``ref`` only relabels a joint's zero angle and never
        moves a segment (#407), so a spec lift keeps every ref at 0 and holds
        its start pose in the keyframe (:meth:`keyframe_angle_offsets`).
        """

    def keyframe_angle_offsets(self) -> dict[str, float]:
        """Joint name -> radians added to the joint's ``ref`` in the keyframe.

        A joint's ``ref`` is the value of its zero angle: the XML geometry is the
        body pose at ``qpos = ref``, so ``ref`` alone never moves a segment. An
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
        """

    _barbell_relpose_for_hand = staticmethod(barbell_relpose_for_hand)
    _attach_barbell_to_hand = staticmethod(attach_barbell_to_hand)

    def _attach_barbell_to_hands(
        self,
        equality: ET.Element,
        *,
        grip_width: float | None = None,
        grip_offset: tuple[float, ...] | None = None,
        hand_quat: HandQuat | Mapping[str, HandQuat] = _IDENTITY_QUAT,
    ) -> None:
        """Weld barbell shaft to both hands (DRY helper for subclasses)."""
        attach_barbell_to_hands(
            equality,
            grip_width=grip_width,
            grip_offset=grip_offset,
            hand_quat=hand_quat,
        )

    set_ref_by_name_map = staticmethod(set_ref_by_name_map)

    def _create_root_element(self) -> ET.Element:
        """Create the ``<mujoco>`` root with option, compiler, and defaults."""
        return create_root_element(self.exercise_name, self.gravity)

    def _create_worldbody(self, root: ET.Element) -> ET.Element:
        """Create worldbody with ground plane."""
        return create_worldbody(root)

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
        return add_contact_section(root)

    def _add_state_sections(self, root: ET.Element, worldbody: ET.Element) -> None:
        """Apply initial pose metadata, actuators, sensors, and keyframe data."""
        add_state_sections(
            root,
            worldbody,
            self.exercise_name,
            self.keyframe_angle_offsets(),
            self.set_initial_pose,
        )

    def _finalize_model(self, root: ET.Element) -> str:
        """Serialize *root* and verify MJCF postconditions.

        Postcondition: returned string is well-formed MJCF XML whose root
        element is ``<mujoco>``.
        """
        ensure_mjcf_root(root)
        return serialize_model(root)

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


__all__ = [
    "ExerciseConfig",
    "ExerciseModelBuilder",
    "HandQuat",
    "_IDENTITY_QUAT",
    "_hand_quat_for_side",
    "_quat_rotate_vector",
    "set_ref_by_name_map",
]
