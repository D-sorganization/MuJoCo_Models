"""Regression tests for the shared exercise builder base class."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises.base import ExerciseConfig, ExerciseModelBuilder
from mujoco_models.shared.barbell import BarbellSpec
from mujoco_models.shared.body import BodyModelSpec, segment_properties


class ForwardingBuilder(ExerciseModelBuilder):
    def __init__(self, config: ExerciseConfig | None = None) -> None:
        super().__init__(config)
        self.set_initial_pose_calls = 0

    @property
    def exercise_name(self) -> str:
        return "forwarding_builder"

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        self._attach_barbell_to_hands(equality, grip_offset=self.grip_offset)

    def set_initial_pose(self, worldbody: ET.Element) -> None:
        self.set_initial_pose_calls += 1


class NoBarbellBuilder(ForwardingBuilder):
    """Builder that opts out of barbell construction via uses_barbell=False."""

    @property
    def uses_barbell(self) -> bool:
        return False

    def attach_barbell(
        self,
        equality: ET.Element,
        body_bodies: dict[str, ET.Element],
        barbell_bodies: dict[str, ET.Element],
    ) -> None:
        raise AssertionError(
            "attach_barbell must not be called when uses_barbell is False"
        )


class OverriddenAccessorBuilder(ForwardingBuilder):
    def __init__(self) -> None:
        super().__init__(
            ExerciseConfig(
                body_spec=BodyModelSpec(total_mass=90.0, height=1.82),
                barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=25.0),
                gravity=(1.0, 2.0, -3.0),
            )
        )
        self._body_spec = BodyModelSpec(total_mass=95.0, height=1.90)
        self._barbell_spec = BarbellSpec.womens_olympic(plate_mass_per_side=10.0)
        self._gravity = (0.5, -0.25, -4.5)
        self._grip_offset = (0.010, 0.020, 0.030, 1.0, 0.0, 0.0, 0.0)

    @property
    def body_spec(self) -> BodyModelSpec:
        return self._body_spec

    @property
    def barbell_spec(self) -> BarbellSpec:
        return self._barbell_spec

    @property
    def gravity(self) -> tuple[float, float, float]:
        return self._gravity

    @property
    def grip_offset(self) -> tuple[float, ...] | None:
        return self._grip_offset


class TestExerciseModelBuilderAccessors:
    def test_accessors_forward_config(self) -> None:
        builder = ForwardingBuilder(ExerciseConfig())
        assert builder.body_spec == builder.config.body_spec
        assert builder.barbell_spec == builder.config.barbell_spec
        assert builder.gravity == builder.config.gravity
        assert builder.grip_offset is None

    def test_uses_barbell_default_true(self) -> None:
        """Default ExerciseModelBuilder.uses_barbell is True (backward compat)."""
        builder = ForwardingBuilder(ExerciseConfig())
        assert builder.uses_barbell is True

    def test_build_with_uses_barbell_true_has_real_mass_shaft(self) -> None:
        """With uses_barbell=True, the barbell shaft is built with real mass."""
        config = ExerciseConfig(
            barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=20.0)
        )
        xml_str = ForwardingBuilder(config).build()
        root = ET.fromstring(xml_str)
        shaft_inertial = root.find(".//body[@name='barbell_shaft']/inertial")
        assert shaft_inertial is not None
        shaft_mass = float(shaft_inertial.get("mass"))  # type: ignore[arg-type]
        assert shaft_mass > 0.0
        assert shaft_mass == pytest.approx(config.barbell_spec.shaft_mass)

    def test_build_with_uses_barbell_false_omits_barbell(self) -> None:
        """With uses_barbell=False, no barbell bodies or welds are emitted."""
        xml_str = NoBarbellBuilder(ExerciseConfig()).build()
        root = ET.fromstring(xml_str)

        body_names = {b.get("name") for b in root.findall(".//body")}
        assert "barbell_shaft" not in body_names
        assert "barbell_left_sleeve" not in body_names
        assert "barbell_right_sleeve" not in body_names

        weld_names = {w.get("name") for w in root.findall(".//weld")}
        assert "barbell_left_weld" not in weld_names
        assert "barbell_right_weld" not in weld_names

    def test_build_uses_accessor_overrides(self) -> None:
        builder = OverriddenAccessorBuilder()
        xml_str = builder.build()
        root = ET.fromstring(xml_str)

        option = root.find("option")
        assert option is not None
        assert option.get("gravity") == "0.500000 -0.250000 -4.500000"

        pelvis = root.find(".//body[@name='pelvis']/inertial")
        assert pelvis is not None
        expected_pelvis_mass, _, _ = segment_properties(
            builder.body_spec.total_mass, builder.body_spec.height, "pelvis"
        )
        assert float(pelvis.get("mass")) == pytest.approx(expected_pelvis_mass)  # type: ignore[arg-type]

        barbell_shaft = root.find(".//body[@name='barbell_shaft']/inertial")
        assert barbell_shaft is not None
        assert float(barbell_shaft.get("mass")) == pytest.approx(
            builder.barbell_spec.shaft_mass
        )  # type: ignore[arg-type]

        weld = root.find(".//weld[@name='barbell_to_hand_l']")
        assert weld is not None
        assert (
            weld.get("relpose")
            == "0.010000 0.020000 0.030000 1.000000 0.000000 0.000000 0.000000"
        )


class TestBuildDecomposition:
    """Tests for helpers extracted from ``build()`` in A-N Refresh 2026-04-14."""

    def test_build_bodies_and_barbell_with_barbell(self) -> None:
        """Both body and barbell bodies are populated when barbell is enabled."""
        builder = ForwardingBuilder(ExerciseConfig())
        worldbody = ET.Element("worldbody")
        equality = ET.Element("equality")

        body_bodies, barbell_bodies = builder._build_bodies_and_barbell(
            worldbody, equality
        )
        assert "pelvis" in body_bodies
        assert "barbell_shaft" in barbell_bodies
        # attach_barbell produced weld constraints
        weld_names = {w.get("name") for w in equality.findall("weld")}
        assert "barbell_to_hand_l" in weld_names
        assert "barbell_to_hand_r" in weld_names

    def test_build_bodies_and_barbell_without_barbell(self) -> None:
        """When ``uses_barbell`` is False, barbell_bodies is empty and no weld is added."""
        builder = NoBarbellBuilder(ExerciseConfig())
        worldbody = ET.Element("worldbody")
        equality = ET.Element("equality")

        body_bodies, barbell_bodies = builder._build_bodies_and_barbell(
            worldbody, equality
        )
        assert body_bodies  # body still built
        assert barbell_bodies == {}
        assert equality.findall("weld") == []

    def test_finalize_model_validates_mjcf_root(self) -> None:
        """``_finalize_model`` returns serialized XML whose root is ``<mujoco>``."""
        builder = ForwardingBuilder(ExerciseConfig())
        root = ET.Element("mujoco", model="dummy")
        ET.SubElement(root, "worldbody")
        xml_str = builder._finalize_model(root)
        assert "<mujoco" in xml_str
        assert ET.fromstring(xml_str).tag == "mujoco"

    def test_finalize_model_rejects_non_mujoco_root(self) -> None:
        """A non-``<mujoco>`` root violates the MJCF postcondition."""
        builder = ForwardingBuilder(ExerciseConfig())
        bad_root = ET.Element("notmujoco")
        with pytest.raises(ValueError, match="MJCF root"):
            builder._finalize_model(bad_root)

    def test_create_equality_adds_named_section(self) -> None:
        """The equality helper isolates constraint-section creation."""
        builder = ForwardingBuilder(ExerciseConfig())
        root = ET.Element("mujoco", model="dummy")
        equality = builder._create_equality(root)
        assert equality.tag == "equality"
        assert root.find("equality") is equality

    def test_add_contact_section_populates_exclusions(self) -> None:
        """The contact helper preserves adjacent-segment collision exclusions."""
        builder = ForwardingBuilder(ExerciseConfig())
        root = ET.Element("mujoco", model="dummy")
        contact = builder._add_contact_section(root)
        exclusion_names = {
            exclude.get("name") for exclude in contact.findall("exclude")
        }
        assert "exclude_pelvis_torso" in exclusion_names
        assert "exclude_thigh_l_shank_l" in exclusion_names
        assert "exclude_shank_r_foot_r" in exclusion_names

    def test_add_state_sections_runs_pose_actuator_sensor_and_keyframe_steps(
        self,
    ) -> None:
        """The state helper owns build steps that depend on final worldbody joints."""
        builder = ForwardingBuilder(ExerciseConfig())
        root = builder._create_root_element()
        worldbody = builder._create_worldbody(root)
        body = ET.SubElement(worldbody, "body", name="test_body", pos="1 2 3")
        ET.SubElement(body, "freejoint", name="test_free")
        ET.SubElement(body, "joint", name="test_hinge", type="hinge", ref="0.25")

        builder._add_state_sections(root, worldbody)

        assert builder.set_initial_pose_calls == 1
        assert root.find(".//position[@joint='test_hinge']") is not None
        assert root.find(".//jointpos[@joint='test_hinge']") is not None
        key = root.find(".//key")
        assert key is not None
        assert key.get("qpos") == "1 2 3 1 0 0 0 0.25"


class TestBarbellAttachmentHelpers:
    """Tests for the barbell-to-hands helpers extracted in A-N Refresh 2026-04-14."""

    def test_barbell_relpose_for_hand_uses_grip_offset(self) -> None:
        """An explicit grip offset should win over any grip width."""
        offset = (0.010, 0.020, 0.030, 1.0, 0.0, 0.0, 0.0)
        relpose = ForwardingBuilder._barbell_relpose_for_hand(
            "l", grip_width=0.4, grip_offset=offset
        )
        assert relpose == offset

    def test_barbell_relpose_for_hand_uses_width_per_side(self) -> None:
        """Grip width is converted into mirrored left/right weld offsets."""
        left = ForwardingBuilder._barbell_relpose_for_hand("l", grip_width=0.4)
        right = ForwardingBuilder._barbell_relpose_for_hand("r", grip_width=0.4)
        assert left == (0.4, 0, 0, 1, 0, 0, 0)
        assert right == (-0.4, 0, 0, 1, 0, 0, 0)

    def test_attach_barbell_to_hands_writes_welds_for_each_side(self) -> None:
        """The helper should still emit one weld per hand with the derived pose."""
        builder = ForwardingBuilder(ExerciseConfig())
        equality = ET.Element("equality")

        builder._attach_barbell_to_hands(equality, grip_width=0.4)

        welds = {w.get("name"): w for w in equality.findall("weld")}
        assert set(welds) == {"barbell_to_hand_l", "barbell_to_hand_r"}
        assert welds["barbell_to_hand_l"].get("relpose") == (
            "0.400000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000"
        )
        assert welds["barbell_to_hand_r"].get("relpose") == (
            "-0.400000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000"
        )
