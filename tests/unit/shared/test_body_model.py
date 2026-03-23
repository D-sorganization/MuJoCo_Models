"""Tests for the full-body musculoskeletal model."""

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.shared.body import BodyModelSpec, create_full_body


class TestBodyModelSpec:
    def test_defaults(self) -> None:
        spec = BodyModelSpec()
        assert spec.total_mass == 80.0
        assert spec.height == 1.75

    def test_custom_values(self) -> None:
        spec = BodyModelSpec(total_mass=100.0, height=1.90)
        assert spec.total_mass == 100.0
        assert spec.height == 1.90

    def test_rejects_zero_mass(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BodyModelSpec(total_mass=0.0)

    def test_rejects_negative_mass(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BodyModelSpec(total_mass=-1.0)

    def test_rejects_zero_height(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BodyModelSpec(height=0.0)

    def test_frozen(self) -> None:
        spec = BodyModelSpec()
        with pytest.raises(AttributeError):
            spec.total_mass = 90.0


class TestCreateFullBody:
    @pytest.fixture()
    def worldbody(self) -> ET.Element:
        return ET.Element("worldbody")

    def test_returns_dict_of_bodies(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        assert isinstance(bodies, dict)
        assert len(bodies) > 0

    def test_pelvis_exists(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        assert "pelvis" in bodies

    def test_torso_exists(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        assert "torso" in bodies

    def test_head_exists(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        assert "head" in bodies

    def test_bilateral_segments(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        for seg in ["upper_arm", "forearm", "hand", "thigh", "shank", "foot"]:
            assert f"{seg}_l" in bodies, f"Missing {seg}_l"
            assert f"{seg}_r" in bodies, f"Missing {seg}_r"

    def test_total_body_count(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        # 3 central + 6 bilateral pairs = 3 + 12 = 15
        assert len(bodies) == 15

    def test_pelvis_has_freejoint(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        pelvis = bodies["pelvis"]
        fj = pelvis.find("freejoint")
        assert fj is not None
        assert fj.get("name") == "ground_pelvis"

    def test_torso_has_hinge(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        torso = bodies["torso"]
        joint = torso.find("joint")
        assert joint is not None
        assert joint.get("type") == "hinge"
        assert joint.get("name") == "lumbar_flex"

    def test_all_bodies_have_inertial(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        for name, body in bodies.items():
            inertial = body.find("inertial")
            assert inertial is not None, f"{name} missing inertial"
            mass = float(inertial.get("mass"))
            assert mass > 0, f"{name} mass={mass}"

    def test_all_bodies_have_geom(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        for name, body in bodies.items():
            geom = body.find("geom")
            assert geom is not None, f"{name} missing geom"

    def test_default_spec_used(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody, spec=None)
        assert len(bodies) == 15

    def test_custom_spec(self, worldbody: ET.Element) -> None:
        spec = BodyModelSpec(total_mass=100.0, height=1.90)
        bodies = create_full_body(worldbody, spec=spec)
        pelvis_inertial = bodies["pelvis"].find("inertial")
        mass = float(pelvis_inertial.get("mass"))
        # Pelvis mass fraction is 0.142 of total
        assert mass == pytest.approx(100.0 * 0.142)

    def test_z_up_pelvis_position(self, worldbody: ET.Element) -> None:
        bodies = create_full_body(worldbody)
        pelvis = bodies["pelvis"]
        pos = pelvis.get("pos")
        parts = [float(x) for x in pos.split()]
        # Z should be ~0.93 (hip height), X and Y ~0
        assert parts[2] == pytest.approx(0.93)
        assert parts[0] == pytest.approx(0.0)
