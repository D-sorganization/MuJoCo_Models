# SPDX-License-Identifier: MIT
"""Tests for ground contact model: foot geometry, friction, and contact exclusions."""

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.exercises.bench_press.bench_press_model import (
    build_bench_press_model,
)
from mujoco_models.exercises.squat.squat_model import build_squat_model
from mujoco_models.shared.body import create_full_body
from mujoco_models.shared.body.body_helpers import _iter_foot_bodies


class TestFootContactGeometry:
    """Verify foot segments have proper contact geometry."""

    @pytest.fixture()
    def worldbody(self) -> ET.Element:
        return ET.Element("worldbody")

    @pytest.fixture()
    def bodies(self, worldbody: ET.Element) -> dict[str, ET.Element]:
        return create_full_body(worldbody)

    def test_foot_contact_geoms_exist(self, bodies: dict[str, ET.Element]) -> None:
        for side in ("l", "r"):
            foot = bodies[f"foot_{side}"]
            contact_geoms = [
                g
                for g in foot.findall("geom")
                if g.get("name") == f"foot_{side}_contact"
            ]
            assert len(contact_geoms) == 1, f"Missing contact geom for foot_{side}"

    def test_foot_contact_geom_is_box(self, bodies: dict[str, ET.Element]) -> None:
        for side in ("l", "r"):
            foot = bodies[f"foot_{side}"]
            contact = foot.find(f".//geom[@name='foot_{side}_contact']")
            assert contact is not None
            assert contact.get("type") == "box"

    def test_foot_contact_contype(self, bodies: dict[str, ET.Element]) -> None:
        for side in ("l", "r"):
            foot = bodies[f"foot_{side}"]
            contact = foot.find(f".//geom[@name='foot_{side}_contact']")
            assert contact is not None
            assert contact.get("contype") == "1"
            assert contact.get("conaffinity") == "1"
            assert contact.get("condim") == "3"

    def test_foot_contact_friction(self, bodies: dict[str, ET.Element]) -> None:
        for side in ("l", "r"):
            foot = bodies[f"foot_{side}"]
            contact = foot.find(f".//geom[@name='foot_{side}_contact']")
            assert contact is not None
            assert contact.get("friction") == "1.0 0.005 0.0001"

    def test_foot_contact_group(self, bodies: dict[str, ET.Element]) -> None:
        for side in ("l", "r"):
            foot = bodies[f"foot_{side}"]
            contact = foot.find(f".//geom[@name='foot_{side}_contact']")
            assert contact is not None
            assert contact.get("group") == "1"

    def test_foot_visual_geom_group(self, bodies: dict[str, ET.Element]) -> None:
        for side in ("l", "r"):
            foot = bodies[f"foot_{side}"]
            visual = foot.find(f".//geom[@name='foot_{side}_geom']")
            assert visual is not None
            assert visual.get("group") == "0"

    def test_foot_contact_dimensions(self, bodies: dict[str, ET.Element]) -> None:
        """Contact box should be approximately 0.26m x 0.10m x 0.02m (half-sizes)."""
        for side in ("l", "r"):
            foot = bodies[f"foot_{side}"]
            contact = foot.find(f".//geom[@name='foot_{side}_contact']")
            assert contact is not None
            size_parts = [float(s) for s in contact.get("size", "").split()]
            assert len(size_parts) == 3
            # Half-sizes: 0.13, 0.05, 0.01
            assert size_parts[0] == pytest.approx(0.13)
            assert size_parts[1] == pytest.approx(0.05)
            assert size_parts[2] == pytest.approx(0.01)

    def test_iter_foot_bodies_skips_missing_unilateral_feet(self) -> None:
        left = ET.Element("body", name="foot_l")
        assert _iter_foot_bodies({"foot_l": left}) == (("l", left),)


class TestContactExclusions:
    """Verify contact exclusion pairs between adjacent segments."""

    def _get_exclusions(self, xml_str: str) -> set[tuple[str, str]]:
        root = ET.fromstring(xml_str)
        contact = root.find("contact")
        assert contact is not None, "Missing <contact> section"
        excludes = contact.findall("exclude")
        return {(e.get("body1", ""), e.get("body2", "")) for e in excludes}

    def test_central_exclusions_exist(self) -> None:
        xml_str = build_squat_model()
        exclusions = self._get_exclusions(xml_str)
        assert ("pelvis", "torso") in exclusions
        assert ("torso", "head") in exclusions

    def test_bilateral_exclusions_exist(self) -> None:
        xml_str = build_squat_model()
        exclusions = self._get_exclusions(xml_str)
        for side in ("l", "r"):
            assert ("pelvis", f"thigh_{side}") in exclusions
            assert ("torso", f"upper_arm_{side}") in exclusions
            assert (f"upper_arm_{side}", f"forearm_{side}") in exclusions
            assert (f"forearm_{side}", f"hand_{side}") in exclusions
            assert (f"thigh_{side}", f"shank_{side}") in exclusions
            assert (f"shank_{side}", f"foot_{side}") in exclusions

    def test_exclusion_count(self) -> None:
        """Should have 2 central + 6 bilateral * 2 sides = 14 exclusions."""
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        contact = root.find("contact")
        assert contact is not None
        excludes = contact.findall("exclude")
        assert len(excludes) == 14


class TestGroundPlaneContact:
    """Verify ground plane has contact properties enabled."""

    def test_ground_has_contact_properties(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        assert worldbody is not None
        ground = worldbody.find("geom[@name='ground']")
        assert ground is not None
        assert ground.get("contype") == "1"
        assert ground.get("conaffinity") == "1"
        assert ground.get("condim") == "3"
        assert ground.get("friction") == "1.0 0.005 0.0001"


class TestContactSolverOptions:
    """Verify contact solver settings in the option element."""

    def test_implicit_integrator(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        assert option.get("integrator") == "implicit"

    def test_elliptic_cone(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        assert option.get("cone") == "elliptic"

    def test_newton_solver(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        assert option.get("solver") == "Newton"

    def test_tolerance(self) -> None:
        xml_str = build_squat_model()
        root = ET.fromstring(xml_str)
        option = root.find("option")
        assert option is not None
        assert option.get("tolerance") == "1e-8"


class TestBenchPressContact:
    """Verify bench press has proper contact geometry."""

    def test_bench_has_contact_geom(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        bench_geom = root.find(".//geom[@name='bench_contact']")
        assert bench_geom is not None
        assert bench_geom.get("type") == "box"
        assert bench_geom.get("contype") == "1"
        assert bench_geom.get("conaffinity") == "1"
        assert bench_geom.get("condim") == "3"
        assert bench_geom.get("friction") == "0.8 0.005 0.0001"

    def test_bench_has_foot_contact(self) -> None:
        """Bench press should still have foot contact geoms for feet on floor."""
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        for side in ("l", "r"):
            contact = root.find(f".//geom[@name='foot_{side}_contact']")
            assert contact is not None, f"Missing foot_{side}_contact in bench press"

    def test_bench_has_contact_exclusions(self) -> None:
        xml_str = build_bench_press_model()
        root = ET.fromstring(xml_str)
        contact = root.find("contact")
        assert contact is not None
        excludes = contact.findall("exclude")
        assert len(excludes) == 14
