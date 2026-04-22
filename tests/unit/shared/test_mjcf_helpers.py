"""Tests for MJCF XML generation helpers."""

import xml.etree.ElementTree as ET

import pytest

from mujoco_models.shared.utils.mjcf_helpers import (
    add_body,
    add_free_joint,
    add_hinge_joint,
    add_weld_constraint,
    diag_inertia_str,
    indent_xml,
    serialize_model,
    vec3_str,
)


class TestVec3Str:
    def test_basic(self) -> None:
        assert vec3_str(1.0, 2.0, 3.0) == "1.000000 2.000000 3.000000"

    def test_zeros(self) -> None:
        assert vec3_str(0, 0, 0) == "0.000000 0.000000 0.000000"

    def test_negative(self) -> None:
        result = vec3_str(-1.5, 0.0, 3.14)
        assert "-1.500000" in result
        assert "3.140000" in result


class TestDiagInertiaStr:
    def test_basic(self) -> None:
        result = diag_inertia_str(0.1, 0.2, 0.3)
        assert "0.100000" in result
        assert "0.200000" in result
        assert "0.300000" in result


class TestAddBody:
    @pytest.fixture()
    def worldbody(self) -> ET.Element:
        return ET.Element("worldbody")

    def test_creates_body_element(self, worldbody: ET.Element) -> None:
        body = add_body(
            worldbody,
            name="test",
            pos=(1, 2, 3),
            mass=5.0,
            inertia_diag=(0.1, 0.2, 0.3),
        )
        assert body.tag == "body"
        assert body.get("name") == "test"  # type: ignore

    def test_has_inertial(self, worldbody: ET.Element) -> None:
        body = add_body(
            worldbody,
            name="test",
            pos=(0, 0, 0),
            mass=5.0,
            inertia_diag=(0.1, 0.2, 0.3),
        )
        inertial = body.find("inertial")
        assert inertial is not None
        assert inertial.get("mass") == "5.000000"  # type: ignore

    def test_has_geom(self, worldbody: ET.Element) -> None:
        body = add_body(
            worldbody,
            name="test",
            pos=(0, 0, 0),
            mass=5.0,
            inertia_diag=(0.1, 0.2, 0.3),
            geom_type="box",
            geom_size=(0.1, 0.2, 0.3),
        )
        geom = body.find("geom")
        assert geom is not None
        assert geom.get("type") == "box"  # type: ignore
        assert geom.get("name") == "test_geom"  # type: ignore

    def test_custom_rgba(self, worldbody: ET.Element) -> None:
        body = add_body(
            worldbody,
            name="test",
            pos=(0, 0, 0),
            mass=1.0,
            inertia_diag=(0.1, 0.1, 0.1),
            geom_rgba="1 0 0 1",
        )
        geom = body.find("geom")
        assert geom.get("rgba") == "1 0 0 1"  # type: ignore

    def test_geom_euler_preserves_three_values(self, worldbody: ET.Element) -> None:
        body = add_body(
            worldbody,
            name="test",
            pos=(0, 0, 0),
            mass=1.0,
            inertia_diag=(0.1, 0.1, 0.1),
            geom_euler=(0.1, 0.2, 0.3),
        )
        geom = body.find("geom")
        assert geom.get("euler") == "0.100000 0.200000 0.300000"  # type: ignore

    def test_geom_euler_preserves_additional_values(self, worldbody: ET.Element) -> None:
        body = add_body(
            worldbody,
            name="test",
            pos=(0, 0, 0),
            mass=1.0,
            inertia_diag=(0.1, 0.1, 0.1),
            geom_euler=(0.1, 0.2, 0.3, 0.4),
        )
        geom = body.find("geom")
        assert geom.get("euler") == "0.100000 0.200000 0.300000 0.400000"  # type: ignore

    def test_geom_size_none(self, worldbody: ET.Element) -> None:
        body = add_body(
            worldbody,
            name="test",
            pos=(0, 0, 0),
            mass=1.0,
            inertia_diag=(0.1, 0.1, 0.1),
            geom_size=None,
        )
        geom = body.find("geom")
        assert geom.get("size") is None  # type: ignore

    def test_position_attribute(self, worldbody: ET.Element) -> None:
        body = add_body(
            worldbody,
            name="test",
            pos=(1.5, -2.0, 3.0),
            mass=1.0,
            inertia_diag=(0.1, 0.1, 0.1),
        )
        assert "1.500000" in body.get("pos")  # type: ignore


class TestAddHingeJoint:
    def test_creates_hinge(self) -> None:
        body = ET.Element("body")
        joint = add_hinge_joint(body, name="hip", axis=(1, 0, 0))
        assert joint.get("type") == "hinge"  # type: ignore
        assert joint.get("name") == "hip"  # type: ignore

    def test_axis_attribute(self) -> None:
        body = ET.Element("body")
        joint = add_hinge_joint(body, name="knee", axis=(0, 1, 0))
        assert "0.000000 1.000000 0.000000" in joint.get("axis")  # type: ignore

    def test_range_attribute(self) -> None:
        body = ET.Element("body")
        joint = add_hinge_joint(body, name="test", range_min=-1.0, range_max=2.0)
        assert "-1.0000" in joint.get("range")  # type: ignore
        assert "2.0000" in joint.get("range")  # type: ignore

    def test_default_range(self) -> None:
        body = ET.Element("body")
        joint = add_hinge_joint(body, name="test")
        assert joint.get("range") is not None  # type: ignore


class TestAddFreeJoint:
    def test_creates_freejoint(self) -> None:
        body = ET.Element("body")
        fj = add_free_joint(body, name="root")
        assert fj.tag == "freejoint"
        assert fj.get("name") == "root"  # type: ignore


class TestAddWeldConstraint:
    @pytest.fixture()
    def equality(self) -> ET.Element:
        return ET.Element("equality")

    def test_creates_weld(self, equality: ET.Element) -> None:
        weld = add_weld_constraint(equality, name="w1", body1="a", body2="b")
        assert weld.tag == "weld"
        assert weld.get("body1") == "a"  # type: ignore
        assert weld.get("body2") == "b"  # type: ignore

    def test_with_relpose(self, equality: ET.Element) -> None:
        weld = add_weld_constraint(
            equality,
            name="w2",
            body1="a",
            body2="b",
            relpose=(0.1, 0.2, 0.3, 1, 0, 0, 0),
        )
        assert weld.get("relpose") is not None  # type: ignore
        assert "0.100000" in weld.get("relpose")  # type: ignore

    def test_without_relpose(self, equality: ET.Element) -> None:
        weld = add_weld_constraint(equality, name="w3", body1="a", body2="b")
        assert weld.get("relpose") is None  # type: ignore


class TestIndentXml:
    def test_indents_children(self) -> None:
        root = ET.Element("mujoco")
        ET.SubElement(root, "option")
        indent_xml(root)
        xml_str = ET.tostring(root, encoding="unicode")
        assert "\n" in xml_str

    def test_empty_element(self) -> None:
        root = ET.Element("mujoco")
        indent_xml(root)
        assert root.tail == "\n"


class TestSerializeModel:
    def test_returns_xml_string(self) -> None:
        root = ET.Element("mujoco", model="test")
        result = serialize_model(root)
        assert "<?xml" in result
        assert "mujoco" in result

    def test_well_formed(self) -> None:
        root = ET.Element("mujoco")
        ET.SubElement(root, "option", gravity="0 0 -9.81")
        result = serialize_model(root)
        parsed = ET.fromstring(result)
        assert parsed.tag == "mujoco"
