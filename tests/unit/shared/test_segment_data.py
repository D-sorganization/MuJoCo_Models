"""Tests for segment data integrity and anthropometric correctness (#2).

Validates that the Winter (2009) segment table is self-consistent:
mass fractions sum to ~1.0, all fractions are positive, and the
segment_properties function returns physically plausible values.
"""

import pytest

from mujoco_models.shared.body.segment_data import (
    SEGMENT_TABLE,
    segment_properties,
    total_mass_fraction,
)


class TestSegmentTable:
    def test_all_segments_present(self) -> None:
        expected = {
            "pelvis",
            "torso",
            "head",
            "upper_arm",
            "forearm",
            "hand",
            "thigh",
            "shank",
            "foot",
        }
        assert set(SEGMENT_TABLE.keys()) == expected

    def test_all_mass_fractions_positive(self) -> None:
        for name, props in SEGMENT_TABLE.items():
            assert props["mass_frac"] > 0, f"{name} mass_frac not positive"

    def test_all_length_fractions_positive(self) -> None:
        for name, props in SEGMENT_TABLE.items():
            assert props["length_frac"] > 0, f"{name} length_frac not positive"

    def test_all_radius_fractions_positive(self) -> None:
        for name, props in SEGMENT_TABLE.items():
            assert props["radius_frac"] > 0, f"{name} radius_frac not positive"

    def test_total_mass_fraction_near_one(self) -> None:
        total = total_mass_fraction()
        assert total == pytest.approx(1.0, abs=0.02)

    def test_segment_has_three_keys(self) -> None:
        for name, props in SEGMENT_TABLE.items():
            assert set(props.keys()) == {
                "mass_frac",
                "length_frac",
                "radius_frac",
            }, f"{name} has unexpected keys"


class TestSegmentProperties:
    def test_returns_three_floats(self) -> None:
        mass, length, radius = segment_properties(80.0, 1.75, "pelvis")
        assert isinstance(mass, float)
        assert isinstance(length, float)
        assert isinstance(radius, float)

    def test_mass_scales_with_total(self) -> None:
        m1, _, _ = segment_properties(80.0, 1.75, "torso")
        m2, _, _ = segment_properties(100.0, 1.75, "torso")
        assert m2 / m1 == pytest.approx(100.0 / 80.0)

    def test_length_scales_with_height(self) -> None:
        _, l1, _ = segment_properties(80.0, 1.75, "thigh")
        _, l2, _ = segment_properties(80.0, 1.90, "thigh")
        assert l2 / l1 == pytest.approx(1.90 / 1.75)

    def test_rejects_zero_mass(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            segment_properties(0.0, 1.75, "pelvis")

    def test_rejects_zero_height(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            segment_properties(80.0, 0.0, "pelvis")

    def test_rejects_negative_mass(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            segment_properties(-10.0, 1.75, "pelvis")

    def test_unknown_segment_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            segment_properties(80.0, 1.75, "nonexistent_segment")

    def test_all_segments_produce_positive_values(self) -> None:
        for name in SEGMENT_TABLE:
            mass, length, radius = segment_properties(80.0, 1.75, name)
            assert mass > 0, f"{name} mass not positive"
            assert length > 0, f"{name} length not positive"
            assert radius > 0, f"{name} radius not positive"
