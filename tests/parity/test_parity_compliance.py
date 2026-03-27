"""Cross-repo parity compliance tests.

These tests verify that this repo's models match the canonical standard.
Identical tests exist in Drake, Pinocchio, and OpenSim repos.
"""

import pytest

from mujoco_models.shared.body.segment_data import SEGMENT_TABLE
from mujoco_models.shared.parity.standard import (
    EXERCISE_PHASE_COUNTS,
    FOOT_CONTACT_DIMS,
    GRAVITY,
    JOINT_LIMITS,
    MENS_BARBELL,
    SEGMENT_MASS_FRACTIONS,
    STANDARD_BODY_MASS,
    STANDARD_HEIGHT,
)


class TestAnthropometricParity:
    def test_default_body_mass(self) -> None:
        assert STANDARD_BODY_MASS == 80.0

    def test_default_height(self) -> None:
        assert STANDARD_HEIGHT == 1.75

    def test_segment_mass_fractions_sum(self) -> None:
        # Central + bilateral*2 should be ~1.0
        total = sum(
            v
            for k, v in SEGMENT_MASS_FRACTIONS.items()
            if k in ("pelvis", "torso", "head")
        )
        total += 2 * sum(
            v
            for k, v in SEGMENT_MASS_FRACTIONS.items()
            if k not in ("pelvis", "torso", "head")
        )
        assert abs(total - 1.0) < 0.02  # Allow 2% tolerance

    def test_all_segments_present(self) -> None:
        required = {
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
        assert required == set(SEGMENT_MASS_FRACTIONS.keys())


class TestJointLimitParity:
    @pytest.mark.parametrize("joint", list(JOINT_LIMITS.keys()))
    def test_limits_are_tuples(self, joint: str) -> None:
        lo, hi = JOINT_LIMITS[joint]
        assert lo < hi

    def test_all_joints_present(self) -> None:
        assert len(JOINT_LIMITS) == 16


class TestBarbellParity:
    def test_mens_bar_mass(self) -> None:
        assert MENS_BARBELL["bar_mass"] == 20.0

    def test_mens_total_length(self) -> None:
        assert MENS_BARBELL["total_length"] == 2.20


class TestContactParity:
    def test_foot_dims(self) -> None:
        assert FOOT_CONTACT_DIMS["length"] == 0.26


class TestExerciseParity:
    @pytest.mark.parametrize("exercise,count", EXERCISE_PHASE_COUNTS.items())
    def test_phase_counts(self, exercise: str, count: int) -> None:
        assert count >= 5


class TestSegmentDataCrossCheck:
    """Verify parity standard SEGMENT_MASS_FRACTIONS matches segment_data.SEGMENT_TABLE."""

    def test_same_segments(self) -> None:
        assert set(SEGMENT_MASS_FRACTIONS.keys()) == set(SEGMENT_TABLE.keys())

    @pytest.mark.parametrize("segment", sorted(SEGMENT_MASS_FRACTIONS.keys()))
    def test_mass_fractions_match(self, segment: str) -> None:
        parity_val = SEGMENT_MASS_FRACTIONS[segment]
        table_val = SEGMENT_TABLE[segment]["mass_frac"]
        assert parity_val == pytest.approx(table_val), (
            f"Parity standard {segment} mass_frac={parity_val} "
            f"!= segment_data {segment} mass_frac={table_val}"
        )


class TestGravityParity:
    def test_gravity_z_down(self) -> None:
        assert GRAVITY[2] == pytest.approx(-9.80665)
