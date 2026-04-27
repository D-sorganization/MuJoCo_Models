# SPDX-License-Identifier: MIT
"""Tests for ExerciseConfig validation and defaults (#21).

Ensures that ExerciseConfig correctly constructs with defaults,
custom values, and rejects invalid inputs through its composed specs.
"""

from __future__ import annotations

import pytest

from mujoco_models.exercises.base import ExerciseConfig
from mujoco_models.shared.barbell import BarbellSpec
from mujoco_models.shared.body import BodyModelSpec


class TestExerciseConfigDefaults:
    def test_default_gravity(self) -> None:
        config = ExerciseConfig()
        assert config.gravity == (0.0, 0.0, -9.80665)

    def test_default_body_spec(self) -> None:
        config = ExerciseConfig()
        assert config.body_spec.total_mass == 80.0  # type: ignore
        assert config.body_spec.height == 1.75

    def test_default_barbell_spec(self) -> None:
        config = ExerciseConfig()
        assert config.barbell_spec.bar_mass == 20.0  # type: ignore
        assert config.barbell_spec.plate_mass_per_side == 0.0

    def test_custom_body_spec(self) -> None:
        spec = BodyModelSpec(total_mass=100.0, height=1.90)
        config = ExerciseConfig(body_spec=spec)
        assert config.body_spec.total_mass == 100.0  # type: ignore

    def test_custom_barbell_spec(self) -> None:
        spec = BarbellSpec.womens_olympic(plate_mass_per_side=30.0)
        config = ExerciseConfig(barbell_spec=spec)
        assert config.barbell_spec.bar_mass == 15.0  # type: ignore
        assert config.barbell_spec.plate_mass_per_side == 30.0

    def test_custom_gravity(self) -> None:
        config = ExerciseConfig(gravity=(0.0, 0.0, -1.62))
        assert config.gravity[2] == pytest.approx(-1.62)


class TestExerciseConfigValidation:
    def test_rejects_negative_body_mass(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            ExerciseConfig(body_spec=BodyModelSpec(total_mass=-1.0))

    def test_rejects_zero_body_height(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            ExerciseConfig(body_spec=BodyModelSpec(height=0.0))

    def test_rejects_negative_plate_mass(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            ExerciseConfig(
                barbell_spec=BarbellSpec(plate_mass_per_side=-5.0),
            )

    def test_rejects_shaft_longer_than_total(self) -> None:
        with pytest.raises(ValueError, match="shaft_length"):
            ExerciseConfig(
                barbell_spec=BarbellSpec(total_length=1.0, shaft_length=1.5),
            )


class TestExerciseConfigComposition:
    def test_total_barbell_mass(self) -> None:
        config = ExerciseConfig(
            barbell_spec=BarbellSpec.mens_olympic(plate_mass_per_side=60.0),
        )
        assert config.barbell_spec.total_mass == pytest.approx(140.0)  # type: ignore

    def test_womens_bar_total_mass(self) -> None:
        config = ExerciseConfig(
            barbell_spec=BarbellSpec.womens_olympic(plate_mass_per_side=40.0),
        )
        assert config.barbell_spec.total_mass == pytest.approx(95.0)  # type: ignore
