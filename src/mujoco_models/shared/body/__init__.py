"""Full-body musculoskeletal model for MuJoCo MJCF.

Provides a simplified but anatomically grounded full-body model with
major body segments and joints suitable for barbell exercise simulation.
"""

from mujoco_models.shared.body.body_model import (
    BodyModelSpec,
    create_full_body,
)

__all__ = ["BodyModelSpec", "create_full_body"]
