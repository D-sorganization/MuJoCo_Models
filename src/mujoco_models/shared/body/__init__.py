# SPDX-License-Identifier: MIT
"""Full-body musculoskeletal model for MuJoCo MJCF.

Provides a simplified but anatomically grounded full-body model with
major body segments and joints suitable for barbell exercise simulation.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization

from mujoco_models.shared.body.body_anthropometrics import (
    BodyModelSpec,
)
from mujoco_models.shared.body.body_model import (
    create_full_body,
)
from mujoco_models.shared.body.segment_data import (
    SEGMENT_TABLE,
    segment_properties,
    total_mass_fraction,
)

__all__ = [
    "BodyModelSpec",
    "SEGMENT_TABLE",
    "create_full_body",
    "segment_properties",
    "total_mass_fraction",
]
