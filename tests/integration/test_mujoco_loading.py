"""Integration tests: verify MJCF models load into MuJoCo without errors."""

from typing import Any

import pytest

try:
    import mujoco

    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

_SKIP_MUJOCO = pytest.mark.skipif(
    not _MUJOCO_AVAILABLE,
    reason="mujoco not installed in this environment",
)

# These imports come after the try/except guard so that E402 is expected here.
from mujoco_models.exercises.bench_press.bench_press_model import (  # noqa: E402
    build_bench_press_model,
)
from mujoco_models.exercises.clean_and_jerk.clean_and_jerk_model import (  # noqa: E402
    build_clean_and_jerk_model,
)
from mujoco_models.exercises.deadlift.deadlift_model import (  # noqa: E402
    build_deadlift_model,
)
from mujoco_models.exercises.snatch.snatch_model import build_snatch_model  # noqa: E402
from mujoco_models.exercises.squat.squat_model import build_squat_model  # noqa: E402

ALL_BUILDERS = [
    ("squat", build_squat_model),
    ("bench_press", build_bench_press_model),
    ("deadlift", build_deadlift_model),
    ("snatch", build_snatch_model),
    ("clean_and_jerk", build_clean_and_jerk_model),
]
_IDS = [n for n, _ in ALL_BUILDERS]


@_SKIP_MUJOCO
class TestMuJoCoLoading:
    @pytest.mark.parametrize("name,builder", ALL_BUILDERS, ids=_IDS)
    def test_model_loads_without_error(self, name: str, builder: Any) -> None:
        xml_str = builder()
        model = mujoco.MjModel.from_xml_string(xml_str)
        assert model.nbody > 1, f"{name}: no bodies after loading"

    @pytest.mark.parametrize("name,builder", ALL_BUILDERS, ids=_IDS)
    def test_model_has_correct_joint_count(self, name: str, builder: Any) -> None:
        xml_str = builder()
        model = mujoco.MjModel.from_xml_string(xml_str)
        # 14 hinge joints: lumbar, neck, 2 shoulder, 2 elbow, 2 wrist,
        # 2 hip, 2 knee, 2 ankle
        assert model.njnt >= 14, f"{name}: expected >= 14 joints, got {model.njnt}"
