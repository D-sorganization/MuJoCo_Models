# SPDX-License-Identifier: MIT
"""Integration tests for headless MP4 video rendering (#412)."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.video import render_video

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_mujoco,
    pytest.mark.rendering,
]


def test_render_video_produces_valid_mp4(tmp_path: Path) -> None:
    """render_video renders N frames offscreen and writes a playable MP4."""
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY["gait"]().build())
    qpos0 = np.array(model.key_qpos[0] if model.nkey > 0 else model.qpos0)

    n_frames = 10
    # Small trajectory oscillating the root pelvis slightly
    trajectory = []
    for i in range(n_frames):
        q = qpos0.copy()
        q[2] += 0.01 * np.sin(i * 0.5)
        trajectory.append(q)

    out_file = tmp_path / "gait_test.mp4"
    result_path = render_video(
        "gait",
        trajectory,
        out_file,
        fps=10,
        width=320,
        height=240,
    )

    assert result_path.exists()
    assert result_path == out_file
    assert result_path.stat().st_size > 0

    # Read back and verify frame count and dimensions
    frames = iio.imread(str(result_path))
    assert len(frames) == n_frames
    assert frames[0].shape == (240, 320, 3)


def test_render_video_with_model_instance(tmp_path: Path) -> None:
    """render_video accepts an MjModel instance directly."""
    model = mujoco.MjModel.from_xml_string(EXERCISE_REGISTRY["squat"]().build())
    qpos = [model.key_qpos[0]] * 5

    out_file = tmp_path / "squat_test.mp4"
    render_video(model, qpos, out_file, fps=5, width=160, height=120)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_render_video_rejects_empty_trajectory(tmp_path: Path) -> None:
    """render_video raises ValueError on empty trajectory."""
    out_file = tmp_path / "empty.mp4"
    with pytest.raises(ValueError, match="trajectory"):
        render_video("gait", [], out_file)
