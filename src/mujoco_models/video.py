# SPDX-License-Identifier: MIT
# Copyright (c) 2026 D-sorganization
"""Headless offscreen video rendering for MuJoCo exercise rollouts (#412).

Renders trajectories of joint poses to MP4 video using `mujoco.Renderer`
and `imageio` with ffmpeg encoding.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from mujoco_models.exercises import EXERCISE_REGISTRY

logger = logging.getLogger(__name__)


def _get_model(exercise: str | mujoco.MjModel) -> mujoco.MjModel:
    """Resolve an exercise name or model instance to an MjModel."""
    if isinstance(exercise, mujoco.MjModel):
        return exercise
    if exercise not in EXERCISE_REGISTRY:
        raise ValueError(
            f"unknown exercise {exercise!r}; available: {sorted(EXERCISE_REGISTRY)}"
        )
    builder = EXERCISE_REGISTRY[exercise]()
    return mujoco.MjModel.from_xml_string(builder.build())


def render_video(
    exercise: str | mujoco.MjModel,
    trajectory: Sequence[np.ndarray | Sequence[float]] | np.ndarray,
    output_path: str | Path,
    *,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera: str | int | None = None,
) -> Path:
    """Render a sequence of poses (qpos) to an MP4 video file offscreen.

    Args:
        exercise: Canonical exercise name or an MjModel instance.
        trajectory: Sequence of qpos vectors representing the motion rollout.
        output_path: Path where the rendered MP4 file will be saved.
        fps: Playback frame rate in frames per second (default: 30).
        width: Video frame width in pixels (default: 640).
        height: Video frame height in pixels (default: 480).
        camera: Optional camera name or camera ID to render from.

    Returns:
        Path to the saved MP4 file.

    Raises:
        ValueError: If trajectory is empty or exercise name is unknown.
        ImportError: If imageio or imageio-ffmpeg is not installed.
    """
    if len(trajectory) == 0:
        raise ValueError("trajectory must contain at least one frame")

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "Video rendering requires imageio and imageio-ffmpeg. "
            "Install with: pip install 'imageio[ffmpeg]'"
        ) from exc

    model = _get_model(exercise)
    data = mujoco.MjData(model)

    logger.info(
        "Rendering %d frames at %dx%d (%d fps) to %s",
        len(trajectory),
        width,
        height,
        fps,
        output_path,
    )

    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: list[np.ndarray] = []

    try:
        for qpos in trajectory:
            data.qpos[:] = np.asarray(qpos, dtype=float)
            mujoco.mj_forward(model, data)
            if camera is not None:
                renderer.update_scene(data, camera=camera)
            else:
                renderer.update_scene(data)
            frames.append(renderer.render())
    finally:
        renderer.close()

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(out), np.stack(frames, axis=0), fps=fps)

    logger.debug("Successfully saved video (%d bytes) to %s", out.stat().st_size, out)
    return out
