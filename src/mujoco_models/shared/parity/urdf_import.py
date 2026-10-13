# SPDX-License-Identifier: MIT
"""Cross-engine URDF import and fingerprinting for Pinocchio_Models (issue #411).

Provides functions to load Pinocchio_Models URDFs in MuJoCo and compute
comparable model fingerprints (total mass, segment masses, COM, joint count).
"""

from __future__ import annotations

import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco

logger = logging.getLogger(__name__)

# Canonical human segments in parity standard
EXPECTED_SEGMENTS: tuple[str, ...] = (
    "pelvis",
    "torso",
    "head",
    "upper_arm_l",
    "forearm_l",
    "hand_l",
    "upper_arm_r",
    "forearm_r",
    "hand_r",
    "thigh_l",
    "shank_l",
    "foot_l",
    "thigh_r",
    "shank_r",
    "foot_r",
)

# Minimal inertia value required by MuJoCo for moving bodies
_MIN_INERTIA = "1e-6"


def resolve_pinocchio_urdf(exercise: str, repo_root: Path | None = None) -> str | None:
    """Resolve and return the URDF XML string for a Pinocchio_Models exercise.

    Checks the following sources in order:
    1. Direct file via ``PINOCCHIO_URDF_PATH`` env var.
    2. Directory of URDFs via ``PINOCCHIO_URDF_DIR`` env var.
    3. Sibling repository ``Pinocchio_Models`` in the fleet workspace.
    4. Installed ``pinocchio_models`` package.

    Returns None if the URDF is unavailable.
    """
    # 1. Direct file
    env_file = os.environ.get("PINOCCHIO_URDF_PATH")
    if env_file and Path(env_file).is_file():
        return Path(env_file).read_text(encoding="utf-8")

    # 2. Directory
    env_dir = os.environ.get("PINOCCHIO_URDF_DIR")
    if env_dir:
        candidate = Path(env_dir) / f"{exercise}.urdf"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")

    # 3. Sibling repo in fleet workspace
    root = repo_root or Path(__file__).resolve().parents[4]
    sibling = root.parent / "Pinocchio_Models"
    if not sibling.exists():
        sibling = root / "Pinocchio_Models"

    if sibling.exists():
        src_path = str(sibling / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    # 4. pinocchio_models package import
    try:
        from pinocchio_models.shared.parity.fingerprint import build_urdf

        return build_urdf(exercise)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not build Pinocchio URDF for %s: %s", exercise, exc)
        return None


def sanitize_urdf_for_mujoco(urdf_str: str) -> str:
    """Prepare a Pinocchio URDF string so MuJoCo's parser can ingest it cleanly.

    1. Adds a virtual ``world`` link and floating root joint to the root link
       (or fixed joint if the root link is a fixture like ``bench``), preventing
       MuJoCo from absorbing the root segment into the world body.
    2. Ensures moving virtual links have inertia >= mjMINVAL so MuJoCo doesn't
       reject them.
    """
    root = ET.fromstring(urdf_str)

    child_links: set[str] = set()
    for j in root.findall("joint"):
        child_el = j.find("child")
        if child_el is not None:
            c_name = child_el.get("link")
            if c_name:
                child_links.add(c_name)
    root_links = [
        link.get("name", "")
        for link in root.findall("link")
        if link.get("name") and link.get("name") not in child_links
    ]

    if root_links and "world" not in [
        link.get("name") for link in root.findall("link")
    ]:
        primary_root = root_links[0]
        world_link = ET.Element("link", {"name": "world"})
        rj_type = "fixed" if primary_root == "bench" else "floating"
        root_joint = ET.Element("joint", {"name": "world_root_joint", "type": rj_type})
        ET.SubElement(root_joint, "parent", {"link": "world"})
        ET.SubElement(root_joint, "child", {"link": primary_root})
        root.insert(0, world_link)
        root.insert(1, root_joint)

    # Ensure non-zero inertia on all inertial blocks
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            if inertia is not None:
                for axis in ("ixx", "iyy", "izz"):
                    val = float(inertia.get(axis, 0.0))
                    if val <= 1e-12:
                        inertia.set(axis, _MIN_INERTIA)

    return ET.tostring(root, encoding="unicode")


def load_pinocchio_urdf(urdf_source: str | Path) -> mujoco.MjModel:
    """Load a Pinocchio_Models URDF into a real MuJoCo MjModel.

    Args:
        urdf_source: Either a path to a URDF file or a raw URDF XML string.

    Returns:
        The loaded MuJoCo MjModel.
    """
    if isinstance(urdf_source, Path) or (
        isinstance(urdf_source, str)
        and "\n" not in urdf_source
        and Path(urdf_source).is_file()
    ):
        raw_xml = Path(urdf_source).read_text(encoding="utf-8")
    else:
        raw_xml = str(urdf_source)

    sanitized_xml = sanitize_urdf_for_mujoco(raw_xml)
    return mujoco.MjModel.from_xml_string(sanitized_xml)


def pinocchio_urdf_fingerprint(
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
) -> dict[str, Any]:
    """Compute the cross-engine fingerprint for a MuJoCo model loaded from URDF.

    Returns a dict containing:
    - ``total_mass_kg``: total mass of all bodies
    - ``human_mass_kg``: sum of masses of canonical anatomical segments
    - ``segment_masses_kg``: mapping of segment name to mass in kg
    - ``joint_count``: number of hinge joints (human DOFs)
    - ``joint_names``: names of all hinge joints
    - ``com_world_m``: world COM at neutral pose
    - ``com_rel_pelvis_m``: COM relative to the pelvis origin
    - ``segment_origins_rel_pelvis_m``: segment origins relative to pelvis
    """
    if data is None:
        data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Segment masses (filtering out virtual links)
    segment_masses: dict[str, float] = {}
    for b in range(model.nbody):
        bname = model.body(b).name
        if bname in EXPECTED_SEGMENTS:
            segment_masses[bname] = float(model.body_mass[b])

    human_mass = sum(segment_masses.values())
    total_mass = float(sum(model.body_mass))

    # Hinge joints
    hinge_joints: list[str] = [
        model.joint(j).name
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]

    # Kinematic origins relative to pelvis
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id >= 0:
        pelvis_pos = data.xpos[pelvis_id]
        com_rel = (data.subtree_com[0] - pelvis_pos).tolist()
        origins_rel: dict[str, list[float]] = {
            model.body(b).name: (data.xpos[b] - pelvis_pos).tolist()
            for b in range(model.nbody)
            if model.body(b).name in EXPECTED_SEGMENTS
        }
    else:
        com_rel = data.subtree_com[0].tolist()
        origins_rel = {
            model.body(b).name: data.xpos[b].tolist()
            for b in range(model.nbody)
            if model.body(b).name in EXPECTED_SEGMENTS
        }

    return {
        "total_mass_kg": total_mass,
        "human_mass_kg": human_mass,
        "segment_masses_kg": segment_masses,
        "joint_count": len(hinge_joints),
        "joint_names": hinge_joints,
        "com_world_m": data.subtree_com[0].tolist(),
        "com_rel_pelvis_m": com_rel,
        "segment_origins_rel_pelvis_m": origins_rel,
    }
