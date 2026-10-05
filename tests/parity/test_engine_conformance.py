# SPDX-License-Identifier: MIT
"""Real-engine conformance of every exercise model (issue #390).

Builds each exercise in the REAL MuJoCo engine and checks the resulting
fingerprint against the vendored fleet parity standard.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import mujoco
import pytest

from mujoco_models.exercises import EXERCISE_REGISTRY
from mujoco_models.model_pack import list_exercises, manifest
from mujoco_models.shared.parity import fingerprint as fp_mod
from mujoco_models.shared.parity._canonical import conformance
from mujoco_models.shared.parity.fingerprint import fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = REPO_ROOT / "src" / "mujoco_models" / "shared" / "parity"
LEDGER = PARITY_DIR / "parity_divergences.json"
STD = conformance.load_standard()
EXERCISES = list_exercises()
N_HUMAN_HINGES = len(conformance.expected_coordinates(STD))


@pytest.mark.parametrize("exercise", EXERCISES)
def test_exercise_loads_in_real_engine(exercise: str) -> None:
    xml = EXERCISE_REGISTRY[exercise]().build()
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nbody >= 1 + 15
    hinges = sum(
        1 for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    )
    assert hinges == N_HUMAN_HINGES == 28
    assert model.nkey == 1
    assert fingerprint(exercise)["loaded_in_engine"] is True


@pytest.mark.parametrize("exercise", EXERCISES)
def test_exercise_conforms_to_standard(exercise: str) -> None:
    fp = fingerprint(exercise)
    divs = conformance.check_fingerprint(fp, STD)
    unexpected, _ = conformance.reconcile(divs, conformance.load_ledger(LEDGER))
    assert not unexpected, [(d.key, d.message) for d in unexpected]


def test_ledger_has_no_stale_entries() -> None:
    """The ledger only ratchets down: an entry matching no divergence is stale."""
    divs = [
        d
        for ex in EXERCISES
        for d in conformance.check_fingerprint(fingerprint(ex), STD)
    ]
    _, stale = conformance.reconcile(divs, conformance.load_ledger(LEDGER))
    assert not stale, f"stale ledger entries: {stale}"


@pytest.mark.parametrize("exercise", ["gait", "sit_to_stand"])
def test_standing_grf_matches_weight(exercise: str) -> None:
    fp = fingerprint(exercise)
    rel = abs(fp["standing_vertical_grf_n"] - fp["standing_weight_n"])
    assert rel / fp["standing_weight_n"] <= STD["tolerances"]["standing_grf_rel"]


def test_ledger_never_excuses_load_failures() -> None:
    entries = json.loads(LEDGER.read_text(encoding="utf-8"))["divergences"]
    assert not [k for k in entries if "load_in_engine" in k]


def test_unknown_exercise_rejected() -> None:
    with pytest.raises(ValueError):
        fingerprint("not_an_exercise")


def test_cli_writes_fingerprint(tmp_path: Path) -> None:
    assert fp_mod.main(["--exercise", "gait", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "mujoco_gait.json").read_text("utf-8"))
    assert data["schema"] == conformance.FINGERPRINT_SCHEMA


def test_standard_json_ships_as_package_data() -> None:
    pkg = files("mujoco_models.shared.parity._canonical")
    assert pkg.joinpath(conformance.STANDARD_FILENAME).is_file()


def test_vendored_files_match_manifest() -> None:
    canon = PARITY_DIR / "_canonical"
    recorded = json.loads((canon / "MANIFEST.json").read_text("utf-8"))["files"]
    for name, digest in recorded.items():
        actual = hashlib.sha256((canon / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} was edited locally; re-vendor it"


def test_capabilities_block_is_complete_and_honest() -> None:
    caps = manifest()["capabilities"]
    assert list(caps) == STD["capabilities"]["keys"]
    for key, entry in caps.items():
        assert entry["level"] in STD["capabilities"]["levels"], key
        if entry["evidence"] is not None:
            assert (REPO_ROOT / entry["evidence"]).exists(), key
    assert caps["load_in_engine"]["level"] == "full"
