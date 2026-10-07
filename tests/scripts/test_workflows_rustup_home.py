"""Rust toolchain jobs on the fleet must not share ``~/.rustup`` (RM#2021)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
RUST_MARKERS = re.compile(
    r"dtolnay/rust-toolchain|actions-rs/toolchain|\brustup\b|\bcargo\b|\bmaturin\b"
)
ISOLATED_HOME = "${{ github.workspace }}/.rustup-home"


def _uses_rust(job: dict[str, Any]) -> bool:
    """Return True when any step of ``job`` installs or runs a Rust toolchain."""
    return any(
        RUST_MARKERS.search(f"{step.get('uses', '')}\n{step.get('run', '')}")
        for step in job.get("steps", [])
    )


def _rustup_home(workflow: dict[str, Any], job: dict[str, Any]) -> str | None:
    """Return the effective RUSTUP_HOME (job env overrides workflow env)."""
    for scope in (job, workflow):
        env = scope.get("env") or {}
        if "RUSTUP_HOME" in env:
            return str(env["RUSTUP_HOME"])
    return None


def test_every_rust_toolchain_job_uses_an_isolated_rustup_home() -> None:
    """RM#2021: one job's toolchain update must not delete rustc under another."""
    offenders: list[str] = []
    rust_jobs = 0
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, job in (workflow.get("jobs") or {}).items():
            if not _uses_rust(job):
                continue
            rust_jobs += 1
            if _rustup_home(workflow, job) != ISOLATED_HOME:
                offenders.append(f"{path.name}:{name}")
    assert rust_jobs, "expected at least one Rust job (rust_core)"
    assert offenders == []
