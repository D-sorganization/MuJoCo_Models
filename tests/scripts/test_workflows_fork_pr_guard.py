"""This repo's real workflows never run fork PR code on the self-hosted fleet."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import fork_pr_runner_guard as guard

pytestmark = pytest.mark.unit

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def test_repo_workflows_have_no_fork_pr_fleet_violations() -> None:
    """RM#1989: every fleet-capable job is fork-guarded."""
    assert guard.find_violations(WORKFLOWS) == []
