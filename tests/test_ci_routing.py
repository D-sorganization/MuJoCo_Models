"""Contracts for reversible public-repository CI routing."""

from pathlib import Path

import yaml

WORKFLOWS = Path(".github/workflows")
HOSTED_MODE = "CI_RUNNER_MODE != 'local'"
# Fork PRs go to a hosted runner first (RM#1989); the picker output is the fallback.
PICKED_RUNNER = (
    "${{ github.event.pull_request && "
    "github.event.pull_request.head.repo.full_name != github.repository && "
    "'ubuntu-latest' || needs.pick-runner.outputs.runner }}"
)


def _workflow(name: str) -> dict:
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    return yaml.load(text, Loader=yaml.BaseLoader)


def test_standard_ci_picker_is_zero_polling_and_reversible() -> None:
    workflow = _workflow("ci-standard.yml")
    picker = workflow["jobs"]["pick-runner"]
    scripts = "\n".join(step.get("run", "") for step in picker["steps"])

    assert HOSTED_MODE in picker["runs-on"]
    assert "gh api" not in scripts
    assert "sleep 15" not in scripts
    assert "runner=ubuntu-latest" in scripts


def test_lightweight_ci_is_hosted_eligible_but_rust_stays_local() -> None:
    jobs = _workflow("ci-standard.yml")["jobs"]

    assert jobs["quality-gate"]["runs-on"] == PICKED_RUNNER
    assert jobs["tests"]["runs-on"] == PICKED_RUNNER
    assert jobs["tests"]["strategy"]["max-parallel"] == "3"
    assert "d-sorg-fleet" in jobs["rust"]["runs-on"]


def test_lightweight_auxiliary_workflows_use_reversible_fast_lane() -> None:
    targets = {
        "anti-phantom-merge.yml": "guard",
        "cve-monitoring.yml": "check-expired",
        "lint-workflow-files.yml": "lint",
        "spec-check.yml": "spec-freshness",
        "Verify-Issue-Closure.yml": "verify-closure",
    }

    for workflow_name, job_name in targets.items():
        assert HOSTED_MODE in _workflow(workflow_name)["jobs"][job_name]["runs-on"]
