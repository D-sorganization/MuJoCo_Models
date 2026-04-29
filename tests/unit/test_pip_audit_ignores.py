"""Tests for pip-audit CVE exception tracking file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
IGNORES_PATH = REPO_ROOT / "docs" / "security" / "pip_audit_ignores.yml"

REQUIRED_FIELDS = {
    "cve",
    "reason",
    "expires",
    "tracking_issue",
    "remediation_status",
    "package",
}
VALID_STATUSES = {"monitoring", "accepted_risk", "patched", "false_positive"}


@pytest.fixture
def ignores_data() -> dict:
    """Load pip_audit_ignores.yml."""
    with IGNORES_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


class TestPipAuditIgnores:
    """Validate the structure and content of pip_audit_ignores.yml."""

    def test_file_exists(self) -> None:
        assert IGNORES_PATH.exists(), f"{IGNORES_PATH} must exist"

    def test_top_level_key(self, ignores_data: dict) -> None:
        assert "exceptions" in ignores_data, "root key 'exceptions' required"
        assert isinstance(ignores_data["exceptions"], list), "exceptions must be a list"

    def test_entry_count(self, ignores_data: dict) -> None:
        assert len(ignores_data["exceptions"]) == 7, "expected 7 tracked CVE exceptions"

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param(e, id=e.get("cve", "unknown"))
            for e in (yaml.safe_load(IGNORES_PATH.read_text()) or {}).get(
                "exceptions", []
            )
        ],
    )
    def test_required_fields(self, entry: dict) -> None:
        missing = REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"missing fields: {missing}"

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param(e, id=e.get("cve", "unknown"))
            for e in (yaml.safe_load(IGNORES_PATH.read_text()) or {}).get(
                "exceptions", []
            )
        ],
    )
    def test_cve_id_format(self, entry: dict) -> None:
        cve = entry["cve"]
        assert cve.startswith("CVE-"), f"{cve} must start with CVE-"
        parts = cve.split("-")
        assert len(parts) == 3, f"{cve} must be CVE-YYYY-NNNNN format"

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param(e, id=e.get("cve", "unknown"))
            for e in (yaml.safe_load(IGNORES_PATH.read_text()) or {}).get(
                "exceptions", []
            )
        ],
    )
    def test_expiration_date_format(self, entry: dict) -> None:
        exp = entry["expires"]
        assert len(exp) == 10 and exp[4] == "-" and exp[7] == "-", (
            f"{exp} must be YYYY-MM-DD"
        )

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param(e, id=e.get("cve", "unknown"))
            for e in (yaml.safe_load(IGNORES_PATH.read_text()) or {}).get(
                "exceptions", []
            )
        ],
    )
    def test_remediation_status_valid(self, entry: dict) -> None:
        status = entry["remediation_status"]
        assert status in VALID_STATUSES, f"{status} not in {VALID_STATUSES}"

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param(e, id=e.get("cve", "unknown"))
            for e in (yaml.safe_load(IGNORES_PATH.read_text()) or {}).get(
                "exceptions", []
            )
        ],
    )
    def test_tracking_issue_format(self, entry: dict) -> None:
        issue = entry["tracking_issue"]
        assert issue.startswith("MuJoCo_Models#176-"), (
            f"{issue} must reference MuJoCo_Models#176-N"
        )
