# Criterion G: Security

**Score:** 7.0 / 10.0

## Findings

### G-001: 6 CVEs ignored in CI without remediation plan
- **Priority:** P1
- **File:** `.github/workflows/ci-standard.yml`
- **Lines:** 94, 95, 96, 97, 98, 99, 100, 101, 102, 103
- **Description:** pip-audit configuration ignores 6 known vulnerabilities.

### G-002: No SBOM generation
- **Priority:** P2
- **File:** `.github/workflows/ci-standard.yml`
- **Description:** No Software Bill of Materials generated or published.

