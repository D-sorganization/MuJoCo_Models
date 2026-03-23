# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email the maintainers with a description of the vulnerability.
3. Include steps to reproduce the issue if possible.
4. Allow reasonable time for the issue to be addressed before public disclosure.

## Security Measures

This project uses the following security practices:

- **Dependency auditing**: `pip-audit` runs in CI to detect known vulnerabilities.
- **Static analysis**: `bandit` scans source code for common security issues.
- **Pinned CI actions**: All GitHub Actions use specific versions.
- **Minimal permissions**: CI workflows use `contents: read` by default.
- **No secrets in code**: The CI checks for placeholder patterns and hardcoded credentials.
