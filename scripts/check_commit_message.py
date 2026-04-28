"""Validate commit messages against this repository's conventional format."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOWED_TYPES = ("feat", "fix", "refactor", "test", "docs", "ci", "chore")
HEADER_PATTERN = re.compile(
    rf"^({'|'.join(ALLOWED_TYPES)})(\([a-z0-9._-]+\))?(!)?: [^\s].*"
)


def is_conventional_subject(subject: str) -> bool:
    """Return whether a commit subject follows the repository convention."""
    return bool(HEADER_PATTERN.fullmatch(subject.strip()))


def read_subject(message_path: Path) -> str:
    """Read the first non-comment line from a Git commit message file."""
    with message_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    return ""


def validate_commit_message(message_path: Path) -> tuple[bool, str]:
    """Validate the first commit-message line and return the result."""
    subject = read_subject(message_path)
    if is_conventional_subject(subject):
        return True, subject
    return False, subject


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Git commit message subject."
    )
    parser.add_argument("message_file", type=Path)
    args = parser.parse_args(argv)

    is_valid, subject = validate_commit_message(args.message_file)
    if is_valid:
        return 0

    allowed = ", ".join(ALLOWED_TYPES)
    sys.stderr.write(
        "Commit message must use conventional format:\n"
        "  type(scope): concise description\n"
        "  type!: concise description\n"
        f"Allowed types: {allowed}\n"
        f"Found: {subject or '<empty message>'}\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
