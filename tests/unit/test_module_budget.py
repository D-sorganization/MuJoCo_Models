"""Enforce CLAUDE.md module budget: max 300 lines per file in src/mujoco_models/."""

from __future__ import annotations

from pathlib import Path

MAX_LINES = 300
SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mujoco_models"
VENDORED_DIRS = {
    SRC_ROOT / "shared" / "parity" / "_canonical",
}


def _is_vendored(path: Path) -> bool:
    return any(v in path.parents or path == v for v in VENDORED_DIRS)


def test_src_files_under_300_line_budget() -> None:
    """Every non-vendored python file under src/mujoco_models/ must have <= 300 lines."""
    over_budget: dict[str, int] = {}
    for py_path in sorted(SRC_ROOT.rglob("*.py")):
        if _is_vendored(py_path):
            continue
        line_count = len(py_path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_LINES:
            rel = str(py_path.relative_to(SRC_ROOT.parent))
            over_budget[rel] = line_count

    assert not over_budget, (
        f"Files in src/ exceed the {MAX_LINES}-line module budget:\n"
        + "\n".join(f"  {path}: {count} lines" for path, count in over_budget.items())
    )
