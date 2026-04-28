from pathlib import Path

from scripts.check_commit_message import (
    is_conventional_subject,
    validate_commit_message,
)


def test_is_conventional_subject_accepts_repo_types() -> None:
    assert is_conventional_subject("docs: add contribution checklist")
    assert is_conventional_subject("fix(parser): reject invalid coordinates")
    assert is_conventional_subject("feat!: remove deprecated model alias")


def test_is_conventional_subject_rejects_non_conventional_subjects() -> None:
    assert not is_conventional_subject("update docs")
    assert not is_conventional_subject("feature: add model")
    assert not is_conventional_subject("docs:add missing space")


def test_validate_commit_message_reads_first_non_comment_line(tmp_path: Path) -> None:
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_text(
        "# Please enter the commit message\n\nchore: update project governance\n",
        encoding="utf-8",
    )

    is_valid, subject = validate_commit_message(message_file)

    assert is_valid
    assert subject == "chore: update project governance"
